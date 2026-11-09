"""Collector base: shared HTTP behavior, pagination loop, dedup, persistence.

Collectors are mode-agnostic: the same code produces identical output formats
whether run locally (``--mode local``) or on GitHub Actions (``--mode ci``);
the manifest records which mode produced it. Quantities (pages, target
listings) are always explicit parameters so that small local tests and full
cloud runs differ only in scale, never in code or format.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from icmi.core.config import PipelineConfig, SourceConfig
from icmi.core.manifest import QualityReport, RunManifest
from icmi.core.records import CanonicalRecord, RawPayloadRef, utc_now
from icmi.core.storage import RawWriter


def _retryable(exc: BaseException) -> bool:
    """Retry network errors and transient HTTP statuses, honor 429/5xx."""
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(exc, TransientHTTPError):
        return True
    return False


class TransientHTTPError(Exception):
    """HTTP status worth retrying (429 or 5xx)."""

    def __init__(self, status_code: int, body_head: str) -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {body_head[:200]!r}")


class CollectorResult:
    """Everything one collection run produced."""

    def __init__(self, manifest: RunManifest, records: list[CanonicalRecord]) -> None:
        self.manifest = manifest
        self.records = records


class BaseCollector(ABC):
    """Shared machinery for per-platform collectors.

    Subclasses implement :meth:`iter_ads` (extract ad payloads from one page's
    JSON) and :meth:`page_params` (translate a page index into request
    params). Everything else — retries, rate limiting, dedup, persistence,
    manifest — is handled here.
    """

    def __init__(
        self,
        source: SourceConfig,
        pipeline: PipelineConfig,
        run_id: str,
        execution_mode: str = "local",
        collection: str = "full",
    ) -> None:
        self.source = source
        self.pipeline = pipeline
        self.run_id = run_id
        self.execution_mode = execution_mode
        self.collection = collection
        self.collection_params = source.collection_params(collection)
        # Stable identifier for filesystem/manifest use; collectors may
        # override with the config key (class attribute source_key).
        self.source_key = getattr(self, "source_key", None) or source.display_name
        self.writer = RawWriter(run_id, platform=self.source_key)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": pipeline.user_agent,
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "fa-IR,fa;q=0.9,en;q=0.8",
            }
        )
        self.failed_requests: list[str] = []
        self.pages: dict[str, int] = {
            "requested": 0,
            "fetched": 0,
            "failed": 0,
            "empty": 0,
            "parse_failed": 0,
        }
        self.skip_reasons: dict[str, int] = {}

    # --- per-platform hooks -------------------------------------------------

    @abstractmethod
    def page_params(self, page_index: int) -> dict[str, Any]:
        """Request params for one page (platform pagination semantics)."""

    @abstractmethod
    def iter_ads(self, page_payload: dict) -> list[dict]:
        """Ad payload dicts from one page response (empty list = empty page)."""

    @abstractmethod
    def parse_listing(self, ad: dict, payload_ref: RawPayloadRef) -> CanonicalRecord | None:
        """Map one raw ad payload to a canonical record (None = skip, with reason)."""

    @classmethod
    def parse_only(cls, source: SourceConfig, pipeline: PipelineConfig) -> BaseCollector:
        """Extraction-only instance (no HTTP session, no writer, no I/O).

        Used by the ETL to re-derive canonical records from raw payloads
        deterministically.
        """
        obj = cls.__new__(cls)
        obj.source = source
        obj.pipeline = pipeline
        obj.source_key = getattr(obj, "source_key", None) or source.display_name
        obj.collection = "full"
        obj.collection_params = {}
        obj.skip_reasons = {}
        return obj

    def observe_page(self, page_payload: dict) -> None:
        """Hook: inspect a fetched page (e.g. capture the next pagination cursor).

        Called once per successfully fetched page, before extraction.
        """
        return None

    def has_more(self, page_payload: dict) -> bool:
        """Whether the source indicates more pages after this one.

        Defaults to True (keep paging until max_pages/empties); platforms
        exposing pagination metadata should override to stop correctly.
        """
        return True

    # --- HTTP ---------------------------------------------------------------

    @retry(
        retry=retry_if_exception(_retryable),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        reraise=True,
    )
    def _fetch_json(self, params: dict[str, Any]) -> dict:
        self.pages["requested"] += 1
        response = self.session.get(
            self.source.base_url,
            params=params,
            timeout=self.pipeline.request_timeout_seconds,
        )
        if response.status_code == 429 or 500 <= response.status_code < 600:
            raise TransientHTTPError(response.status_code, response.text)
        response.raise_for_status()
        self.pages["fetched"] += 1
        return response.json()

    def _polite_wait(self) -> None:
        delay = self.pipeline.request_delay_seconds
        jitter = self.pipeline.request_jitter_seconds
        if delay:
            time.sleep(delay + random.uniform(0, jitter))

    # --- main loop -----------------------------------------------------------

    def collect(self, target_listings: int, max_pages: int) -> CollectorResult:
        """Collect up to ``target_listings`` unique listings over at most
        ``max_pages`` pages. Stops early on repeated empty pages or hard
        failures; never silently truncates without recording it."""
        started_at = utc_now()
        records: list[CanonicalRecord] = []
        seen_ids: set[str] = set()
        raw_entries: list[dict] = []
        duplicates = 0
        consecutive_empty = 0

        for page_index in range(max_pages):
            try:
                payload = self._fetch_json(self.page_params(page_index))
                self.writer.write_page(page_index, payload)
            except requests.HTTPError as exc:
                self.pages["failed"] += 1
                self.failed_requests.append(f"page {page_index}: {exc}")
                if exc.response is not None and exc.response.status_code in (401, 403):
                        self.skip_reasons["access_forbidden"] = (
                            self.skip_reasons.get("access_forbidden", 0) + 1
                        )
                        break  # blocking: stop instead of hammering
                continue
            except (requests.ConnectionError, requests.Timeout, TransientHTTPError) as exc:
                self.pages["failed"] += 1
                self.failed_requests.append(f"page {page_index}: {type(exc).__name__}: {exc}")
                continue

            self.observe_page(payload)
            ads = self.iter_ads(payload)
            new_before = len(records)

            for ad in ads:
                listing_id = self.listing_id_of(ad)
                if not listing_id:
                    self.skip_reasons["no_listing_id"] = (
                        self.skip_reasons.get("no_listing_id", 0) + 1
                    )
                    continue
                if listing_id in seen_ids:
                    duplicates += 1
                    continue
                seen_ids.add(listing_id)
                ref = RawPayloadRef(
                    run_id=self.run_id,
                    platform=self.source_key,
                    file="records.jsonl",
                    key=listing_id,
                )
                record = self.parse_listing(ad, ref)
                if record is None:
                    self.skip_reasons["skipped_records"] = (
                        self.skip_reasons.get("skipped_records", 0) + 1
                    )
                    continue
                records.append(record)
                raw_entries.append({"source_listing_id": listing_id, "raw_payload": ad})
                if len(records) >= target_listings:
                    break

            # A page yielding zero NEW records is stale (repeated result set,
            # e.g. a broken cursor chain); it counts toward the same stop
            # threshold as empty pages.
            if len(records) == new_before:
                if ads:
                    self.pages["stale"] = self.pages.get("stale", 0) + 1
                else:
                    self.pages["empty"] += 1
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    break
                self._polite_wait()
                continue
            consecutive_empty = 0

            if len(records) >= target_listings:
                break
            if not self.has_more(payload):
                break
            self._polite_wait()

        self.writer.write_records(raw_entries)
        manifest = self._build_manifest(started_at, records, duplicates)
        manifest_path = self.writer.write_manifest(manifest.model_dump(mode="json"))
        manifest.outputs["manifest_path"] = str(manifest_path)
        return CollectorResult(manifest=manifest, records=records)

    def _build_manifest(
        self,
        started_at: datetime,
        records: list[CanonicalRecord],
        duplicates: int,
    ) -> RunManifest:
        pages_failed = self.pages["failed"] + self.pages["parse_failed"]
        status = "success"
        if pages_failed or self.failed_requests:
            status = "partial" if records else "failed"
        quality = QualityReport.build(records, self.source.field_support)
        manifest = RunManifest(
            run_id=self.run_id,
            platform=self.source_key,
            execution_mode=self.execution_mode,
            collection=self.collection,
            status=status,
            started_at=started_at,
            completed_at=utc_now(),
            extraction_version=self.pipeline.extraction_version,
            config_schema_version=1,
            pages=self.pages,
            records={
                "extracted": len(records),
                "duplicates_within_run": duplicates,
            },
            skip_reasons=self.skip_reasons,
            failed_requests=self.failed_requests,
            quality=quality,
            outputs={
                "raw_dir": str(self.writer.raw_dir_path),
            },
        )
        return manifest
