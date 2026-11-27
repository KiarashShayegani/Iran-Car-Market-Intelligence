"""Divar.ir collector (EXPERIMENTAL PROTOTYPE — bounded local runs only).

Endpoint: POST https://api.divar.ir/v8/postlist/w/search (decoded 2026-11-20,
see docs/internal/DIVAR_INVESTIGATION.md):

- Category form-data: {"category": {"str": {"value": "light"}}} — the "str"
  wrapper object is mandatory; direct str_value returns a 400.
- Newest-first: {"sort": {"str": {"value": "sort_date"}}}.
- Pagination: cursor-style — each response carries pagination.data which the
  next request echoes verbatim in pagination_data (verified: 8 pages, 194
  unique tokens, zero dupes/missing).

List rows are display-level (title, price TEXT «۳,۲۳۰,۰۰۰,۰۰۰ تومان»,
location text, MILEAGE text on the top chip «۳۰۰,۰۰۰ کیلومتر» / «صفر
کیلومتر», image, token → public URL). Structured brand/model/trim/year/
description require the per-post SSR page (divar.ir/v/<token>) — detail
enrichment is a separate upstream step, NOT part of this list-only
prototype.

The response uses widget envelopes: list_widgets[] with widget_type ==
"POST_ROW" carry a data node per ad.
"""

from __future__ import annotations

import json
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from icmi.collectors.base import BaseCollector, TransientHTTPError
from icmi.core.records import CanonicalRecord, RawPayloadRef, utc_now


def _nonempty(value: Any) -> bool:
    return value is not None and value != ""


class DivarCollector(BaseCollector):
    """Collector for Divar's widget-based search API (POST + cursor)."""

    source_key = "divar"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cursor: dict[str, Any] | None = None
        self.session.headers.update(
            {"Accept": "application/json", "Origin": "https://divar.ir",
             "Referer": "https://divar.ir/s/tehran/car"}
        )

    def page_params(self, page_index: int) -> dict[str, Any]:
        """Build the POST body; page 0 is the cold start, later pages carry
        the previous response's pagination.data cursor."""
        city_ids = self.collection_params.get("city_ids", ["1"])
        body: dict[str, Any] = {
            "city_ids": city_ids,
            "search_data": {"form_data": {"data": {
                "category": {"str": {"value": self.collection_params.get("category", "light")}},
                "sort": {"str": {"value": self.collection_params.get("sort", "sort_date")}},
            }}},
        }
        if page_index > 0 and self._cursor:
            body["pagination_data"] = self._cursor
        return body

    @retry(
        retry=retry_if_exception_type(
            (requests.ConnectionError, requests.Timeout, TransientHTTPError)
        ),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        reraise=True,
    )
    def _fetch_json(self, params: dict[str, Any]) -> dict:
        """POST variant: params is the JSON body; 403/429/5xx are failures."""
        self.pages["requested"] += 1
        response = self.session.post(
            self.source.base_url,
            json=params,
            timeout=self.pipeline.request_timeout_seconds,
        )
        if response.status_code in (401, 403):
            # Divar blocks the endpoint — record and stop the run (bounded,
            # no hammering)
            self.pages["failed"] += 1
            self.failed_requests.append(
                f"POST {response.status_code}: {response.text[:120]}"
            )
            raise requests.HTTPError(
                f"{response.status_code} — endpoint blocked",
                response=response,
            )
        if response.status_code == 429 or 500 <= response.status_code < 600:
            self.pages["failed"] += 1
            raise TransientHTTPError(response.status_code, response.text[:120])
        response.raise_for_status()
        self.pages["fetched"] += 1
        return response.json()

    def observe_page(self, page_payload: dict) -> None:
        """Store the pagination cursor for the next request (BaseCollector
        calls this once per fetched page, before extraction)."""
        cursor = (page_payload.get("pagination") or {}).get("data")
        if isinstance(cursor, dict):
            self._cursor = cursor

    def iter_ads(self, page_payload: dict) -> list[dict]:
        ads: list[dict] = []
        seen: set[str] = set()
        for w in page_payload.get("list_widgets", []):
            if w.get("widget_type") != "POST_ROW":
                continue
            data = w.get("data") or {}
            token = data.get("token")
            if not token or token in seen:
                continue
            seen.add(token)
            ads.append(data)
        return ads

    def has_more(self, page_payload: dict) -> bool:
        pag = page_payload.get("pagination") or {}
        return bool(pag.get("has_next_page", False))

    def listing_id_of(self, ad: dict) -> str | None:
        token = ad.get("token")
        return str(token) if token else None

    def parse_listing(self, ad: dict, payload_ref: RawPayloadRef) -> CanonicalRecord | None:
        errors: dict[str, str] = {}
        token = str(ad.get("token") or "")
        record = CanonicalRecord(
            source_platform=self.source_key,
            source_listing_id=token,
            original_url=f"https://divar.ir/v/{token}" if token else None,
            scraped_at=utc_now(),
            extraction_version=self.pipeline.extraction_version,
            raw_payload_ref=payload_ref,
        )

        def set_field(field: str, extractor) -> None:
            try:
                setattr(record, field, extractor())
            except Exception as exc:  # noqa: BLE001 — parser bugs are extraction failures
                errors[field] = f"{type(exc).__name__}: {exc}"

        # list rows carry: title, price TEXT, location, image, and MILEAGE
        # text (the top chip «۳۰۰,۰۰۰ کیلومتر» / «صفر کیلومتر»).
        # brand/model/trim/year/description are NOT in the list payload —
        # they require the per-post SSR page (detail enrichment, separate
        # upstream step). Honest missing over fabricated values.
        set_field("title", lambda: ad.get("title"))
        price_text = ad.get("middle_description_text")
        set_field("price_raw", lambda: price_text if _nonempty(price_text) else None)
        # NOTE: price_amount is derived downstream (store normalizes price_raw);
        # CanonicalRecord itself carries only price_raw — do not set it here.
        set_field("location_raw", lambda: ad.get("bottom_description_text"))

        # mileage from the top chip — only when it actually is a mileage
        # text (other layouts may put condition text there; never fabricate)
        top_text = str(ad.get("top_description_text") or "").strip()

        def _mileage_raw():
            if top_text and ("کیلومتر" in top_text or top_text == "صفر"):
                return top_text
            return None

        set_field("mileage_raw", _mileage_raw)

        set_field("image_count", lambda: ad.get("image_count"))
        image = ad.get("image_url")
        set_field("image_urls", lambda: [image] if _nonempty(image) else [])
        # trust/boost flags (نردبان شده / پله شده) — Divar-provided, not
        # verified by us. NOTE: source_published_at stays None — list rows
        # carry no posting date (the SEO date lives on the SSR page).
        def _top_tag():
            tag = ad.get("image_top_left_tag") or {}
            return tag.get("text") if isinstance(tag, dict) else None

        set_field(
            "source_attributes",
            lambda: {
                "top_description_text": ad.get("top_description_text"),
                "red_text": ad.get("red_text"),
                "image_top_left_tag": _top_tag(),
                "has_chat": ad.get("has_chat"),
                "layout_type": ad.get("layout_type"),
                "widget_envelope": "POST_ROW",
            },
        )
        record.extraction_errors = errors
        return record

    @staticmethod
    def dump_payload(ad: dict) -> str:
        return json.dumps(ad, ensure_ascii=False, default=str)

