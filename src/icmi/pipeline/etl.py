"""ETL: raw collection runs → canonical store (cross-run dedup + history).

Ingestion re-derives canonical records from the immutable raw payloads via
the platform collector's extraction — the same code path as collection-time
extraction, so raw → canonical is deterministic and re-runnable. A collection
failure never touches the store; ingestion only starts from successfully
written records.jsonl artifacts.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from icmi.collectors.base import BaseCollector
from icmi.collectors.registry import get_collector
from icmi.core.config import PipelineConfig, SourceConfig
from icmi.core.paths import manifests_dir, raw_dir
from icmi.core.records import RawPayloadRef
from icmi.pipeline.store import CanonicalStore, IngestStats


def ingest_run(
    run_id: str,
    platform: str,
    store: CanonicalStore,
    source: SourceConfig,
    pipeline: PipelineConfig,
) -> IngestStats:
    """Ingest one collected run's raw records into the canonical store."""
    records_path = raw_dir() / run_id / platform / "records.jsonl"
    if not records_path.is_file():
        raise FileNotFoundError(
            f"No raw records for run '{run_id}' platform '{platform}': {records_path}"
        )

    collector_cls: type[BaseCollector] = get_collector(platform)
    extractor = collector_cls.parse_only(source, pipeline)

    records = []
    with open(records_path, encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            entry = json.loads(line)
            listing_id = str(entry["source_listing_id"])
            payload_ref = RawPayloadRef(
                run_id=run_id,
                platform=platform,
                file=f"records.jsonl#{line_number}",
                key=listing_id,
            )
            record = extractor.parse_listing(entry["raw_payload"], payload_ref)
            if record is None:
                continue
            records.append(record)

    stats = store.upsert_listings(records, run_id)

    manifest_path = manifests_dir() / f"{run_id}.json"
    if manifest_path.is_file():
        store.record_run(
            json.loads(manifest_path.read_text(encoding="utf-8")), run_id, platform
        )
    return stats


def stats_summary(stats: IngestStats) -> dict:
    return asdict(stats)
