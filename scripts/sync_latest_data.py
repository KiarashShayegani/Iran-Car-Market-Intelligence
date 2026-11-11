#!/usr/bin/env python3
"""Import an ICMI snapshot (Actions artifact or local export) into the local
data layer, then ingest its runs into the authoritative canonical DB.

This is the receiving end of the transfer seam:
- manifests    → data/manifests/
- raw runs     → data/raw/<run_id>/<platform>/
- analytical   → data/analytical/ (read-only Parquet snapshots)
- canonical DB → IGNORED. The local SQLite is authoritative; it is updated
  exclusively by local ingestion of the imported raw runs (cross-run dedup
  and price-history rules apply). No SQLite merging, ever.

Idempotent by construction: runs already recorded in the local store are
skipped (run-level fast path), and record-level upsert rules guarantee no
duplicate listings or observations even on partial re-ingestion. Importing
the same artifact twice changes nothing.

Examples:
    python scripts/sync_latest_data.py --source icmi_snapshot_20261110T120000.zip
    python scripts/sync_latest_data.py --source ~/Downloads/probe-manifests
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from icmi.core.config import load_sources
from icmi.core.paths import analytical_dir, manifests_dir, raw_dir
from icmi.pipeline.etl import ingest_run
from icmi.pipeline.store import CanonicalStore


def _extract_if_zip(source: Path, workdir: Path) -> Path:
    if source.is_file() and source.suffix == ".zip":
        extract_dir = workdir / source.stem
        with zipfile.ZipFile(source) as zf:
            zf.extractall(extract_dir)
        return extract_dir
    if source.is_dir():
        return source
    raise FileNotFoundError(f"Snapshot source not found or not a dir/zip: {source}")


def _copy_tree(src: Path, dest: Path) -> int:
    copied = 0
    if not src.is_dir():
        return 0
    for item in src.rglob("*"):
        if item.is_file():
            rel = item.relative_to(src)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            copied += 1
    return copied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="snapshot dir or zip path")
    parser.add_argument(
        "--skip-ingest", action="store_true",
        help="copy artifacts only; do not ingest runs into the canonical DB",
    )
    parser.add_argument(
        "--skip-derived", action="store_true",
        help="do not copy analytical parquet / index from the snapshot "
        "(used when extraction versions may differ — raw runs still import)",
    )
    parser.add_argument("--db", default=None, help="alternative canonical DB path")
    args = parser.parse_args(argv)

    source = Path(args.source).expanduser().resolve()
    sources_cfg = load_sources()
    stats_by_run: dict[str, object] = []

    with tempfile.TemporaryDirectory(prefix="icmi-sync-") as tmp:
        root = _extract_if_zip(source, Path(tmp))

        # manifests
        n = _copy_tree(root / "manifests", manifests_dir())
        print(f"manifests copied: {n} file(s)")

        # raw runs
        n = _copy_tree(root / "raw", raw_dir())
        print(f"raw files copied: {n} file(s)")

        # analytical snapshots (never authoritative)
        analytical = root / "analytical"
        if args.skip_derived:
            print("derived data copy skipped (--skip-derived)")
        elif analytical.is_dir() and any(analytical.glob("*.parquet")):
            n = _copy_tree(analytical, analytical_dir())
            print(f"analytical parquet copied: {n} file(s)")

        # canonical DB from the artifact is intentionally ignored
        if (root / "canonical" / "icmi.db").is_file():
            print(
                "note: snapshot's canonical DB ignored — the local store is "
                "authoritative and is updated by ingestion only"
            )

        if args.skip_ingest:
            print("ingest skipped (--skip-ingest)")
            return 0

        raw_root = raw_dir()
        run_ids = sorted(
            d.name for d in raw_root.iterdir() if d.is_dir() and any(d.glob("*/records.jsonl"))
        ) if raw_root.is_dir() else []

        store = CanonicalStore(Path(args.db) if args.db else None)
        try:
            already_known = set(store.ingested_run_ids())
            new_runs = [rid for rid in run_ids if rid not in already_known]
            print(
                f"runs discovered: {len(run_ids)} | "
                f"already ingested: {len(run_ids) - len(new_runs)}"
            )
            for run_id in new_runs:
                platform_dirs = [
                    d.name for d in (raw_root / run_id).iterdir()
                    if d.is_dir() and (d / "records.jsonl").is_file()
                ]
                for platform in platform_dirs:
                    if platform not in sources_cfg.sources:
                        print(f"  {run_id}/{platform}: unknown platform — skipped", file=sys.stderr)
                        continue
                    stats = ingest_run(
                        run_id=run_id,
                        platform=platform,
                        store=store,
                        source=sources_cfg.sources[platform],
                        pipeline=sources_cfg.pipeline,
                    )
                    print(
                        f"  {run_id}/{platform}: new={stats.new_listings} "
                        f"updated={stats.updated_listings} "
                        f"observations+={stats.price_observations_appended} "
                        f"(changes={stats.price_change_events})"
                    )
                    stats_by_run.append(run_id)  # type: ignore[dict-item]
        finally:
            store.close()

    print(f"ingested runs this import: {len(stats_by_run)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
