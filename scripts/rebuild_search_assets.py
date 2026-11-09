#!/usr/bin/env python3
"""Rebuild every derived view from already-collected raw data:
ingest runs → canonical DB → analytics exports → documents → vector index.

Use after new collection runs (or after an extraction-logic change) to make
the API/web app see fresh data. Equivalent to run_local_pipeline.py's
ETL+export steps, without any collection.

Example:
    python scripts/rebuild_search_assets.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from icmi.core.config import load_sources
from icmi.core.paths import raw_dir
from icmi.pipeline.etl import ingest_run
from icmi.pipeline.store import CanonicalStore


def run_step(label: str, cmd: list[str]) -> None:
    print(f"--- {label}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"!!! step failed (exit {result.returncode}): {label}", file=sys.stderr)
        sys.exit(result.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=None, help="alternative canonical DB path")
    args = parser.parse_args(argv)

    sources = load_sources()
    store = CanonicalStore(Path(args.db) if args.db else None)
    try:
        raw_root = raw_dir()
        run_ids = sorted(
            d.name
            for d in raw_root.iterdir()
            if d.is_dir() and any(d.glob("*/records.jsonl"))
        ) if raw_root.is_dir() else []
        for run_id in run_ids:
            for platform_dir in sorted((raw_root / run_id).iterdir()):
                if not platform_dir.is_dir():
                    continue
                platform = platform_dir.name
                if platform not in sources.sources:
                    continue
                stats = ingest_run(
                    run_id=run_id,
                    platform=platform,
                    store=store,
                    source=sources.sources[platform],
                    pipeline=sources.pipeline,
                )
                state = "skipped (already ingested)" if stats.skipped_runs else (
                    f"new={stats.new_listings} updated={stats.updated_listings} "
                    f"observations+={stats.price_observations_appended}"
                )
                print(f"ingest {run_id}/{platform}: {state}")
    finally:
        store.close()

    scripts_dir = Path(__file__).parent
    run_step("analytical exports", [sys.executable, str(scripts_dir / "export_analytics.py")])
    run_step("documents", [sys.executable, str(scripts_dir / "build_documents.py")])
    run_step("vector index", [sys.executable, str(scripts_dir / "build_index.py")])
    print("all derived views rebuilt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
