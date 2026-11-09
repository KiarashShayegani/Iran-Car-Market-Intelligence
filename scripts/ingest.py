#!/usr/bin/env python3
"""Ingest collected run(s) into the canonical SQLite store.

Examples:
    python scripts/ingest.py --run-id 20261109T192853-bama-local
    python scripts/ingest.py --run-id 20261109T194351-sheypoor-local --db path/to/icmi.db

The platform is inferred from the run directory contents. Ingestion is
idempotent: re-ingesting a run updates listings and appends price
observations only when the price actually changed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from icmi.core.config import load_sources
from icmi.core.paths import raw_dir
from icmi.pipeline.etl import ingest_run
from icmi.pipeline.store import CanonicalStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True, help="run id, e.g. 20261109T192853-bama-local")
    parser.add_argument("--db", default=None, help="alternative canonical DB path")
    parser.add_argument("--config", default=None, help="alternative sources.yaml")
    args = parser.parse_args(argv)

    sources = load_sources(args.config if args.config else None)
    run_dir = raw_dir() / args.run_id
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2
    platforms = [
        p.name for p in sorted(run_dir.iterdir())
        if p.is_dir() and (p / "records.jsonl").is_file()
    ]
    if not platforms:
        print(f"No records.jsonl found under {run_dir}", file=sys.stderr)
        return 2

    store = CanonicalStore(Path(args.db) if args.db else None)
    try:
        for platform in platforms:
            if platform not in sources.sources:
                print(f"Unknown platform in run: '{platform}' — skipped", file=sys.stderr)
                continue
            stats = ingest_run(
                run_id=args.run_id,
                platform=platform,
                store=store,
                source=sources.sources[platform],
                pipeline=sources.pipeline,
            )
            print(
                f"{args.run_id} / {platform}: new={stats.new_listings} "
                f"updated={stats.updated_listings} "
                f"observations+={stats.price_observations_appended} "
                f"(changes={stats.price_change_events})"
            )
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
