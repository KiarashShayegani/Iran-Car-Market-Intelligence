#!/usr/bin/env python3
"""Full local pipeline: collect → ingest → analytical export → optional publish.

The offline-first execution path: everything runs on this machine with no
cloud involvement. Quantities are explicit so test runs stay small; scale up
deliberately.

Examples:
    python scripts/run_local_pipeline.py --platforms bama --max-pages 2 --target-listings 24
    python scripts/run_local_pipeline.py                       # all enabled, small defaults
    python scripts/run_local_pipeline.py --skip-collect        # ETL+export only
    python scripts/run_local_pipeline.py --publish --include-raw --zip
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from icmi.core.config import load_sources
from icmi.core.paths import raw_dir
from icmi.pipeline.analytical import export_analytics
from icmi.pipeline.etl import ingest_run
from icmi.pipeline.store import CanonicalStore


def build_run_id(platform: str) -> str:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{platform}-local"


def run_collect(platform: str, max_pages: int, target: int) -> str | None:
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "collect.py"),
        "--platform", platform,
        "--max-pages", str(max_pages),
        "--target-listings", str(target),
    ]
    print(f"--- collecting {platform} (pages<={max_pages}, target={target})")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"!!! collection for {platform} failed (exit {result.returncode}) — "
              "continuing with other platforms; raw/manifests reflect the failure",
              file=sys.stderr)
        return None
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--platforms", default=None,
        help="comma-separated config keys (default: every enabled source)",
    )
    parser.add_argument("--max-pages", type=int, default=2)
    parser.add_argument("--target-listings", type=int, default=48)
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--include-raw", action="store_true")
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args(argv)

    sources = load_sources()
    platforms = (
        [p.strip() for p in args.platforms.split(",") if p.strip()]
        if args.platforms
        else list(sources.enabled_sources())
    )
    print(f"platforms: {platforms}")

    # 1) collect
    if not args.skip_collect:
        for platform in platforms:
            run_collect(platform, args.max_pages, args.target_listings)

    # 2) ingest every run that has raw records
    if not args.skip_ingest:
        store = CanonicalStore()
        try:
            raw_root = raw_dir()
            run_ids = sorted(
                d.name for d in raw_root.iterdir()
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
                    if stats.skipped_runs:
                        print(f"ingest {run_id}/{platform}: already ingested — skipped")
                    else:
                        print(
                            f"ingest {run_id}/{platform}: new={stats.new_listings} "
                            f"updated={stats.updated_listings} "
                            f"observations+={stats.price_observations_appended} "
                            f"(changes={stats.price_change_events})"
                        )
        finally:
            store.close()

    # 3) analytical export
    store = CanonicalStore()
    try:
        result = export_analytics(store)
    finally:
        store.close()
    print(f"analytical: {result['listings_rows']} listings, "
          f"{result['observation_rows']} observations -> parquet")

    # 4) optional snapshot publish
    if args.publish:
        publish_cmd = [
            sys.executable,
            str(Path(__file__).parent / "publish_data.py"),
            *(["--include-raw"] if args.include_raw else []),
            *(["--zip"] if args.zip else []),
        ]
        subprocess.run(publish_cmd, check=True)

    print("local pipeline complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
