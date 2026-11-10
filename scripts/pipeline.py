#!/usr/bin/env python3
"""Unified ICMI pipeline CLI — the single entrypoint used by manual runs,
GitHub Actions workflows, and (future, disabled) scheduler wrappers.

Examples:
    python scripts/pipeline.py all                          # ingest→export→documents→index
    python scripts/pipeline.py collect --platforms bama --max-pages 3
    python scripts/pipeline.py collect all                  # collect then refresh views
    python scripts/pipeline.py ingest export index          # refresh views only
"""

from __future__ import annotations

import argparse
import sys

from icmi.pipeline.runner import PipelineRunner

ALL_VIEW_STAGES = ["ingest", "export", "documents", "index"]
KNOWN_STAGES = ["collect", "ingest", "export", "documents", "index", "all"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "stages",
        nargs="+",
        choices=KNOWN_STAGES,
        help="stages to run, in order; 'all' = ingest+export+documents+index",
    )
    parser.add_argument(
        "--platforms", default=None, help="comma-separated platform filter"
    )
    parser.add_argument(
        "--mode", choices=("manual", "ci", "scheduled"), default="manual",
        help="execution mode recorded in manifests (default: manual)",
    )
    parser.add_argument(
        "--collection", default="full",
        help="named collection preset (e.g. brand-pride) — applies to collect",
    )
    parser.add_argument("--max-pages", type=int, default=2)
    parser.add_argument("--target-listings", type=int, default=48)
    parser.add_argument("--config", default=None, help="alternative sources.yaml")
    args = parser.parse_args(argv)

    stages: list[str] = []
    for stage in args.stages:
        if stage == "all":
            stages.extend(ALL_VIEW_STAGES)
        else:
            stages.append(stage)

    platforms = (
        [p.strip() for p in args.platforms.split(",") if p.strip()]
        if args.platforms
        else None
    )

    runner = PipelineRunner(mode=args.mode, config_path=args.config)
    report = runner.run(
        stages, platforms=platforms, max_pages=args.max_pages,
        target_listings=args.target_listings, collection=args.collection,
    )

    for line in report.summary_lines():
        print(line)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
