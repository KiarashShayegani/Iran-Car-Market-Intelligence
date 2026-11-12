#!/usr/bin/env python3
"""ICMI collection CLI.

Examples:
    # small local verification run (default): 3 pages, 36 listings max
    python scripts/collect.py --platform bama

    # explicit scale control
    python scripts/collect.py --platform bama --max-pages 10 --target-listings 100

    # CI mode (identical outputs; manifest records the execution mode)
    python scripts/collect.py --platform bama --mode ci

Outputs (under the repo data dir):
    data/raw/<run_id>/<platform>/pages/page_NNNN.json   verbatim responses
    data/raw/<run_id>/<platform>/records.jsonl          raw ad payloads
    data/manifests/<run_id>.json                        run manifest + quality
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

from icmi.collectors.registry import get_collector
from icmi.core.config import load_sources


def build_run_id(platform: str, mode: str) -> str:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{platform}-{mode}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--platform", required=True, help="config key, e.g. bama")
    parser.add_argument(
        "--mode", choices=("local", "ci"), default="local",
        help="execution mode recorded in the manifest (default: local)",
    )
    parser.add_argument(
        "--max-pages", type=int, default=3,
        help="maximum pages to request (default: 3 — keep small unless scaling)",
    )
    parser.add_argument(
        "--target-listings", type=int, default=36,
        help="stop after this many unique listings (default: 36)",
    )
    parser.add_argument(
        "--collection", default="full",
        help="named collection preset from config (default: full)",
    )
    parser.add_argument("--config", default=None, help="alternative sources.yaml")
    args = parser.parse_args(argv)

    sources = load_sources(args.config if args.config else None)
    if args.platform not in sources.sources:
        print(
            f"Unknown platform '{args.platform}'. "
            f"Configured: {', '.join(sources.sources)}",
            file=sys.stderr,
        )
        return 2
    source = sources.sources[args.platform]
    if not source.enabled:
        print(f"Platform '{args.platform}' is disabled in config.", file=sys.stderr)
        return 2

    # the collection policy is the single source of truth for budgets:
    # requested values are clamped to the platform's ceilings so no caller
    # (runner, workflow inputs, manual) can exceed them by accident
    try:
        from icmi.pipeline.policy import load_policy

        policy = load_policy().sources[args.platform]
        if args.max_pages > policy.max_pages_per_run:
            print(
                f"policy clamp: max-pages {args.max_pages} -> "
                f"{policy.max_pages_per_run} ({args.platform} ceiling)"
            )
            args.max_pages = policy.max_pages_per_run
        if args.target_listings > policy.max_listings_per_run:
            print(
                f"policy clamp: target-listings {args.target_listings} -> "
                f"{policy.max_listings_per_run} ({args.platform} ceiling)"
            )
            args.target_listings = policy.max_listings_per_run
    except (KeyError, ValueError, FileNotFoundError):
        pass  # no policy entry — requested values stand

    collector_cls = get_collector(args.platform)
    run_id = build_run_id(args.platform, args.mode)
    try:
        source.collection_params(args.collection)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    collector = collector_cls(
        source=source,
        pipeline=sources.pipeline,
        run_id=run_id,
        execution_mode=args.mode,
        collection=args.collection,
    )
    result = collector.collect(
        target_listings=args.target_listings, max_pages=args.max_pages
    )

    for line in result.manifest.summary_lines():
        print(line)
    print(f"manifest: {result.manifest.outputs.get('manifest_path', '')}")

    # operational journal (best-effort; never affects the run's outcome)
    try:
        from icmi.ops.journal import journal_collection_from_manifest

        journal_collection_from_manifest(result.manifest.model_dump(mode="json"))
    except Exception as exc:  # noqa: BLE001 — journaling must not break runs
        print(f"journal: skipped ({exc})", file=sys.stderr)
    return 0 if result.manifest.status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
