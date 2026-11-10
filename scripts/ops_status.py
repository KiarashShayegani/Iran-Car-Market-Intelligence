#!/usr/bin/env python3
"""Operational status report: per-platform collection state, DB health,
index freshness, backup status, image cache.

Examples:
    python scripts/ops_status.py           # human-readable
    python scripts/ops_status.py --json    # machine-readable
"""

from __future__ import annotations

import argparse
import json
import sys

from icmi.ops.status import build_ops_status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args(argv)

    report = build_ops_status()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1, default=str))
        return 0

    print(f"ops status @ {report['generated_at']}")
    print(f"{'platform':<18} {'listings':>8} {'last run':<22} {'status':<10} {'P0%':>6}")
    for row in report["platforms"]:
        print(
            f"{row['platform']:<18} {row['listings']:>8} "
            f"{(row['last_run_at'] or '—')[:19]:<22} "
            f"{row['last_run_status'] or '—':<10} "
            f"{str(row['last_p0_coverage'] if row['last_p0_coverage'] is not None else '—'):>6}"
        )
    db = report["database"]
    if db.get("exists"):
        print(
            f"database: integrity={db['integrity_check']} listings={db['listings']} "
            f"observations={db['price_observations']} runs={db['runs_recorded']}"
        )
    idx = report["search_index"]
    if idx.get("exists"):
        print(
            f"index: built={idx.get('built_at')} vectors={idx.get('vectors')} "
            f"staleness={idx.get('staleness')}"
        )
    backup = report["backup"]
    print(f"backups: {backup['snapshots']} snapshot(s), latest={backup['latest_snapshot'] or '—'}")
    cache = report["image_cache"]
    print(f"image cache: {cache['cached_images']} images, {cache['total_mb']} MB")
    for note in report["notes"]:
        print(f"note: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
