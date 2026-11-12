#!/usr/bin/env python3
"""Operational status report: per-platform collection state, DB health,
index freshness, backup status, image cache — plus health checks with
rolling baselines (warnings/criticals) and the operational journal tail.

Examples:
    python scripts/ops_status.py             # human-readable
    python scripts/ops_status.py --json      # machine-readable (incl. health)
    python scripts/ops_status.py --journal   # last 30 journal events
    python scripts/ops_status.py --problems  # only warnings + criticals
"""

from __future__ import annotations

import argparse
import json
import sys

from icmi.ops.health import build_health_report, summarize_health
from icmi.ops.journal import journal_tail
from icmi.ops.status import build_ops_status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--journal", type=int, default=0, metavar="N",
                        help="also print the last N journal events")
    parser.add_argument("--problems", action="store_true",
                        help="print only health warnings and criticals")
    args = parser.parse_args(argv)

    report = build_ops_status()
    report["health"] = build_health_report()

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1, default=str))
        return 0 if report["health"]["critical_count"] == 0 else 1

    if args.problems:
        problems = [c for c in report["health"]["checks"]
                    if c["level"] in ("warning", "critical")]
        if not problems:
            print("no health problems")
            return 0
        for c in problems:
            print(f"{c['level'].upper():8} {c['check']}: {c['detail']}")
        return 1 if any(c["level"] == "critical" for c in problems) else 0

    print(f"ops status @ {report['generated_at']}")
    print(summarize_health(report["health"]))
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
    if args.journal:
        print(f"\nlast {args.journal} journal events:")
        for rec in journal_tail(args.journal):
            level = rec.get("level", "info")
            marker = "*" if level != "info" else " "
            ctx = ", ".join(
                f"{k}={v}" for k, v in rec.items()
                if k not in ("ts", "kind", "event", "level")
            )
            print(f" {marker} {rec.get('ts', '?')[:19]} [{rec.get('kind')}] "
                  f"{rec.get('event')}: {ctx[:150]}")
    return 0 if report["health"]["critical_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
