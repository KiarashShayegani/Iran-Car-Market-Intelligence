#!/usr/bin/env python3
"""Seven-day pilot metrics — computed from data that already exists.

Reads manifests, the authoritative DB, and the import ledger; no network
calls except an optional Actions query for scheduled-vs-actual accounting.

Usage:
    python scripts/pilot_report.py [--days 7] [--json]

Metrics per brief §8-pilot: per-source run counts/success rate, durations,
records, unique listings, new listings, price changes, requests, duplicate
rates, P0 coverage, artifact sizes, sync/import accounting, DB health.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from datetime import UTC, datetime, timedelta

from icmi.core.paths import canonical_dir, manifests_dir
from icmi.ops.status import _collect_manifests

PLATFORMS = ("bama", "sheypoor", "hamrah_mechanic")


def _ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    except ValueError:
        return None


def _manifest_rows(since: datetime) -> list[dict]:
    rows = []
    for m in _collect_manifests():
        started = _ts(m.get("started_at"))
        if started and started >= since:
            m["_started"] = started
            rows.append(m)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    since = datetime.now(tz=UTC) - timedelta(days=args.days)
    manifests = _manifest_rows(since)
    db = canonical_dir() / "icmi.db"
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    out: dict = {"window_days": args.days, "since": since.isoformat(timespec="seconds"),
                 "sources": {}, "database": {}, "synchronization": {}}

    for platform in PLATFORMS:
        runs = [m for m in manifests if m.get("platform") == platform]
        runs.sort(key=lambda m: m["_started"])
        successes = [m for m in runs if m.get("status") == "success"]
        extracted = [m.get("records") or 0 for m in successes]
        durations = []
        for m in successes:
            start, end = _ts(m.get("started_at")), _ts(m.get("completed_at"))
            if start and end:
                durations.append((end - start).total_seconds())
        # unique listings seen in window + new listings (first-ever observations)
        run_ids = [m.get("run_id") for m in runs]
        unique = 0
        new = 0
        qmarks = ",".join("?" * len(run_ids)) if run_ids else None
        if run_ids:
            unique = conn.execute(
                f"SELECT COUNT(DISTINCT listing_key) FROM price_observations "
                f"WHERE run_id IN ({qmarks})", run_ids,
            ).fetchone()[0]
            new = conn.execute(
                f"SELECT COUNT(DISTINCT o.listing_key) FROM price_observations o "
                f"WHERE o.run_id IN ({qmarks}) AND NOT EXISTS ("
                f"SELECT 1 FROM price_observations o2 WHERE o2.listing_key = o.listing_key "
                f"AND o2.run_id NOT IN ({qmarks}))",
                (*run_ids, *run_ids),
            ).fetchone()[0]
        price_changes = conn.execute(
            f"SELECT COUNT(*) FROM price_observations o WHERE o.run_id IN ({qmarks}) "
            f"AND EXISTS (SELECT 1 FROM price_observations o2 WHERE "
            f"o2.listing_key = o.listing_key AND o2.run_id < o.run_id)",
            run_ids if run_ids else ["-"],
        ).fetchone()[0]
        p0s = [m.get("p0_coverage") for m in successes if m.get("p0_coverage") is not None]
        dup_rates = [
            (m.get("duplicates") or 0) / max(1, m.get("records") or 1)
            for m in successes if (m.get("records") or 0) >= 10
        ]  # records = extracted count, duplicates = within-run dupes
        out["sources"][platform] = {
            "runs": len(runs),
            "successes": len(successes),
            "failures": len(runs) - len(successes),
            "success_rate_pct": round(100 * len(successes) / len(runs), 1) if runs else None,
            "median_duration_s": round(statistics.median(durations), 1) if durations else None,
            "records_total": sum(extracted),
            "records_per_run_median": statistics.median(extracted) if extracted else 0,
            "unique_listings_touched": unique,
            "brand_new_listings": new,
            "price_change_events": price_changes,
            "p0_coverage_range": [min(p0s), max(p0s)] if p0s else None,
            "max_duplicate_rate": round(max(dup_rates), 3) if dup_rates else 0,
            "last_run_at": runs[-1].get("completed_at") if runs else None,
        }

    out["database"] = {
        "listings": conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0],
        "observations": conn.execute("SELECT COUNT(*) FROM price_observations").fetchone()[0],
        "runs": conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
    }
    conn.close()

    # synchronization accounting from the import ledger
    ledgers = sorted(manifests_dir().glob("actions-run-*.json"))
    imports = []
    for lf in ledgers:
        try:
            d = json.loads(lf.read_text(encoding="utf-8"))
        except ValueError:
            continue
        imported_at = d.get("imported_at")
        if imported_at and datetime.fromtimestamp(imported_at, tz=UTC) >= since:
            imports.append(d)
    out["synchronization"] = {
        "imports_in_window": len(imports),
        "all_idempotent": all(i.get("idempotent") for i in imports),
        "listings_delta_from_imports": sum(
            (i.get("after", {}).get("listings", 0)) - (i.get("before", {}).get("listings", 0))
            for i in imports
        ),
    }

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
        return 0

    print(f"PILOT REPORT — window: last {args.days} days (since {out['since']})\n")
    for platform, s in out["sources"].items():
        print(f"{platform}:")
        print(f"  runs={s['runs']} success={s['successes']} "
              f"({s['success_rate_pct']}%) failures={s['failures']}")
        print(f"  records={s['records_total']} "
              f"(median/run={s['records_per_run_median']}) "
              f"unique={s['unique_listings_touched']} new={s['brand_new_listings']}")
        print(f"  price_changes={s['price_change_events']} "
              f"p0={s['p0_coverage_range']} max_dup_rate={s['max_duplicate_rate']}")
        print(f"  median_duration={s['median_duration_s']}s last={s['last_run_at']}")
    print(f"\ndatabase: {out['database']}")
    print(f"synchronization: {out['synchronization']}")
    return 0


if __name__ == "__main__":
    main()
