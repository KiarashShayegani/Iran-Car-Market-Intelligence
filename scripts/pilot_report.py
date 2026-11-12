#!/usr/bin/env python3
"""Seven-day pilot metrics — computed from data that already exists.

Reads manifests, the authoritative DB, the import ledger, and the pilot
log (docs/internal/PILOT_LOG.md); no network calls.

Distinguishes (owner requirement §1): retrieved records, unique listings
touched, brand-new listings, reobservations, and meaningful price-change
events — so "records processed" is never confused with "useful new data".

Per source (§2): freshness (hours since last successful collection vs
cadence), reliability (success rate + consecutive failures), and volume.

Chain accounting (§4): scheduled pilot days vs dispatched, per-source
outcomes, artifact publication, successful local imports, and
failed/missed executions — parsed from PILOT_LOG.md.

Usage:
    python scripts/pilot_report.py [--days 7] [--json] [--chain]
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
from datetime import UTC, datetime, timedelta
from pathlib import Path

from icmi.core.paths import canonical_dir, manifests_dir
from icmi.ops.status import _collect_manifests

PLATFORMS = ("bama", "sheypoor", "hamrah_mechanic")
CADENCE_HOURS = {"bama": 36.0, "sheypoor": 24.0, "hamrah_mechanic": 96.0}
PILOT_START = datetime(2026, 11, 13, 2, 30, tzinfo=UTC)
REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _source_metrics(conn: sqlite3.Connection, runs: list[dict], cadence: float) -> dict:
    successes = [m for m in runs if m.get("status") == "success"]
    retrieved = [m.get("records") or 0 for m in successes]
    durations = []
    for m in successes:
        start, end = _ts(m.get("started_at")), _ts(m.get("completed_at"))
        if start and end:
            durations.append((end - start).total_seconds())
    run_ids = [m.get("run_id") for m in runs]
    unique = new = price_changes = 0
    if run_ids:
        qm = ",".join("?" * len(run_ids))
        unique = conn.execute(
            f"SELECT COUNT(DISTINCT listing_key) FROM price_observations "
            f"WHERE run_id IN ({qm})", run_ids,
        ).fetchone()[0]
        new = conn.execute(
            f"SELECT COUNT(DISTINCT o.listing_key) FROM price_observations o "
            f"WHERE o.run_id IN ({qm}) AND NOT EXISTS ("
            f"SELECT 1 FROM price_observations o2 WHERE o2.listing_key = o.listing_key "
            f"AND o2.run_id NOT IN ({qm}))",
            (*run_ids, *run_ids),
        ).fetchone()[0]
        price_changes = conn.execute(
            f"SELECT COUNT(*) FROM price_observations o WHERE o.run_id IN ({qm}) "
            f"AND EXISTS (SELECT 1 FROM price_observations o2 WHERE "
            f"o2.listing_key = o.listing_key AND o2.run_id < o.run_id)",
            run_ids,
        ).fetchone()[0]
    p0s = [m.get("p0_coverage") for m in successes if m.get("p0_coverage") is not None]
    dup_rates = [
        (m.get("duplicates") or 0) / max(1, m.get("records") or 1)
        for m in successes if (m.get("records") or 0) >= 10
    ]
    # consecutive failures (most recent first)
    streak = 0
    for m in reversed(runs):
        if m.get("status") == "failed":
            streak += 1
        else:
            break
    last = _ts(successes[-1].get("completed_at")) if successes else None
    last_age = (datetime.now(tz=UTC) - last).total_seconds() / 3600 if last else None
    return {
        "runs": len(runs),
        "successes": len(successes),
        "failures": len(runs) - len(successes),
        "consecutive_failures": streak,
        "reliability_pct": round(100 * len(successes) / len(runs), 1) if runs else None,
        "freshness_hours_since_last_success": round(last_age, 1)
        if last_age is not None else None,
        "fresh_within_cadence": last_age is not None and last_age <= cadence * 1.5,
        "retrieved_records": sum(retrieved),
        "records_per_run_median": statistics.median(retrieved) if retrieved else 0,
        "unique_listings_touched": unique,
        "brand_new_listings": new,
        "reobservations": unique - new,
        "price_change_events": price_changes,
        "p0_coverage_range": [min(p0s), max(p0s)] if p0s else None,
        "max_duplicate_rate": round(max(dup_rates), 3) if dup_rates else 0,
        "median_duration_s": round(statistics.median(durations), 1) if durations else None,
        "last_success_at": successes[-1].get("completed_at") if successes else None,
    }


def _chain_accounting() -> dict:
    """Scheduled vs dispatched vs published vs imported vs missed (§4).

    Parsed from PILOT_LOG.md (one line per pilot day, written by the
    automation) + the import ledger (successful local imports).
    """
    log_path = REPO_ROOT / "docs" / "internal" / "PILOT_LOG.md"
    entries = []
    if log_path.is_file():
        entries = [
            line for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("- ")
        ]
    days_scheduled = max(
        0, int((datetime.now(tz=UTC) - PILOT_START).total_seconds() // 86400) + 1
    )
    dispatched = []
    for entry in entries:
        m = re.search(r"run (\d+)", entry)
        if m:
            dispatched.append(m.group(1))
    ledger_imports = []
    for lf in manifests_dir().glob("actions-run-*.json"):
        try:
            d = json.loads(lf.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if d.get("imported_at") and datetime.fromtimestamp(
            d["imported_at"], tz=UTC
        ) >= PILOT_START:
            ledger_imports.append(str(d.get("actions_run_id")))
    return {
        "pilot_days_elapsed": days_scheduled,
        "log_entries": len(entries),
        "dispatched_runs_logged": len(dispatched),
        "successful_local_imports": len(set(ledger_imports) & set(dispatched)),
        "missed_days": max(0, days_scheduled - len(dispatched)),
        "note": "missed days are recorded, never silently retried; "
                "auto_sync catches the backlog on the next successful night",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--pilot", action="store_true",
                        help="window = the pilot period only (PILOT_START → "
                        "now) — pre-pilot experimental data is excluded so "
                        "the final report represents the seven-day period")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--chain", action="store_true",
                        help="print only the execution-chain accounting")
    args = parser.parse_args(argv)

    if args.pilot:
        since = PILOT_START
    else:
        since = datetime.now(tz=UTC) - timedelta(days=args.days)
    manifests = _manifest_rows(since)
    db = canonical_dir() / "icmi.db"
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    out: dict = {
        "window_days": args.days,
        "pilot_window": args.pilot,
        "since": since.isoformat(timespec="seconds"),
        "sources": {},
        "database": {},
        "synchronization": {},
        "chain": _chain_accounting(),
        "budget_policy": "conservative budgets per collection_policy "
                         "schema_version 2 — changes require pilot evidence "
                         "and owner approval",
    }

    for platform in PLATFORMS:
        runs = [m for m in manifests if m.get("platform") == platform]
        runs.sort(key=lambda m: m["_started"])
        out["sources"][platform] = _source_metrics(
            conn, runs, CADENCE_HOURS[platform]
        )

    out["database"] = {
        "listings": conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0],
        "observations": conn.execute(
            "SELECT COUNT(*) FROM price_observations").fetchone()[0],
        "runs": conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
    }
    conn.close()

    ledgers = sorted(manifests_dir().glob("actions-run-*.json"))
    imports = []
    for lf in ledgers:
        try:
            d = json.loads(lf.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if d.get("imported_at") and datetime.fromtimestamp(
            d["imported_at"], tz=UTC
        ) >= since:
            imports.append(d)
    out["synchronization"] = {
        "imports_in_window": len(imports),
        "all_idempotent": all(i.get("idempotent") for i in imports),
        "listings_delta_from_imports": sum(
            (i.get("after", {}).get("listings", 0))
            - (i.get("before", {}).get("listings", 0))
            for i in imports
        ),
    }

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
        return 0

    if args.chain:
        print(json.dumps(out["chain"], indent=1))
        return 0

    window = "the pilot period (pre-pilot data excluded)" if args.pilot \
        else f"last {args.days} days"
    print(f"PILOT REPORT — window: {window} (since {out['since']})\n")
    for platform, s in out["sources"].items():
        fresh = "fresh" if s["fresh_within_cadence"] else "STALE"
        print(f"{platform}:")
        print(f"  reliability: {s['successes']}/{s['runs']} success "
              f"({s['reliability_pct']}%), "
              f"consecutive_failures={s['consecutive_failures']}")
        print(f"  freshness: {s['freshness_hours_since_last_success']}h since "
              f"last success [{fresh}] (cadence ~{CADENCE_HOURS[platform]:.0f}h)")
        print(f"  retrieved={s['retrieved_records']} "
              f"(median/run={s['records_per_run_median']}) → "
              f"unique={s['unique_listings_touched']} "
              f"(new={s['brand_new_listings']}, reobserved={s['reobservations']})")
        print(f"  price_changes={s['price_change_events']} p0={s['p0_coverage_range']} "
              f"max_dup_rate={s['max_duplicate_rate']} "
              f"duration~{s['median_duration_s']}s")
    print(f"\ndatabase: {out['database']}")
    print(f"synchronization: {out['synchronization']}")
    print(f"chain: {json.dumps(out['chain'])}")
    print(f"budgets: {out['budget_policy']}")
    return 0


if __name__ == "__main__":
    main()
