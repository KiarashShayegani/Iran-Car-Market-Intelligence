"""Health checks with rolling baselines — warnings and criticals.

Reads only data that already exists (manifests, runs table, backups,
index meta). Every check states its baseline rule and minimum sample
size so naturally small or young collections do not produce false
alarms. Cadence-lag checks report at ``info`` level while schedules are
inactive and escalate to ``warning`` once `schedules_active=True`.

Monitoring failures must never prevent collection: `build_health_report`
never raises for per-check errors — a failing check reports itself as a
critical finding instead.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from datetime import UTC, datetime

from icmi.core.paths import canonical_dir, manifests_dir

# Expected inter-run windows per platform, in hours. Used only for lag
# visibility; while schedules are dormant these surface as info.
CADENCE_HOURS = {"bama": 36.0, "sheypoor": 36.0, "hamrah_mechanic": 96.0}

_ENABLED_PLATFORMS = ("bama", "sheypoor", "hamrah_mechanic")


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        # DB timestamps (ingested_at etc.) are stored naive — they are UTC
        dt = dt.replace(tzinfo=UTC)
    return dt


def build_health_report(
    store: sqlite3.Connection | None = None,
    *,
    schedules_active: bool = False,
) -> dict:
    checks: list[dict] = []

    def check(name: str, level: str, detail: str, **extra) -> None:
        checks.append({"check": name, "level": level, "detail": detail, **extra})

    try:
        manifests = _load_manifests()
        _manifest_checks(manifests, check)
        _platform_checks(manifests, check)
        _yield_checks(manifests, check)
        _coverage_checks(manifests, check)
        _price_checks(store, check)
    except Exception as exc:  # noqa: BLE001 — a broken check must be visible,
        # not fatal.
        check("health_engine", "critical", f"health checks failed: {exc!r}")

    try:
        _database_checks(store, check)
        _backup_checks(check)
        _index_checks(check)
        _sync_lag_checks(store, check, schedules_active)
    except Exception as exc:  # noqa: BLE001
        check("health_engine", "critical", f"health checks failed: {exc!r}")

    levels = [c["level"] for c in checks]
    return {
        "generated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "schedules_active": schedules_active,
        "warning_count": levels.count("warning"),
        "critical_count": levels.count("critical"),
        "checks": checks,
    }


# --- manifest-derived checks -------------------------------------------------


def _load_manifests() -> list[dict]:
    out = []
    for path in sorted(manifests_dir().glob("*.json")):
        if path.name.startswith("actions-run-"):
            continue  # import ledger, not a collection manifest
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, OSError):
            out.append({"run_id": path.stem, "status": "corrupt_manifest"})
    return out


def _manifest_checks(manifests: list[dict], check) -> None:
    corrupt = [m.get("run_id") for m in manifests if m.get("status") == "corrupt_manifest"]
    if corrupt:
        check("manifest_integrity", "critical",
              f"{len(corrupt)} corrupt manifest file(s): {corrupt[:3]}")
    else:
        check("manifest_integrity", "info",
              f"{len(manifests)} manifests parse cleanly")


def _platform_checks(manifests: list[dict], check) -> None:
    for platform in _ENABLED_PLATFORMS:
        runs = [m for m in manifests if m.get("platform") == platform]
        runs.sort(key=lambda m: m.get("started_at") or "")
        if not runs:
            continue
        # consecutive failures
        streak = 0
        for m in reversed(runs):
            if m.get("status") == "failed":
                streak += 1
            elif m.get("status") == "success":
                break
        if streak >= 3:
            check("consecutive_failures", "critical",
                  f"{platform}: {streak} consecutive failed collection runs",
                  platform=platform, streak=streak)
        elif streak >= 2:
            check("consecutive_failures", "warning",
                  f"{platform}: {streak} consecutive failed collection runs",
                  platform=platform, streak=streak)
        # zero-record success
        last = runs[-1]
        if last.get("status") == "success" and not last.get("records", {}).get("extracted"):
            check("zero_record_run", "warning",
                  f"{platform}: latest successful run collected 0 records "
                  f"({last.get('run_id')})", platform=platform)
        # unexpected pagination termination
        pages = last.get("pages", {}) or {}
        if last.get("status") == "success" and pages.get("failed"):
            check("pagination_anomaly", "warning",
                  f"{platform}: latest run marked success with "
                  f"{pages['failed']} failed page(s)", platform=platform)


def _yield_checks(manifests: list[dict], check) -> None:
    """Yield far below the trailing baseline (min 3 prior runs).

    Only runs with a comparable page budget are compared — deliberately
    small experiment runs (1 page vs the usual 3) must not alarm.
    """
    for platform in _ENABLED_PLATFORMS:
        runs = [
            m for m in manifests
            if m.get("platform") == platform and m.get("status") == "success"
        ]
        runs.sort(key=lambda m: m.get("started_at") or "")
        yields_ = [m.get("records", {}).get("extracted") or 0 for m in runs]
        budgets = [
            (m.get("pages", {}) or {}).get("requested") or 0 for m in runs
        ]
        if len(yields_) < 4:
            continue
        latest_budget = budgets[-1]
        comparable = [
            y for y, b in zip(yields_[:-1], budgets[:-1], strict=False)
            if b == latest_budget and y > 0
        ]
        if len(comparable) < 3:
            continue
        baseline = statistics.median(comparable)
        latest = yields_[-1]
        if baseline >= 20 and latest < 0.3 * baseline:
            check("yield_drop", "warning",
                  f"{platform}: latest yield {latest} is <30% of the trailing "
                  f"median {baseline:.0f} at the same page budget "
                  f"({latest_budget} pages, {len(comparable)} baseline runs)",
                  platform=platform)
        # duplicate growth
        dup_rates = []
        for m in runs:
            extracted = m.get("records", {}).get("extracted") or 0
            dupes = m.get("records", {}).get("duplicates_within_run") or 0
            if extracted >= 10:
                dup_rates.append(dupes / extracted)
        if dup_rates and dup_rates[-1] > 0.30:
            check("duplicate_growth", "warning",
                  f"{platform}: latest run duplicate rate "
                  f"{dup_rates[-1]:.0%} > 30%", platform=platform)


def _coverage_checks(manifests: list[dict], check) -> None:
    for platform in _ENABLED_PLATFORMS:
        p0s = [
            m.get("quality", {}).get("p0_coverage_pct")
            for m in manifests
            if m.get("platform") == platform
            and m.get("status") == "success"
            and m.get("quality", {}).get("p0_coverage_pct") is not None
        ]
        if len(p0s) < 4:
            continue
        baseline = statistics.median(p0s[:-1])
        latest = p0s[-1]
        if baseline is not None and latest is not None and latest < baseline - 5:
            check("p0_deterioration", "warning",
                  f"{platform}: latest P0 coverage {latest:.1f}% dropped >5pts "
                  f"below trailing median {baseline:.1f}%", platform=platform)


def _price_checks(store: sqlite3.Connection | None, check) -> None:
    """Median price drift among listings observed in BOTH of the last two
    runs (min 10 common listings).

    Comparing whole-run medians is meaningless for rotating feeds — each
    run samples a different market slice. The shared-cohort comparison is
    the only honest signal; with zero overlap (rotating feeds) the check
    silently passes.
    """
    own = False
    if store is None:
        db = canonical_dir() / "icmi.db"
        if not db.is_file():
            return
        store = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        own = True
    try:
        for platform in _ENABLED_PLATFORMS:
            rows = store.execute(
                """
                SELECT o.run_id, o.listing_key, o.price_amount
                FROM price_observations o
                JOIN runs r ON r.run_id = o.run_id
                WHERE r.platform = ? AND o.price_amount IS NOT NULL
                ORDER BY r.started_at
                """,
                (platform,),
            ).fetchall()
            per_run: dict[str, dict[str, float]] = {}
            order: list[str] = []
            for run_id, key, amount in rows:
                if run_id not in per_run:
                    per_run[run_id] = {}
                    order.append(run_id)
                per_run[run_id][key] = amount
            if len(order) < 2:
                continue
            prev_map, latest_map = per_run[order[-2]], per_run[order[-1]]
            common = set(prev_map) & set(latest_map)
            if len(common) < 10:
                continue
            ratios = [
                latest_map[k] / prev_map[k]
                for k in common
                if prev_map[k] > 0
            ]
            if len(ratios) < 10:
                continue
            drift = statistics.median(ratios)
            if abs(drift - 1) > 0.15:
                check("price_distribution_shift", "warning",
                      f"{platform}: median same-listing price drift "
                      f"{(drift - 1) * 100:+.1f}% across {len(common)} "
                      "listings re-observed in the last two runs "
                      "(min 10 shared listings)", platform=platform)
    finally:
        if own:
            store.close()


# --- infrastructure checks ---------------------------------------------------


def _database_checks(store: sqlite3.Connection | None, check) -> None:
    own = False
    if store is None:
        db = canonical_dir() / "icmi.db"
        if not db.is_file():
            check("database", "critical", "authoritative DB not found")
            return
        store = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        own = True
    try:
        integrity = store.execute("PRAGMA quick_check").fetchone()[0]
        if integrity != "ok":
            check("database_integrity", "critical",
                  f"SQLite integrity check failed: {integrity}")
        else:
            listings = store.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
            check("database_integrity", "info",
                  f"integrity ok ({listings} listings)")
    finally:
        if own:
            store.close()


def _backup_checks(check) -> None:
    backups = sorted(
        (canonical_dir()).glob("pre-import-*.db"), key=lambda p: p.stat().st_mtime
    )
    if not backups:
        check("backup_freshness", "info", "no imports yet — no pre-import backups")
        return
    age_hours = (
        datetime.now(tz=UTC) - datetime.fromtimestamp(backups[-1].stat().st_mtime, tz=UTC)
    ).total_seconds() / 3600
    if age_hours > 24 * 7:
        check("backup_freshness", "warning",
              f"newest pre-import backup is {age_hours / 24:.1f} days old")
    else:
        check("backup_freshness", "info",
              f"newest pre-import backup {age_hours:.1f}h old "
              f"({backups[-1].name})")


def _index_checks(check) -> None:
    from icmi.core.paths import index_dir

    meta_path = index_dir() / "index_meta.json"
    db = canonical_dir() / "icmi.db"
    if not meta_path.is_file() or not db.is_file():
        check("index_freshness", "info", "no index meta — nothing to compare")
        return
    try:
        info = json.loads(meta_path.read_text(encoding="utf-8"))
    except ValueError:
        check("index_freshness", "warning", "index_meta.json is corrupt")
        return
    built = _parse_ts(info.get("built_at"))
    if built is None:
        return
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        newest = conn.execute("SELECT MAX(first_seen_at) FROM listings").fetchone()[0]
    finally:
        conn.close()
    newest_dt = _parse_ts(newest)
    if newest_dt and built < newest_dt:
        check("index_freshness", "warning",
              "search index is older than the newest listing — rebuild with "
              "scripts/rebuild_search_assets.py")


def _sync_lag_checks(store: sqlite3.Connection | None, check,
                     schedules_active: bool) -> None:
    own = False
    if store is None:
        db = canonical_dir() / "icmi.db"
        if not db.is_file():
            return
        store = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        own = True
    level = "warning" if schedules_active else "info"
    try:
        for platform in _ENABLED_PLATFORMS:
            row = store.execute(
                "SELECT MAX(ingested_at) FROM runs WHERE platform = ?",
                (platform,),
            ).fetchone()
            last_ingested = _parse_ts(row[0]) if row and row[0] else None
            if last_ingested is None:
                continue
            hours = (datetime.now(tz=UTC) - last_ingested).total_seconds() / 3600
            cadence = CADENCE_HOURS[platform]
            if hours > cadence * 2:
                check("sync_lag", level,
                      f"{platform}: nothing ingested for {hours:.0f}h "
                      f"(cadence window {cadence:.0f}h)", platform=platform)
    finally:
        if own:
            store.close()


def summarize_health(report: dict) -> str:
    """One-line human summary for CLI output."""
    icon = "OK" if not report["critical_count"] and not report["warning_count"] \
        else ("CRITICAL" if report["critical_count"] else "WARNING")
    parts = [f"health={icon}"]
    if report["critical_count"]:
        parts.append(f"{report['critical_count']} critical")
    if report["warning_count"]:
        parts.append(f"{report['warning_count']} warnings")
    worst = [c for c in report["checks"] if c["level"] in ("warning", "critical")]
    if worst:
        parts.append("; ".join(f"{c['check']}: {c['detail']}" for c in worst[:3]))
    return " | ".join(parts)
