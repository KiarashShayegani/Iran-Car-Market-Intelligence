"""Operational status report — answers the owner's ops questions locally.

Built from data that already exists (manifests, runs table, listings, index
meta, snapshot exports) — no new infrastructure. Exposed as a CLI
(scripts/ops_status.py) and an API endpoint (/ops-status).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC
from pathlib import Path

from icmi.core.paths import (
    canonical_dir,
    data_dir,
    index_dir,
    manifests_dir,
    raw_dir,
)

KNOWN_PLATFORMS = ("bama", "sheypoor", "hamrah_mechanic", "divar", "karnameh")


def build_ops_status(store: sqlite3.Connection | None = None) -> dict:
    """Assemble the operational picture. `store` may be an open
    CanonicalStore; otherwise the default DB is opened read-only."""
    from icmi.ops.health import build_health_report  # local import: no cycle

    manifests = _collect_manifests()
    db_status = _database_status(store)
    return {
        "generated_at": _now_iso(),
        "platforms": _platform_status(manifests, db_status),
        "database": db_status,
        "health": build_health_report(store),
        "search_index": _index_status(),
        "backup": _backup_status(),
        "image_cache": _image_cache_status(),
        "notes": [
            "Corpus is a small experimental sample — never read these numbers as market coverage.",
        ],
    }


def _collect_manifests() -> list[dict]:
    out = []
    for path in sorted(manifests_dir().glob("*.json")):
        try:
            m = json.loads(path.read_text(encoding="utf-8"))
            out.append(
                {
                    "run_id": m.get("run_id", path.stem),
                    "platform": m.get("platform"),
                    "mode": m.get("execution_mode"),
                    "collection": m.get("collection"),
                    "status": m.get("status"),
                    "started_at": m.get("started_at"),
                    "completed_at": m.get("completed_at"),
                    "records": m.get("records", {}).get("extracted"),
                    "duplicates": m.get("records", {}).get("duplicates_within_run"),
                    "pages": m.get("pages", {}),
                    "p0_coverage": m.get("quality", {}).get("p0_coverage_pct"),
                    "p1_coverage": m.get("quality", {}).get("p1_coverage_pct"),
                    "failed_requests": len(m.get("failed_requests", [])),
                    "skip_reasons": m.get("skip_reasons", {}),
                }
            )
        except (ValueError, OSError):
            out.append({"run_id": path.stem, "status": "corrupt_manifest"})
    return out


def _platform_status(manifests: list[dict], db_status: dict) -> list[dict]:
    rows = []
    for platform in KNOWN_PLATFORMS:
        runs = [m for m in manifests if m.get("platform") == platform]
        last = max(runs, key=lambda m: m.get("started_at") or "") if runs else None
        rows.append(
            {
                "platform": platform,
                "listings": db_status["listings_per_platform"].get(platform, 0),
                "last_run_id": last.get("run_id") if last else None,
                "last_run_status": last.get("status") if last else None,
                "last_run_at": last.get("completed_at") if last else None,
                "last_run_mode": last.get("mode") if last else None,
                "last_records": last.get("records") if last else None,
                "last_p0_coverage": last.get("p0_coverage") if last else None,
                "last_failed_requests": last.get("failed_requests") if last else None,
                "total_runs": len(runs),
            }
        )
    return rows


def _database_status(store: sqlite3.Connection | None) -> dict:
    own = False
    if store is None:
        db_path = canonical_dir() / "icmi.db"
        if not db_path.is_file():
            return {"exists": False}
        store = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        own = True
    try:
        per_platform = {
            row[0]: row[1]
            for row in store.execute(
                "SELECT source_platform, COUNT(*) FROM listings GROUP BY 1"
            )
        }
        integrity = store.execute("PRAGMA quick_check").fetchone()[0]
        return {
            "exists": True,
            "integrity_check": integrity,
            "listings": store.execute("SELECT COUNT(*) FROM listings").fetchone()[0],
            "price_observations": store.execute(
                "SELECT COUNT(*) FROM price_observations"
            ).fetchone()[0],
            "runs_recorded": store.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
            "listings_per_platform": per_platform,
        }
    finally:
        if own:
            store.close()


def _index_status() -> dict:
    meta_path = index_dir() / "index_meta.json"
    if not meta_path.is_file():
        return {"exists": False}
    try:
        info = json.loads(meta_path.read_text(encoding="utf-8"))
    except ValueError:
        return {"exists": True, "corrupt": True}
    staleness = "unknown"
    db_path = canonical_dir() / "icmi.db"
    if db_path.is_file() and info.get("built_at"):
        store = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            newest = store.execute("SELECT MAX(first_seen_at) FROM listings").fetchone()[0]
            if newest:
                staleness = (
                    "fresh"
                    if info["built_at"] >= newest
                    else "stale — rebuild with scripts/rebuild_search_assets.py"
                )
        finally:
            store.close()
    return {
        "exists": True,
        "built_at": info.get("built_at"),
        "vectors": info.get("vector_count"),
        "model_id": info.get("model_id"),
        "staleness": staleness,
    }


def _backup_status() -> dict:
    exports = data_dir() / "exports"
    snapshots = (
        sorted(exports.glob("*.zip"), key=lambda p: p.stat().st_mtime)
        if exports.is_dir()
        else []
    )
    return {
        "snapshots": len(snapshots),
        "latest_snapshot": snapshots[-1].name if snapshots else None,
        "latest_snapshot_at": (
            snapshots[-1].stat().st_mtime and _mtime_iso(snapshots[-1])
        )
        if snapshots
        else None,
    }


def _image_cache_status() -> dict:
    cache_root = data_dir() / "image-cache"
    files = list(cache_root.rglob("*.bin")) if cache_root.is_dir() else []
    return {
        "cached_images": len(files),
        "total_mb": round(sum(f.stat().st_size for f in files) / (1024 * 1024), 2),
    }


def _raw_runs() -> int:
    raw = raw_dir()
    return (
        len([d for d in raw.iterdir() if d.is_dir()])
        if raw.is_dir()
        else 0
    )


def _mtime_iso(path: Path) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat(
        timespec="seconds"
    )


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now(tz=UTC).isoformat(timespec="seconds")
