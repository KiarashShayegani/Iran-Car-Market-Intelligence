#!/usr/bin/env python3
"""One-command sync from GitHub Actions artifacts into the local data layer.

Finds the latest successful collect-etl Actions run, downloads its snapshot
artifact, verifies it (zip CRC + traversal, SQLite integrity_check, manifest
JSON, extraction-version fingerprint), archives the current canonical DB,
imports raw runs through the standard local ingestion (run-level skip +
record-level idempotence; the runner DB is never merged), runs an
idempotence probe (second import must change nothing), and prints a report.

Requires GITHUB_TOKEN (Actions: read) — already in the repo .env.

Examples:
    python scripts/sync_actions_artifact.py --latest
    python scripts/sync_actions_artifact.py --latest --dry-run
    python scripts/sync_actions_artifact.py --run-id 42926888076
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import requests

GITHUB_API = "https://api.github.com"
REPO = "KiarashShayegani/Iran-Car-Market-Intelligence"
EXPECTED_WORKFLOW = "collect-etl.yml"


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}


def _find_run(token: str, run_id: int | None) -> dict:
    if run_id:
        run = requests.get(
            f"{GITHUB_API}/repos/{REPO}/actions/runs/{run_id}",
            headers=_headers(token), timeout=30,
        ).json()
    else:
        runs = requests.get(
            f"{GITHUB_API}/repos/{REPO}/actions/runs",
            headers=_headers(token),
            params={"status": "success", "per_page": 20}, timeout=30,
        ).json()
        run = next(
            (r for r in runs.get("workflow_runs", []) if r["path"].endswith(EXPECTED_WORKFLOW)),
            None,
        )
    if not run or run.get("id") is None:
        raise SystemExit("no successful collect-etl Actions run found")
    if run.get("conclusion") != "success":
        raise SystemExit(f"run {run['id']} conclusion={run.get('conclusion')} — refusing to import")
    return run


def _validate_members(zf: zipfile.ZipFile, root: Path) -> None:
    """Defense-in-depth over zipfile's built-in sanitization: reject absolute
    paths, '..' traversal, and members escaping the extraction root."""
    root_resolved = root.resolve()
    for name in zf.namelist():
        if name.startswith("/"):
            raise SystemExit(f"unsafe absolute zip member: {name}")
        parts = Path(name).parts
        if ".." in parts:
            raise SystemExit(f"zip traversal attempt: {name}")
        if not (root / name).resolve().is_relative_to(root_resolved):
            raise SystemExit(f"zip member escapes root: {name}")


def _download_snapshot(token: str, run_id: int, workdir: Path) -> tuple[Path, dict]:
    arts = requests.get(
        f"{GITHUB_API}/repos/{REPO}/actions/runs/{run_id}/artifacts",
        headers=_headers(token), timeout=30,
    ).json()
    art = next(
        (a for a in arts.get("artifacts", []) if a["name"].startswith("icmi-snapshot")),
        None,
    )
    if art is None:
        raise SystemExit("no icmi-snapshot artifact on this run")
    t0 = time.time()
    r = requests.get(art["archive_download_url"], headers=_headers(token), timeout=300)
    duration = round(time.time() - t0, 1)
    artifact_zip = workdir / "artifact.zip"
    artifact_zip.write_bytes(r.content)
    with zipfile.ZipFile(artifact_zip) as zf:
        if zf.testzip() is not None:
            raise SystemExit("outer artifact zip is corrupt")
        _validate_members(zf, workdir)
        zf.extractall(workdir)
    inner = next(workdir.rglob("icmi_snapshot_*.zip"))
    snapshot_dir = workdir / inner.stem
    with zipfile.ZipFile(inner) as zf:
        if zf.testzip() is not None:
            raise SystemExit("inner snapshot zip is corrupt")
        _validate_members(zf, snapshot_dir)
        zf.extractall(snapshot_dir)
    print(
        f"artifact downloaded: {art['name']} | {len(r.content)//1024} KB compressed | "
        f"{duration}s | extraction verified"
    )
    return snapshot_dir, {"compressed_kb": len(r.content) // 1024, "download_seconds": duration}


def _extraction_fingerprint(snapshot_dir: Path) -> str | None:
    """Extraction version recorded in the snapshot's manifests — used to gate
    derived-data installation when the snapshot was built by incompatible
    extraction logic."""
    manifests = snapshot_dir / "manifests"
    if not manifests.is_dir():
        return None
    for f in sorted(manifests.glob("*.json")):
        try:
            m = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if m.get("extraction_version"):
            return m["extraction_version"]
    return None


def _current_extraction_version() -> str:
    from icmi.core.config import load_sources

    return load_sources().pipeline.extraction_version


def _verify(snapshot_dir: Path) -> tuple[dict, bool]:
    """Returns (report, version_mismatch). Hard errors land in
    report['errors']; a version mismatch is NOT a hard error — raw runs still
    import, derived data is skipped."""
    report: dict = {"errors": []}
    version_mismatch = False

    snap_db = snapshot_dir / "canonical" / "icmi.db"
    if not snap_db.is_file():
        report["errors"].append("no canonical/icmi.db in snapshot")
        return report, version_mismatch
    conn = sqlite3.connect(f"file:{snap_db}?mode=ro", uri=True)
    try:
        report["integrity"] = conn.execute("PRAGMA integrity_check").fetchone()[0]
        report["listings"] = conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
        report["observations"] = conn.execute(
            "SELECT COUNT(*) FROM price_observations").fetchone()[0]
        report["runs"] = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    finally:
        conn.close()
    if report["integrity"] != "ok":
        report["errors"].append(f"snapshot DB integrity: {report['integrity']}")

    manifests = snapshot_dir / "manifests"
    if manifests.is_dir():
        for f in manifests.glob("*.json"):
            try:
                json.loads(f.read_text(encoding="utf-8"))
            except ValueError:
                report["errors"].append(f"corrupt manifest: {f.name}")

    fingerprint = _extraction_fingerprint(snapshot_dir)
    report["extraction_version"] = fingerprint
    current = _current_extraction_version()
    if fingerprint and current and fingerprint != current:
        version_mismatch = True
        report["extraction_version_mismatch"] = (
            f"snapshot={fingerprint} local={current} — derived data will be "
            "skipped; raw runs still import"
        )
    return report, version_mismatch


def _import_snapshot(snapshot_dir: Path, skip_derived: bool = False) -> None:
    import subprocess

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "sync_latest_data.py"),
        "--source", str(snapshot_dir),
    ]
    if skip_derived:
        cmd.append("--skip-derived")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"import failed:\n{result.stdout}\n{result.stderr}")
    print(result.stdout.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--latest", action="store_true", help="latest successful collect-etl run")
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.latest and args.run_id is None:
        parser.error("either --latest or --run-id is required")

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN not set — add it to the repo .env", file=sys.stderr)
        return 2

    run = _find_run(token, args.run_id)
    print(f"actions run: {run['id']} | conclusion={run.get('conclusion')} | {run['created_at']}")

    with tempfile.TemporaryDirectory(prefix="icmi-actions-sync-") as tmp:
        workdir = Path(tmp)
        snapshot_dir, dl_meta = _download_snapshot(token, run["id"], workdir)
        verification, version_mismatch = _verify(snapshot_dir)
        print(f"verification: {json.dumps(verification, default=str)}")
        if verification["errors"]:
            print("ABORTING — verification errors:", verification["errors"], file=sys.stderr)
            return 2
        if version_mismatch:
            print(
                "extraction version mismatch — importing raw runs only; "
                "derived data skipped and will be rebuilt locally"
            )

        from icmi.core.paths import canonical_dir

        db_path = canonical_dir() / "icmi.db"
        before = _counts(db_path)
        already = _already_ingested_run_ids(db_path)
        snapshot_runs = _snapshot_run_ids(snapshot_dir)
        new_runs = [r for r in snapshot_runs if r not in already]
        print(f"snapshot runs: {len(snapshot_runs)} | new to import: {len(new_runs)}")

        if args.dry_run:
            print("dry-run: no changes made")
            return 0

        # pre-import backup of the authoritative DB (owner safeguard)
        backup = db_path.parent / f"pre-import-{run['id']}-{int(time.time())}.db"
        if db_path.is_file():
            src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            dst = sqlite3.connect(backup)
            src.backup(dst)
            dst.close()
            src.close()
            print(f"pre-import DB backup: {backup}")

        _import_snapshot(snapshot_dir, skip_derived=version_mismatch)

        after = _counts(db_path)
        print(f"local DB: {before} -> {after}")
        if after["listings"] < before["listings"] or after["runs"] < before["runs"]:
            print(
                "FATAL: local DB shrank during import — restoring pre-import backup",
                file=sys.stderr,
            )
            shutil.copy2(backup, db_path)
            return 1

        # idempotence probe: second import must change nothing
        _import_snapshot(snapshot_dir, skip_derived=version_mismatch)
        final = _counts(db_path)
        if final != after:
            print("FATAL: second import changed the DB — idempotence violated", file=sys.stderr)
            return 1
        print("idempotence verified: second import changed nothing")

        report = {
            "run_id": run["id"],
            "snapshot_runs": len(snapshot_runs),
            "new_runs_imported": len(new_runs),
            "before": before, "after": after,
            "download": dl_meta,
            "verification": verification,
            "version_mismatch": version_mismatch,
        }
        print("report:", json.dumps(report, default=str))
    return 0


def _counts(db_path: Path) -> dict:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return {
            "listings": conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0],
            "observations": conn.execute(
                "SELECT COUNT(*) FROM price_observations").fetchone()[0],
            "runs": conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
        }
    finally:
        conn.close()


def _already_ingested_run_ids(db_path: Path) -> set[str]:
    if not db_path.is_file():
        return set()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return {r[0] for r in conn.execute("SELECT run_id FROM runs")}
    finally:
        conn.close()


def _snapshot_run_ids(snapshot_dir: Path) -> list[str]:
    raw = snapshot_dir / "raw"
    return sorted(d.name for d in raw.iterdir() if d.is_dir()) if raw.is_dir() else []


if __name__ == "__main__":
    sys.exit(main())
