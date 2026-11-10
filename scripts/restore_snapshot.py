#!/usr/bin/env python3
"""Restore an ICMI snapshot (backup) into the local data layer — with
verification, dry-run mode, and automatic archiving of current state.

Restore semantics (documented, deliberate):
- A restore is a ROLLBACK: the canonical DB, manifests, analytical exports,
  and documents/index come from the snapshot; the current canonical DB is
  archived to data/pre-restore-<stamp>/ first (never silently overwritten).
- Raw runs are MERGED (copied per run-id, never overwriting newer local runs
  with the same id) — raw data is append-only immutable history.
- The runner DB inside the snapshot is used as-is (it IS the authoritative
  backup); no merging of databases ever happens (owner rule).

Verification before anything is replaced:
- zip/dir structural check
- SQLite PRAGMA integrity_check on the snapshot DB
- manifests parse as JSON
- record counts reported for the owner to compare

Examples:
    python scripts/restore_snapshot.py --source data/exports/icmi_snapshot_X.zip --dry-run
    python scripts/restore_snapshot.py --source data/exports/icmi_snapshot_X.zip
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import tempfile
import zipfile
from datetime import UTC
from pathlib import Path

from icmi.core.paths import data_dir


def _validate_zip_members(zf: zipfile.ZipFile, root: Path) -> None:
    """Defense-in-depth over Python's built-in sanitization (final G3
    verification): reject absolute paths, '..' traversal, and any member
    whose resolved location escapes the extraction root."""
    root_resolved = root.resolve()
    for name in zf.namelist():
        if name.startswith("/") or "\\" in name.split("/")[0:1]:
            raise ValueError(f"unsafe absolute zip member: {name}")
        parts = Path(name).parts
        if ".." in parts:
            raise ValueError(f"zip traversal attempt: {name}")
        target = (root / name).resolve()
        if not target.is_relative_to(root_resolved):
            raise ValueError(f"zip member escapes root: {name}")


def _extract(source: Path, workdir: Path) -> Path:
    if source.is_file() and source.suffix == ".zip":
        with zipfile.ZipFile(source) as zf:
            bad = zf.testzip()
            if bad is not None:
                raise ValueError(f"corrupt zip member: {bad}")
            _validate_zip_members(zf, workdir)
            zf.extractall(workdir)
        return workdir
    if source.is_dir():
        return source
    raise FileNotFoundError(f"snapshot not found: {source}")


def _verify(snapshot_root: Path) -> dict:
    report: dict = {"manifests": 0, "raw_runs": 0, "db": None, "errors": []}

    manifests = snapshot_root / "manifests"
    if manifests.is_dir():
        for f in manifests.glob("*.json"):
            try:
                json.loads(f.read_text(encoding="utf-8"))
                report["manifests"] += 1
            except ValueError:
                report["errors"].append(f"corrupt manifest: {f.name}")

    raw = snapshot_root / "raw"
    if raw.is_dir():
        report["raw_runs"] = len([d for d in raw.iterdir() if d.is_dir()])

    db_path = snapshot_root / "canonical" / "icmi.db"
    if db_path.is_file():
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            report["db"] = {
                "integrity_check": conn.execute("PRAGMA integrity_check").fetchone()[0],
                "listings": conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0],
                "price_observations": conn.execute(
                    "SELECT COUNT(*) FROM price_observations"
                ).fetchone()[0],
                "runs": conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
            }
        finally:
            conn.close()
    else:
        report["errors"].append("snapshot has no canonical/icmi.db")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="snapshot zip or directory")
    parser.add_argument(
        "--dry-run", action="store_true", help="verify + report only, change nothing"
    )
    parser.add_argument("--skip-index", action="store_true", help="do not restore index/")
    args = parser.parse_args(argv)

    source = Path(args.source).expanduser().resolve()
    data_root = data_dir()

    with tempfile.TemporaryDirectory(prefix="icmi-restore-") as tmp:
        root = _extract(source, Path(tmp))
        verification = _verify(root)

        print(f"source: {source}")
        print(f"verification: manifests={verification['manifests']} "
              f"raw_runs={verification['raw_runs']} db={verification['db']}")
        if verification["errors"]:
            print("errors:", verification["errors"])
            return 2
        if verification["db"] is None:
            print("refusing: no canonical DB in snapshot")
            return 2

        if args.dry_run:
            print("dry-run: no changes made")
            return 0

        stamp = __import__("datetime").datetime.now(tz=__import__("datetime").timezone.utc)
        stamp = stamp.strftime("%Y%m%dT%H%M%S")
        archive = data_root / f"pre-restore-{stamp}"
        archive.mkdir(parents=True, exist_ok=True)

        # archive current canonical DB (the rollback point)
        current_db = data_root / "canonical" / "icmi.db"
        if current_db.is_file():
            (archive / "canonical").mkdir(parents=True, exist_ok=True)
            shutil.copy2(current_db, archive / "canonical" / "icmi.db")
            print(f"current DB archived to {archive / 'canonical' / 'icmi.db'}")

        # canonical: replace (rollback semantics)
        snap_db = root / "canonical" / "icmi.db"
        current_db.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snap_db, current_db)
        print("canonical DB restored")

        # manifests + analytical: replace
        for name in ("manifests", "analytical"):
            src = root / name
            if src.is_dir():
                dest = data_root / name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(src, dest)
                print(f"{name} restored ({len(list(src.rglob('*.*')))} files)")

        # index: replace unless skipped
        idx = root / "index"
        if idx.is_dir() and not args.skip_index:
            dest = data_root / "index"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(idx, dest)
            print("index restored")

        # raw: merge per run-id (never overwrite a newer local run with same id)
        raw_src = root / "raw"
        if raw_src.is_dir():
            raw_dest = data_root / "raw"
            raw_dest.mkdir(parents=True, exist_ok=True)
            for run_dir in raw_src.iterdir():
                target = raw_dest / run_dir.name
                if not target.exists():
                    shutil.copytree(run_dir, target)
                    print(f"raw run copied: {run_dir.name}")
                else:
                    print(f"raw run kept (exists locally): {run_dir.name}")

        report = {
            "restored_at": stamp_iso(),
            "source": str(source),
            "verification": verification,
            "steps": [
                "canonical DB replaced (pre-restore copy archived)",
                "manifests replaced",
                "analytical replaced",
                "index restored" if not args.skip_index else "index skipped (--skip-index)",
                "raw runs merged (never overwritten)",
            ],
        }
        report_path = data_root / f"restore-report-{stamp_iso()}.json"
        report_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"restore report: {report_path}")
        print(
            "restore complete — if interrupted, rerun the same command "
            "(idempotent); the pre-restore archive is kept either way."
        )
        return 0


def stamp_iso() -> str:
    from datetime import datetime

    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")


if __name__ == "__main__":
    sys.exit(main())
