#!/usr/bin/env python3
"""Publish a portable ICMI data snapshot (the transfer seam).

Bundles manifests, raw runs, analytical Parquet exports, and (optionally) the
canonical SQLite DB into one directory or zip — the artifact format that
GitHub Actions uploads and that `sync_latest_data.py` imports. Transport is
deliberately trivial (local filesystem); a future cloud provider plugs in
behind this same pair of scripts without touching the data layer.

Examples:
    python scripts/publish_data.py                        # manifests + analytical + db
    python scripts/publish_data.py --include-raw --zip    # full snapshot, zipped
    python scripts/publish_data.py --run-id 20261110T...-bama-local --include-raw
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

from icmi.core.paths import analytical_dir, canonical_dir, manifests_dir, raw_dir


def publish(
    out_base: Path,
    *,
    include_raw: bool = False,
    include_db: bool = True,
    include_analytical: bool = True,
    include_manifests: bool = True,
    run_id: str | None = None,
    zip_output: bool = False,
) -> Path:
    from icmi.pipeline.analytical import snapshot_stamp

    stamp = snapshot_stamp()
    suffix = f"-{run_id}" if run_id else ""
    snapshot_dir = out_base / f"icmi_snapshot_{stamp}{suffix}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []

    if include_manifests:
        src = manifests_dir()
        if src.is_dir() and any(src.glob("*.json")):
            for f in src.glob("*.json"):
                if run_id and f.stem != run_id:
                    continue
                dest = snapshot_dir / "manifests" / f.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
            copied.append("manifests")

    if include_raw:
        src_root = raw_dir()
        if src_root.is_dir():
            runs = [run_id] if run_id else [
                d.name for d in sorted(src_root.iterdir()) if d.is_dir()
            ]
            for rid in runs:
                run_src = src_root / rid
                if run_src.is_dir():
                    shutil.copytree(run_src, snapshot_dir / "raw" / rid)
            copied.append("raw")

    if include_analytical:
        src = analytical_dir()
        if src.is_dir() and any(src.glob("*.parquet")):
            (snapshot_dir / "analytical").mkdir(parents=True, exist_ok=True)
            for f in src.glob("*.parquet"):
                shutil.copy2(f, snapshot_dir / "analytical" / f.name)
            copied.append("analytical")

    if include_db:
        db = canonical_dir() / "icmi.db"
        if db.is_file():
            (snapshot_dir / "canonical").mkdir(parents=True, exist_ok=True)
            shutil.copy2(db, snapshot_dir / "canonical" / "icmi.db")
            copied.append("canonical-db")

    if zip_output:
        zip_path = snapshot_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(snapshot_dir.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(snapshot_dir))
        shutil.rmtree(snapshot_dir)
        print(f"snapshot: {zip_path}")
        print(f"included: {', '.join(copied) or '(nothing)'}")
        return zip_path

    print(f"snapshot: {snapshot_dir}")
    print(f"included: {', '.join(copied) or '(nothing)'}")
    return snapshot_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", default=None, help="base output dir (default: data/exports)"
    )
    parser.add_argument("--include-raw", action="store_true", help="include raw runs")
    parser.add_argument(
        "--no-db", action="store_true", help="exclude the canonical SQLite DB"
    )
    parser.add_argument(
        "--no-analytical", action="store_true", help="exclude Parquet exports"
    )
    parser.add_argument("--run-id", default=None, help="restrict to one run")
    parser.add_argument("--zip", action="store_true", help="produce a zip artifact")
    args = parser.parse_args(argv)

    out_base = Path(args.out) if args.out else canonical_dir().parent / "exports"
    publish(
        out_base,
        include_raw=args.include_raw,
        include_db=not args.no_db,
        include_analytical=not args.no_analytical,
        run_id=args.run_id,
        zip_output=args.zip,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
