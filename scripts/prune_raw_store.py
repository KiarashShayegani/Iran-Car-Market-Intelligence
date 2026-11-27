#!/usr/bin/env python3
"""Raw-payload retention policy (owner decision 2026-11-22 — "Option B").

The canonical SQLite store is the authoritative data layer: search, browse,
dashboard, estimate and ask all read it (plus the vector index) — the raw
payload store (data/raw/<run_id>/...) is only an ETL re-derivation input.
It is also the growth driver (~19 KB/ad per run, re-archived every sweep).

Policy (keeps the historic trail the product needs):
- price_observations / listings: NEVER pruned — the price time-series is the
  product's compounding value (dashboard, trends, estimate).
- raw payloads older than --min-age-days (default 60): archived per run to
  data/raw/archive/<run_id>.tar.gz (default, reversible via tar) or DELETED
  with --delete (unrecoverable). Recent raws stay for re-derivation safety.

Run weekly (or after every banking round). Every action is journaled.

Usage:
    python scripts/prune_raw_store.py --dry-run
    python scripts/prune_raw_store.py                      # archive >60d
    python scripts/prune_raw_store.py --min-age-days 7     # archive >7d
    python scripts/prune_raw_store.py --delete             # DELETE >60d
"""

from __future__ import annotations

import argparse
import re
import shutil
import tarfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

_RUN_DATE_RE = re.compile(r"^(\d{8})T(\d{6})")


def _run_datetime(run_dir: Path) -> datetime | None:
    m = _RUN_DATE_RE.match(run_dir.name)
    if m:
        try:
            naive = datetime.strptime(f"{m.group(1)}{m.group(2)}", "%Y%m%d%H%M%S")
            return naive.replace(tzinfo=UTC)
        except ValueError:
            return None
    # fallback: directory modification time
    mtime = run_dir.stat().st_mtime
    return datetime.fromtimestamp(mtime, tz=UTC)


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-age-days", type=int, default=60,
                        help="archive/delete run dirs older than this (default 60)")
    parser.add_argument("--delete", action="store_true",
                        help="DELETE old run dirs instead of tar.gz archiving")
    parser.add_argument("--dry-run", action="store_true",
                        help="report planned actions without touching anything")
    args = parser.parse_args()

    from icmi.ops.journal import journal_event

    raw_root = Path("data/raw")
    archive_root = raw_root / "archive"
    if not raw_root.is_dir():
        print("no data/raw directory — nothing to prune")
        return 0

    cutoff = datetime.now(tz=UTC) - timedelta(
        days=args.min_age_days
    )

    candidates: list[tuple[Path, datetime]] = []
    total_old, total_recent = 0, 0
    for run_dir in sorted(raw_root.iterdir()):
        if not run_dir.is_dir():
            continue
        run_dt = _run_datetime(run_dir)
        size = _dir_size(run_dir)
        if run_dt is not None and run_dt < cutoff:
            candidates.append((run_dir, run_dt))
            total_old += size
        else:
            total_recent += size

    mode = "DELETE" if args.delete else "ARCHIVE (tar.gz, reversible)"
    print(f"retention: raw payloads older than {args.min_age_days} days → {mode}")
    print(f"cutoff: {cutoff:%Y-%m-%dT%H:%M:%SZ}")
    print(f"run dirs: {len(candidates)} old of {len(list(raw_root.glob('*')))} total")
    print(f"size: old={total_old / 1e6:.1f} MB  recent(kept)={total_recent / 1e6:.1f} MB")

    if not candidates:
        print("nothing to prune")
        return 0
    if args.dry_run:
        for run_dir, run_dt in candidates[:10]:
            print(f"  would {mode.lower()}: {run_dir.name} ({run_dt:%Y-%m-%d})")
        if len(candidates) > 10:
            print(f"  … and {len(candidates) - 10} more")
        return 0

    saved_bytes = 0
    archive_root.mkdir(parents=True, exist_ok=True)
    done = 0
    for run_dir, _run_dt in candidates:
        if args.delete:
            shutil.rmtree(run_dir)
            saved_bytes += _dir_size(run_dir) if run_dir.exists() else 0
        else:
            archive_path = archive_root / f"{run_dir.name}.tar.gz"
            if archive_path.exists():
                print(f"  skip (archive exists): {archive_path.name}")
                shutil.rmtree(run_dir)
                continue
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(run_dir, arcname=run_dir.name)
            shutil.rmtree(run_dir)
            saved_bytes += _dir_size(archive_path)  # what REMAINS on disk
        done += 1

    after = _dir_size(raw_root) if raw_root.exists() else 0
    print(f"processed {done} run dirs ({mode})")
    print(f"data/raw after: {after / 1e6:.1f} MB")
    journal_event(
        "raw_prune", "completed",
        mode=mode, runs=done, min_age_days=args.min_age_days,
        old_mb=round(total_old / 1e6, 1), raw_after_mb=round(after / 1e6, 1),
    )
    print(f"journal: raw_prune logged at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
