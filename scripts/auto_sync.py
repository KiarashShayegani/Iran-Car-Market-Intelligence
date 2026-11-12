#!/usr/bin/env python3
"""Automatic artifact retrieval + catch-up sync (lock-safe).

One command suitable for cron / systemd timers / Task Scheduler / startup
catch-up. Wraps the manual sync path with:

- a process lock (overlapping scheduled + manual runs serialize or skip),
- chronological import of ALL pending artifacts (--pending),
- derived-data refresh when new data landed (export documents index),
- a health rollup + journal events for every outcome.

Never touches the authoritative DB except through the standard verified
import path. Exit codes: 0 = nothing pending or all imported cleanly,
1 = failures to investigate (details in logs/journal.jsonl).

Examples:
    python scripts/auto_sync.py            # scheduled/catch-up run
    python scripts/auto_sync.py --dry-run  # report pending only
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from icmi.core.locking import FileLock, LockHeldError
from icmi.ops.health import build_health_report
from icmi.ops.journal import journal_event


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="report pending artifacts without importing")
    parser.add_argument("--no-derived", action="store_true",
                        help="skip export/documents/index refresh even if new data landed")
    parser.add_argument("--health-only", action="store_true",
                        help="run the health rollup only (no GitHub calls)")
    args = parser.parse_args(argv)

    lock = FileLock(Path("data/.sync.lock"), holder="auto_sync")
    try:
        lock.acquire()
    except LockHeldError as exc:
        print(f"another sync/pipeline is running ({exc}) — nothing done")
        journal_event("sync", "skipped_lock_held", level="warning", detail=str(exc))
        return 0

    try:
        return _run(args)
    finally:
        lock.release()


def _run(args: argparse.Namespace) -> int:
    script = Path(__file__).parent
    failures = 0

    if args.health_only:
        health = build_health_report()
        journal_event("health", "rollup",
                      level="critical" if health["critical_count"] else
                      ("warning" if health["warning_count"] else "info"),
                      critical=health["critical_count"],
                      warnings=health["warning_count"])
        print(f"health: {health['critical_count']} critical, "
              f"{health['warning_count']} warnings")
        return 1 if health["critical_count"] else 0

    started = journal_event("sync", "auto_sync_started", dry_run=args.dry_run)

    # 1. import everything pending (chronological, ledger-deduplicated)
    cmd = [sys.executable, str(script / "sync_actions_artifact.py")]
    cmd.append("--dry-run" if args.dry_run else "--pending")
    result = subprocess.run(cmd, capture_output=True, text=True)
    tail = "\n".join((result.stdout or result.stderr).strip().splitlines()[-12:])
    print(tail)
    if result.returncode != 0:
        failures += 1
        journal_event("sync", "auto_sync_import_failed", level="critical",
                      exit_code=result.returncode)
    else:
        imported = sum(
            line.count("import_complete") for line in (result.stdout or "").splitlines()
        )
        journal_event("sync", "auto_sync_import_done", imported=imported)

    # 2. refresh derived views if new data landed (cheap guard: ingest
    #    stage already ran inside the import; views need explicit rebuild)
    fresh = _new_data(result.stdout or "")
    if not args.dry_run and not args.no_derived and failures == 0 and fresh:
        for stage in ("export", "documents", "index"):
            r = subprocess.run(
                [sys.executable, str(script / "pipeline.py"), stage],
                capture_output=True, text=True,
            )
            line = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()[-1:] or [""]
            print(f"{stage}: [{'ok' if r.returncode == 0 else 'FAIL'}] {line[0][:120]}")
            journal_event("sync", f"derived_{stage}",
                          level="warning" if r.returncode else "info",
                          exit_code=r.returncode, detail=line[0][:200])
            if r.returncode != 0:
                failures += 1

    # 3. health rollup
    health = build_health_report()
    journal_event("health", "rollup",
                  level="critical" if health["critical_count"] else
                  ("warning" if health["warning_count"] else "info"),
                  critical=health["critical_count"],
                  warnings=health["warning_count"],
                  ts_started=started["ts"])
    print(f"health: {health['critical_count']} critical, "
          f"{health['warning_count']} warnings "
          f"(details: python scripts/ops_status.py --problems)")

    journal_event("sync", "auto_sync_finished",
                  level="warning" if failures else "info", failures=failures)
    return 1 if failures else 0


def _new_data(sync_output: str) -> bool:
    """True when the sync output shows an import that added or changed data.

    The ingest stage reports 'new=N updated=M observations+=K' — any
    non-zero value means derived views are stale. Ledger short-circuits
    ('already imported') produce no such line.
    """
    return any(
        int(m) > 0
        for m in re.findall(r"new=(\d+)", sync_output)
    ) or any(
        int(m) > 0
        for m in re.findall(r"updated=(\d+)", sync_output)
    )


if __name__ == "__main__":
    sys.exit(main())
