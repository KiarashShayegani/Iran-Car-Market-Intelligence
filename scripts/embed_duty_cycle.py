#!/usr/bin/env python3
"""Thermally-safe embedding runner: encode bursts with pause/abort/resume.

Wraps scripts/build_index.py in a duty cycle so a large corpus encodes on
thermally-constrained hardware:

- waits for cooldown (package temp < --cool-at) before each burst
- launches build_index.py (nice'd, single torch thread) and monitors temp
  every 10s; kills the burst at --abort-at (checkpointed encoding keeps the
  progress up to the last 1,500-chunk slice — atomic save per slice)
- repeats until index_meta.json vector_count == documents.jsonl line count
- logs every burst to logs/journal.jsonl via the ops journal

Exit 0 = fully indexed. Exit 1 = gave up (max_rounds) with progress durable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COOL_AT = 78.0
RESUME_AT = 62.0
ABORT_AT = 85.0


def package_temp() -> float | None:
    try:
        out = subprocess.run(
            ["sensors"], capture_output=True, text=True, timeout=10
        ).stdout
        for line in out.splitlines():
            if line.startswith("Package id 0:"):
                token = line.split("+")[1].split("°")[0]
                return float(token)
    except Exception:  # noqa: BLE001 — monitoring must not break the run
        return None
    return None


def progress() -> tuple[int, int]:
    """(indexed_chunks, total_chunks) from the index meta + documents file."""
    meta = REPO / "data" / "index" / "index_meta.json"
    docs = REPO / "data" / "index" / "documents.jsonl"
    total = sum(1 for _ in docs.open(encoding="utf-8")) if docs.is_file() else 0
    indexed = 0
    if meta.is_file():
        try:
            indexed = int(json.loads(meta.read_text(encoding="utf-8"))["vector_count"])
        except (ValueError, KeyError, OSError):
            indexed = 0
    return indexed, total


def refresh_documents() -> None:
    """Regenerate data/index/documents.jsonl from the live store.

    build_index.py embeds straight from the canonical store, so
    documents.jsonl is a side artifact — if it is stale (new runs banked
    since the last build), the total-vs-indexed progress check would be
    wrong and the duty cycle would silently no-op. Refreshing first makes
    the target reflect the data that actually needs embedding."""
    import sys

    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "build_documents.py")],
        cwd=REPO, check=False,
    )


def wait_cooldown() -> None:
    while True:
        t = package_temp()
        if t is None or t < RESUME_AT:
            return
        print(f"  cooling: {t:.0f}C (resume at <{RESUME_AT:.0f}C)", flush=True)
        time.sleep(30)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-rounds", type=int, default=60)
    args = parser.parse_args(argv)

    from icmi.ops.journal import journal_event

    # target must reflect the live store, not a stale documents.jsonl
    refresh_documents()
    journal_event("embedding", "documents_refreshed")

    indexed, total = progress()
    print(f"duty-cycle embedding: {indexed}/{total} chunks indexed at start")

    for round_no in range(1, args.max_rounds + 1):
        indexed, total = progress()
        if total and indexed >= total:
            print(f"complete: {indexed}/{total} chunks indexed")
            journal_event("embedding", "complete", indexed=indexed, total=total)
            return 0

        wait_cooldown()
        t0 = package_temp()
        print(f"round {round_no}: starting burst from {indexed}/{total} "
              f"(temp {t0:.0f}C)", flush=True)
        journal_event("embedding", "burst_started", round=round_no,
                      indexed=indexed, total=total)

        proc = subprocess.Popen(
            ["nice", "-n", "10", "uv", "run", "python",
             str(REPO / "scripts" / "build_index.py")],
            cwd=REPO,
            env={"PATH": "/usr/bin:/bin:/usr/local/bin",
                 "HOME": str(REPO), "OMP_NUM_THREADS": "2",
                 "VIRTUAL_ENV": str(REPO / ".venv")},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        aborted = False
        while proc.poll() is None:
            time.sleep(10)
            t = package_temp()
            if t is not None and t >= ABORT_AT:
                proc.kill()
                aborted = True
                print(f"  ABORT at {t:.0f}C — checkpoint holds, cooling", flush=True)
                journal_event("embedding", "burst_aborted", level="warning",
                              round=round_no, temp=t)
                break

        if not aborted:
            out = proc.stdout.read() if proc.stdout else ""
            tail = out.strip().splitlines()[-2:]
            for line in tail:
                print(f"  build: {line}", flush=True)
            indexed, total = progress()
            if total and indexed >= total:
                print(f"complete: {indexed}/{total} chunks indexed")
                journal_event("embedding", "complete", indexed=indexed, total=total)
                return 0
            journal_event("embedding", "burst_finished", round=round_no,
                          indexed=indexed, total=total)
        else:
            wait_cooldown()

    print(f"gave up after {args.max_rounds} rounds — progress is durable, "
          "rerun to continue")
    indexed, total = progress()
    journal_event("embedding", "gave_up", level="warning",
                  rounds=args.max_rounds, indexed=indexed, total=total)
    return 1


if __name__ == "__main__":
    sys.exit(main())
