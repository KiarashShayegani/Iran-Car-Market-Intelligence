"""Operational journal — persistent structured JSONL event log.

One line per event at ``logs/journal.jsonl`` with size-based rotation
(keeps 3 rotations of 5 MB). Events describe collection runs, artifact
sync/import outcomes, and health-check rollups so a developer can answer
"what happened, where, and when" without a dashboard. Never logs secrets,
tokens, or request headers — only operational facts already recorded in
manifests and reports.

Events follow the schema::

    {"ts": "...", "kind": "collection|sync|health", "event": "...",
     "level": "info|warning|critical", ...context fields}
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

JOURNAL_NAME = "journal.jsonl"
MAX_BYTES = 5 * 1024 * 1024
KEEP_ROTATIONS = 3

_LEVELS = ("info", "warning", "critical")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def journal_dir() -> Path:
    d = _repo_root() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def journal_path() -> Path:
    return journal_dir() / JOURNAL_NAME


def _rotate() -> None:
    path = journal_path()
    if not path.is_file() or path.stat().st_size < MAX_BYTES:
        return
    for i in range(KEEP_ROTATIONS, 0, -1):
        src = path.with_suffix(f".jsonl.{i}")
        if src.is_file():
            if i == KEEP_ROTATIONS:
                src.unlink()
            else:
                src.replace(path.with_suffix(f".jsonl.{i + 1}"))
    path.replace(path.with_suffix(".jsonl.1"))


def journal_event(
    kind: str, event: str, level: str = "info", **fields: object
) -> dict:
    """Append one structured event; returns the written record.

    Failures to write never raise — monitoring must not break operations.
    """
    if level not in _LEVELS:
        level = "info"
    record = {
        "ts": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "kind": kind,
        "event": event,
        "level": level,
        **fields,
    }
    try:
        _rotate()
        with journal_path().open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass  # journaling is best-effort by design
    return record


def journal_collection_from_manifest(manifest: dict) -> dict:
    """Log one collection run from its finalized manifest."""
    quality = manifest.get("quality", {})
    return journal_event(
        "collection",
        "run_completed" if manifest.get("status") == "success" else "run_failed",
        level="info" if manifest.get("status") == "success" else "warning",
        run_id=manifest.get("run_id"),
        platform=manifest.get("platform"),
        mode=manifest.get("execution_mode"),
        collection=manifest.get("collection"),
        extraction_version=manifest.get("extraction_version"),
        status=manifest.get("status"),
        started_at=manifest.get("started_at"),
        completed_at=manifest.get("completed_at"),
        pages=manifest.get("pages", {}),
        records_extracted=manifest.get("records", {}).get("extracted"),
        duplicates_within_run=manifest.get("records", {}).get(
            "duplicates_within_run"
        ),
        p0_coverage=quality.get("p0_coverage_pct"),
        p1_coverage=quality.get("p1_coverage_pct"),
        failed_requests=len(manifest.get("failed_requests", [])),
        skip_reasons=manifest.get("skip_reasons", {}),
    )


def journal_tail(n: int = 20, level: str | None = None) -> list[dict]:
    """Last n events (optionally filtered by level), oldest first."""
    path = journal_path()
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for line in reversed(lines):
        if len(out) >= n:
            break
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if level and rec.get("level") != level:
            continue
        out.append(rec)
    return list(reversed(out))


def journal_age_seconds() -> float | None:
    """Seconds since the journal last changed (None if it never did)."""
    path = journal_path()
    if not path.is_file():
        return None
    return time.time() - path.stat().st_mtime


def _env_probe() -> str:  # pragma: no cover - diagnostics helper
    return os.environ.get("ICMI_DATA_DIR", "")
