"""Portable advisory file lock — prevents overlapping pipeline runs.

Cross-platform (Linux + Windows): atomic create via O_CREAT|O_EXCL; a lock
older than `stale_after_seconds` is considered abandoned and may be taken
over (crash recovery). Never used for multi-process coordination beyond
best-effort safety — the underlying stages are idempotent anyway.
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC
from pathlib import Path


class LockHeldError(Exception):
    """Another pipeline run holds the lock (and it is not stale)."""


class FileLock:
    def __init__(
        self,
        path: Path,
        holder: str = "pipeline",
        stale_after_seconds: int = 6 * 3600,
    ) -> None:
        self.path = path
        self.holder = holder
        self.stale_after_seconds = stale_after_seconds
        self._acquired = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(
                    fd,
                    json.dumps(
                        {
                            "holder": self.holder,
                            "pid": os.getpid(),
                            "acquired_at": utc_iso(),
                        }
                    ).encode(),
                )
                os.close(fd)
                self._acquired = True
                return
            except FileExistsError:
                if self._is_stale():
                    # abandoned lock (crash) — take over
                    self.path.unlink(missing_ok=True)
                    continue
                raise LockHeldError(
                    f"Another pipeline run holds {self.path}. "
                    "If you are sure no run is active, delete the lock file."
                ) from None

    def release(self) -> None:
        if self._acquired:
            self.path.unlink(missing_ok=True)
            self._acquired = False

    def _is_stale(self) -> bool:
        try:
            age = time.time() - self.path.stat().st_mtime
            return age > self.stale_after_seconds
        except OSError:
            return True

    def __enter__(self) -> FileLock:
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def utc_iso() -> str:
    from datetime import datetime

    return datetime.now(tz=UTC).isoformat(timespec="seconds")
