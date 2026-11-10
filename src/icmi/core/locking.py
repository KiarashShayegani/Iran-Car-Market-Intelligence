"""Portable advisory file lock — prevents overlapping pipeline runs.

Cross-platform (Linux + Windows). Takeover policy (final G3 verification,
owner-reviewed):

1. Lock exists -> read holder metadata (pid, hostname, acquired_at).
2. Same hostname AND pid provably alive -> HELD, regardless of age — a
   genuinely long-running process can never be hijacked by the stale rule.
3. Same hostname AND pid provably dead -> abandoned (crash) -> takeover.
4. Different hostname (lock copied between machines) or unparseable content
   -> fall back to the age rule (stale_after_seconds).
5. Otherwise -> LockHeldError with a clear message.

`_pid_alive` is portable: POSIX uses kill(pid, 0); Windows uses
OpenProcess/GetExitCodeProcess (never os.kill on Windows — any signal value
there terminates the process). The Windows branch is written but untested on
real Windows hardware (honest unknown, G6 checklist).
"""

from __future__ import annotations

import json
import os
import socket
import time
from datetime import UTC
from pathlib import Path


class LockHeldError(Exception):
    """Another pipeline run holds the lock (and its holder is alive)."""


def utc_iso() -> str:
    from datetime import datetime

    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _pid_alive(pid: int) -> bool | None:
    """True = alive, False = dead, None = cannot determine."""
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return False
            exit_code = ctypes.c_ulong()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            kernel32.CloseHandle(handle)
            return bool(ok) and exit_code.value == STILL_ACTIVE
        except Exception:  # noqa: BLE001 — liveness probe must never crash the pipeline
            return None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists, owned by another user
    except OSError:
        return None


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
                            "hostname": socket.gethostname(),
                            "acquired_at": utc_iso(),
                        }
                    ).encode(),
                )
                os.close(fd)
                self._acquired = True
                return
            except FileExistsError:
                verdict = self._existing_lock_verdict()
                if verdict == "held":
                    raise LockHeldError(
                        f"Another pipeline run holds {self.path} "
                        "(holder process is alive). If this is wrong, delete the lock file."
                    ) from None
                # stale → takeover and retry the exclusive create
                self.path.unlink(missing_ok=True)
                continue

    def _existing_lock_verdict(self) -> str:
        """'held' | 'stale' — after inspecting holder liveness/age."""
        try:
            meta = json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            meta = None

        if isinstance(meta, dict) and isinstance(meta.get("pid"), int):
            same_host = meta.get("hostname") in (None, socket.gethostname())
            alive = _pid_alive(meta["pid"])
            if alive is True and same_host:
                return "held"  # genuinely running — never hijack by age
            if alive is False:
                return "stale"  # crash recovery, regardless of age
            # alive unknown (probe failed) or different host -> age fallback
        age = time.time() - self.path.stat().st_mtime
        return "stale" if age > self.stale_after_seconds else "held"

    def release(self) -> None:
        if self._acquired:
            self.path.unlink(missing_ok=True)
            self._acquired = False

    def __enter__(self) -> FileLock:
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()
