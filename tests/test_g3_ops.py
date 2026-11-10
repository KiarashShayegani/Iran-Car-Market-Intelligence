"""Tests for G3: file locking, unified pipeline runner stages, image proxy
hardening, and restore verification (all offline)."""

import json

import pytest

from icmi.api.image_proxy import ImageCache, ImageProxyError
from icmi.core.locking import FileLock, LockHeldError
from icmi.pipeline.runner import PipelineRunner

# --- locking ---------------------------------------------------------------------


def test_lock_acquire_and_release(tmp_path):
    lock = FileLock(tmp_path / "lock", holder="test")
    lock.acquire()
    assert lock.path.is_file()
    lock.release()
    assert not lock.path.exists()


def test_lock_second_acquirer_blocked(tmp_path):
    lock1 = FileLock(tmp_path / "lock", holder="one")
    lock1.acquire()
    with pytest.raises(LockHeldError, match="holds"):
        FileLock(tmp_path / "lock", holder="two").acquire()
    lock1.release()
    FileLock(tmp_path / "lock", holder="two").acquire()  # now fine


def _dead_pid() -> int:
    import subprocess
    import sys

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()  # ensure it has exited
    return proc.pid


def test_lock_stale_takeover_requires_dead_holder(tmp_path):
    """New policy: stale takeover requires a provably DEAD holder pid —
    mtime age alone must never hijack a live process."""
    import json
    import os
    import time

    lock1 = FileLock(tmp_path / "lock", holder="dead", stale_after_seconds=1)
    lock1.acquire()
    meta = json.loads(lock1.path.read_text(encoding="utf-8"))
    meta["pid"] = _dead_pid()  # holder died mid-run
    lock1.path.write_text(json.dumps(meta), encoding="utf-8")
    old = time.time() - 7200
    os.utime(lock1.path, (old, old))
    lock2 = FileLock(tmp_path / "lock", holder="two", stale_after_seconds=3600)
    lock2.acquire()  # dead holder → takeover allowed
    lock2.release()


def test_lock_live_holder_never_hijacked_by_age(tmp_path):
    """Owner concern: a genuinely long-running process must never lose its
    lock to the stale rule."""
    import os
    import time

    lock1 = FileLock(tmp_path / "lock", holder="long-runner", stale_after_seconds=1)
    lock1.acquire()  # pid = this live test process
    old = time.time() - 7200  # hours old
    os.utime(lock1.path, (old, old))
    with pytest.raises(LockHeldError, match="alive"):
        FileLock(tmp_path / "lock", holder="two", stale_after_seconds=1).acquire()
    lock1.release()


def test_lock_corrupt_falls_back_to_age(tmp_path):
    import os
    import time

    lock_path = tmp_path / "lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("not-json", encoding="utf-8")
    fresh = time.time()
    os.utime(lock_path, (fresh, fresh))
    with pytest.raises(LockHeldError):
        FileLock(lock_path, holder="two", stale_after_seconds=3600).acquire()
    old = time.time() - 7200
    os.utime(lock_path, (old, old))
    FileLock(lock_path, holder="two", stale_after_seconds=3600).acquire()


# --- runner stages (offline; no collect) ------------------------------------------


@pytest.fixture()
def runner_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ICMI_DATA_DIR", str(tmp_path / "data"))
    return PipelineRunner(mode="manual")


def test_runner_ingest_empty_is_ok(runner_env):
    result = runner_env.run(["ingest"])
    assert result.ok
    ingest = next(s for s in result.stages if s.name == "ingest")
    assert "new=0" in ingest.detail


def test_runner_unknown_stage_reports_failure(runner_env):
    result = runner_env.run(["ingest", "bogus_stage"])
    assert not result.ok
    bogus = next(s for s in result.stages if s.name == "bogus_stage")
    assert "unknown stage" in bogus.detail


def test_runner_ingest_picks_up_raw_runs(runner_env):
    from icmi.core.paths import raw_dir
    from tests.fixtures.bama_payloads import bama_ad

    records = raw_dir() / "run-x" / "bama" / "records.jsonl"
    records.parent.mkdir(parents=True)
    ad = bama_ad("abc123")
    records.write_text(
        json.dumps({"source_listing_id": "abc123", "raw_payload": ad}) + "\n",
        encoding="utf-8",
    )
    result = runner_env.run(["ingest"])
    assert result.ok
    ingest = next(s for s in result.stages if s.name == "ingest")
    assert "new=1" in ingest.detail
    # rerun → run-level skip
    result2 = runner_env.run(["ingest"])
    ingest2 = next(s for s in result2.stages if s.name == "ingest")
    assert "runs_skipped=1" in ingest2.detail


def test_runner_lock_blocks_parallel_runs(runner_env, tmp_path):
    runner_env.run(["ingest"])  # completes and releases
    # simulate a held lock
    lock_path = runner_env._lock_path
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("{}", encoding="utf-8")
    result = runner_env.run(["ingest"])
    assert not result.ok
    assert "lock" in result.stages[0].name
    lock_path.unlink()


# --- image proxy hardening ----------------------------------------------------------


def test_proxy_rejects_redirect(tmp_path):
    class RedirectResponse:
        status_code = 302
        content = b""
        headers = {"location": "https://evil.example.com/nowhere"}

    cache = ImageCacheWithFetch(
        tmp_path, lambda url: RedirectResponse()
    )
    with pytest.raises(ImageProxyError, match="upstream fetch failed"):
        cache.get("https://cdn.example.com/a.jpg")


def test_proxy_rejects_non_image_content(tmp_path):
    class HtmlResponse:
        status_code = 200
        content = b"<html>not an image</html>"
        headers = {"content-type": "text/html"}

    cache = ImageCacheWithFetch(tmp_path, lambda url: HtmlResponse())
    with pytest.raises(ImageProxyError, match="upstream fetch failed"):
        cache.get("https://cdn.example.com/a.jpg")


def test_proxy_rejects_oversize(tmp_path):
    class BigResponse:
        status_code = 200
        content = b"x" * (16 * 1024 * 1024)
        headers = {"content-type": "image/webp"}

    cache = ImageCacheWithFetch(tmp_path, lambda url: BigResponse())
    with pytest.raises(ImageProxyError, match="upstream fetch failed"):
        cache.get("https://cdn.example.com/big.jpg")


def ImageCacheWithFetch(tmp_path, fetcher):
    from icmi.api.image_proxy import ImageCache

    return ImageCache(
        allow_hosts=["cdn.example.com"], max_cache_mb=5, ttl_days=1,
        root=tmp_path / "cache", fetcher=fetcher, now=lambda: 1000.0,
        resolver=lambda host: None,  # safe IPs — DNS guard tested separately
    )


def test_proxy_dns_guard_rejects_private_resolution(tmp_path):
    """Final G3 verification: an allowlisted hostname that RESOLVES to a
    private/loopback address is rejected — DNS-rebinding mitigation."""
    cache = ImageCache(
        allow_hosts=["cdn.example.com"], max_cache_mb=5, ttl_days=1,
        root=tmp_path / "cache",
        fetcher=lambda url: FakeResponse(b"IMG"),
        now=lambda: 1000.0,
        resolver=lambda host: "resolves_to_disallowed_address (127.0.0.1)",
    )
    with pytest.raises(ImageProxyError, match="disallowed"):
        cache.get("https://cdn.example.com/a.jpg")


def test_proxy_dns_guard_reports_resolution_failure(tmp_path):
    cache = ImageCache(
        allow_hosts=["cdn.example.com"], max_cache_mb=5, ttl_days=1,
        root=tmp_path / "cache",
        fetcher=lambda url: FakeResponse(b"IMG"),
        now=lambda: 1000.0,
        resolver=lambda host: "dns_resolution_failed",
    )
    with pytest.raises(ImageProxyError, match="dns_resolution_failed"):
        cache.get("https://cdn.example.com/a.jpg")


class FakeResponse:
    def __init__(self, content: bytes, content_type: str = "image/webp") -> None:
        self.status_code = 200
        self.content = content
        self.headers = {"content-type": content_type}
