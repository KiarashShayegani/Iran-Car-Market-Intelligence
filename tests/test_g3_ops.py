"""Tests for G3: file locking, unified pipeline runner stages, image proxy
hardening, and restore verification (all offline)."""

import json

import pytest

from icmi.api.image_proxy import ImageProxyError
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


def test_lock_stale_takeover(tmp_path):
    import os
    import time

    lock1 = FileLock(tmp_path / "lock", holder="dead", stale_after_seconds=1)
    lock1.acquire()
    # simulate age
    old = time.time() - 7200
    os.utime(lock1.path, (old, old))
    lock2 = FileLock(tmp_path / "lock", holder="two", stale_after_seconds=3600)
    lock2.acquire()  # stale → takeover allowed
    lock2.release()


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
    )
