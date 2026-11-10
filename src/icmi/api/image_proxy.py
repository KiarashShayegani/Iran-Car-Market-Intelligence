"""Cache-aside advertisement image proxy (G1 spike decision).

Fallback path only: the frontend renders CDN images directly and calls this
proxy solely when a direct fetch fails. Guarantees:

- Host allowlist (from config/sources.yaml `image_cache.allow_hosts`) —
  never an open proxy.
- Bounded cache under `data/image-cache/`: LRU eviction by file access time
  when total size exceeds `max_cache_mb`; TTL sweep for expired entries.
- Stale-on-error: if the upstream fetch fails but a (possibly expired) cached
  copy exists, the cached copy is served rather than an error.
- Atomic writes (temp file + rename) so readers never see partial files.
- Cached images are derived, rebuildable data — never committed, never
  republished outside this application.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests

from icmi.core.paths import image_cache_dir
from icmi.core.records import utc_now

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)
FETCH_TIMEOUT_SECONDS = 20


class ImageProxyError(Exception):
    """Raised when an image cannot be served.

    `kind` distinguishes client errors (disallowed host → 400) from upstream
    failures (fetch failed → 502).
    """

    def __init__(self, message: str, kind: str = "upstream") -> None:
        self.kind = kind
        super().__init__(message)


@dataclass
class CachedImage:
    path: Path
    content_type: str


class ImageCache:
    def __init__(
        self,
        allow_hosts: list[str],
        max_cache_mb: int = 300,
        ttl_days: int = 14,
        root: Path | None = None,
        fetcher=None,
        now=None,
    ) -> None:
        self.allow_hosts = {h.lower() for h in allow_hosts}
        self.max_bytes = max_cache_mb * 1024 * 1024
        self.ttl_seconds = ttl_days * 86400
        self.root = root or image_cache_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        self._fetcher = fetcher or self._default_fetch
        self._now = now or time.time
        self._last_sweep = 0.0

    # --- public API ---------------------------------------------------------

    def get(self, url: str) -> CachedImage:
        """Serve from cache when fresh (or as stale fallback); fetch otherwise."""
        host = self._allowed_host(url)
        if host is None:
            raise ImageProxyError(
                f"host not allowed: {urlparse(url).hostname}", kind="disallowed"
            )

        path, meta = self._cache_paths(url)
        fresh = (
            path.is_file()
            and (self._now() - meta["fetched_at"]) < self.ttl_seconds
        )

        if not fresh:
            fetched = self._fetch_to_cache(url, path, meta)
            if not fetched and path.is_file():
                # stale-on-error: serve the expired copy rather than failing
                return CachedImage(path=path, content_type=meta["content_type"])
            if not fetched:
                raise ImageProxyError(f"upstream fetch failed: {url}")

        self._maybe_sweep()
        return CachedImage(path=path, content_type=meta["content_type"])

    # --- internals ------------------------------------------------------------

    def _allowed_host(self, url: str) -> str | None:
        try:
            host = (urlparse(url).hostname or "").lower()
        except ValueError:
            return None
        return host if host in self.allow_hosts else None

    def _cache_paths(self, url: str) -> tuple[Path, dict]:
        key = hashlib.sha256(url.encode()).hexdigest()
        base = self.root / key[:2]
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"{key}.bin"
        meta_path = base / f"{key}.meta.json"
        if meta_path.is_file():
            import json

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            meta = {"url": url, "content_type": "application/octet-stream", "fetched_at": 0.0}
        return path, meta

    def _fetch_to_cache(self, url: str, path: Path, meta: dict) -> bool:
        try:
            response = self._fetcher(url)
        except Exception:
            return False
        if getattr(response, "status_code", 0) != 200:
            return False
        content = getattr(response, "content", b"")
        if not content:
            return False
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(content)
        tmp.replace(path)  # atomic
        meta["fetched_at"] = self._now()
        meta["content_type"] = response.headers.get("content-type", "image/jpeg").split(";")[0]
        meta["size"] = len(content)
        import json

        path.with_suffix(".meta.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )
        self._evict_if_needed()
        return True

    @staticmethod
    def _default_fetch(url: str):
        return requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=FETCH_TIMEOUT_SECONDS
        )

    def _evict_if_needed(self) -> None:
        entries = [
            (p.stat().st_atime_ns, p, p.with_suffix(".meta.json"), p.stat().st_size)
            for p in self.root.rglob("*.bin")
            if p.is_file()
        ]
        total = sum(size for *_, size in entries)
        if total <= self.max_bytes:
            return
        entries.sort()  # oldest access first
        for _, bin_path, meta_path, size in entries:
            if total <= self.max_bytes:
                break
            bin_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            total -= size

    def _maybe_sweep(self) -> None:
        """TTL sweep, throttled to once per hour per process."""
        if self._now() - self._last_sweep < 3600:
            return
        self._last_sweep = self._now()
        cutoff = self._now() - self.ttl_seconds
        for meta_path in self.root.rglob("*.meta.json"):
            import json

            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("fetched_at", 0) < cutoff:
                    meta_path.with_suffix("").unlink(missing_ok=True)
                    meta_path.unlink(missing_ok=True)
            except (ValueError, OSError):
                continue

    def stats(self) -> dict:
        files = list(self.root.rglob("*.bin"))
        return {
            "cached_images": len(files),
            "total_mb": round(sum(f.stat().st_size for f in files) / (1024 * 1024), 2),
            "max_mb": self.max_bytes // (1024 * 1024),
            "as_of": utc_now().isoformat(timespec="seconds"),
        }
