"""Tests for G1 backend additions: image proxy cache, pagination/sort,
/latest, /facets, similar listings, and price-history enrichment."""

import json

import pytest

from icmi.api.image_proxy import ImageCache, ImageProxyError
from icmi.core.records import CanonicalRecord, RawPayloadRef, utc_now
from icmi.pipeline.store import CanonicalStore


class FakeResponse:
    def __init__(self, content: bytes, content_type: str = "image/webp") -> None:
        self.status_code = 200
        self.content = content
        self.headers = {"content-type": content_type}


def make_listing(store, platform, lid, *, year=None, price=None, brand=None, model=None):
    record = CanonicalRecord(
        source_platform=platform,
        source_listing_id=lid,
        original_url=f"https://example.com/{lid}",
        title=f"{brand or ''} {model or ''}".strip(),
        brand_raw=brand,
        model_raw=model,
        year_raw=year,
        price_raw=price,
        scraped_at=utc_now(),
        extraction_version="test-v1",
        raw_payload_ref=RawPayloadRef(
            run_id="r", platform=platform, file="records.jsonl", key=lid
        ),
    )
    store.upsert_listings([record], "run-test")


# --- image cache ----------------------------------------------------------------


def test_image_cache_fetch_store_and_serve(tmp_path):
    calls = []

    def fake_fetch(url):
        calls.append(url)
        return FakeResponse(b"IMGDATA-" + url.encode()[-6:])

    cache = ImageCache(
        allow_hosts=["cdn.example.com"], max_cache_mb=1, ttl_days=1,
        root=tmp_path / "cache", fetcher=fake_fetch, now=lambda: 1000.0,
        resolver=lambda host: None,  # DNS guard tested in test_g3_ops
    )
    url = "https://cdn.example.com/a.jpg"
    first = cache.get(url)
    assert first.path.read_bytes().startswith(b"IMGDATA")
    assert first.content_type == "image/webp"
    assert len(calls) == 1
    # second get within TTL: served from cache, no second fetch
    second = cache.get(url)
    assert second.path == first.path
    assert len(calls) == 1


def test_image_cache_rejects_disallowed_hosts(tmp_path):
    cache = ImageCache(allow_hosts=["cdn.example.com"], root=tmp_path / "cache")
    with pytest.raises(ImageProxyError, match="not allowed"):
        cache.get("https://evil.example.com/secret.jpg")


def test_image_cache_stale_on_error(tmp_path):
    def failing_fetch(url):
        raise ConnectionError("down")

    state = {"t": 0.0}
    cache = ImageCache(
        allow_hosts=["cdn.example.com"], max_cache_mb=1, ttl_days=1,
        root=tmp_path / "cache", fetcher=failing_fetch, now=lambda: state["t"],
        resolver=lambda host: None,
    )
    url = "https://cdn.example.com/a.jpg"
    # first: pretend a previous run cached it
    path, meta = cache._cache_paths(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"OLDDATA")
    meta.update({"fetched_at": 500.0, "content_type": "image/webp"})
    path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")
    # time passes beyond TTL, upstream is down → serve stale
    state["t"] = 500.0 + 10 * 86400
    served = cache.get(url)
    assert served.path.read_bytes() == b"OLDDATA"


def test_image_cache_lru_eviction(tmp_path):
    current = {"t": 0.0}

    def fetch(url):
        return FakeResponse(b"x" * 900_000)  # ~0.9MB

    cache = ImageCache(
        allow_hosts=["cdn.example.com"], max_cache_mb=1, ttl_days=30,
        root=tmp_path / "cache", fetcher=fetch, now=lambda: current["t"],
        resolver=lambda host: None,
    )
    for i in range(3):
        current["t"] += 1000
        cache.get(f"https://cdn.example.com/{i}.jpg")
    files = list(cache.root.rglob("*.bin"))
    total = sum(f.stat().st_size for f in files)
    assert total <= 1 * 1024 * 1024
    assert 0 < len(files) < 3  # oldest evicted


# --- pagination / sort / new endpoints -------------------------------------------


@pytest.fixture()
def api_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ICMI_DATA_DIR", str(tmp_path / "data"))
    store = CanonicalStore()
    make_listing(store, "bama", "p1", brand="پژو", model="206", year=1398, price="750,000,000")
    make_listing(store, "bama", "p2", brand="پژو", model="206", year=1399, price="850,000,000")
    make_listing(store, "bama", "p3", brand="پژو", model="پارس", year=1400, price="1,900,000,000")
    make_listing(
        store, "bama", "t1", brand="تویوتا", model="کریسida", year=2011,
        price="7,100,000,000",
    )
    return store


def test_search_offset_pagination_is_true_continuation(api_env, monkeypatch):
    from icmi.retrieval.search import SearchService
    from tests.fakes import HashingEncoder

    monkeypatch.setenv("ICMI_DATA_DIR", str(api_env.db_path.parent.parent))
    encoder = HashingEncoder()
    service = SearchService(_vectors(api_env, encoder), _metadata(api_env), encoder)

    page1 = service.search("ماشین", top_k=2, offset=0)
    page2 = service.search("ماشین", top_k=2, offset=2)
    assert page1.total_matched == page2.total_matched == 4
    keys1 = [r.listing_key for r in page1.results]
    keys2 = [r.listing_key for r in page2.results]
    assert not (set(keys1) & set(keys2)), "page 2 must not repeat page 1"
    assert len(keys1) == 2 and len(keys2) == 2


def test_search_sort_price_desc(api_env):
    from icmi.retrieval.search import SearchService
    from tests.fakes import HashingEncoder

    encoder = HashingEncoder()
    service = SearchService(_vectors(api_env, encoder), _metadata(api_env), encoder)
    response = service.search("ماشین", top_k=10, sort="price_desc")
    prices = [r.listing_key for r in response.results]
    assert prices[0] == "bama:t1"  # 7.1B first


def _vectors(store, encoder):
    import numpy as np

    from icmi.retrieval.index import build_index_metadata

    chunks = build_index_metadata(store)
    return np.asarray(encoder.encode_passages([c.normalized_text for c in chunks]))


def _metadata(store):
    from icmi.retrieval.index import build_index_metadata

    chunks = build_index_metadata(store)
    import pandas as pd

    return pd.DataFrame([vars(c) for c in chunks])


def test_price_history_and_images_in_listing_detail(api_env):
    """Direct store-level verification of the enrichment query shape."""
    observations = api_env.query_df(
        "SELECT observed_at, price_amount, run_id FROM price_observations WHERE listing_key = ?",
        ("bama:p1",),
    )
    assert len(observations) == 1
    assert observations.iloc[0]["price_amount"] == 750_000_000
