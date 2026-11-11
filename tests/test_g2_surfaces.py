"""Tests for G2 backend: estimate outlier trimming, /compare, mileage stats."""

import pytest

from icmi.analytics.comparables import estimate_market_range
from icmi.core.records import CanonicalRecord, RawPayloadRef, utc_now
from icmi.pipeline.store import CanonicalStore


def make_listing(store, lid, *, brand, model=None, year=None, mileage=None, price):
    record = CanonicalRecord(
        source_platform="bama",
        source_listing_id=lid,
        original_url=f"https://example.com/{lid}",
        title=f"{brand} {model or ''}".strip(),
        brand_raw=brand,
        model_raw=model,
        year_raw=year,
        mileage_raw=mileage,
        price_raw=price,
        scraped_at=utc_now(),
        extraction_version="test-v1",
        raw_payload_ref=RawPayloadRef(
            run_id="r", platform="bama", file="records.jsonl", key=lid
        ),
    )
    store.upsert_listings([record], "run-test")


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("ICMI_DATA_DIR", str(tmp_path / "data"))
    store = CanonicalStore()
    for i, price in enumerate(
        ["700,000,000", "720,000,000", "740,000,000", "760,000,000",
         "780,000,000", "800,000,000", "3,000,000,000"]
    ):
        make_listing(store, f"a{i}", brand="پژو", model="206", year=1398,
                     mileage=45000, price=price)
    return store


def test_estimate_trim_outliers_flag(store):
    with_trim = estimate_market_range(
        store, brand="پژو", model="206", year=1398, trim_outliers=True
    )
    without = estimate_market_range(
        store, brand="پژو", model="206", year=1398, trim_outliers=False
    )
    assert without.max == 3_000_000_000  # outlier retained untrimmed
    assert with_trim.max < 1_000_000_000  # IQR trim dropped the 3B outlier
    assert with_trim.count < without.count


def test_estimate_trim_never_eats_small_samples(tmp_path, monkeypatch):
    # 6+ samples required; with exactly 5 nothing is trimmed (isolated data dir)
    monkeypatch.setenv("ICMI_DATA_DIR", str(tmp_path / "data2"))
    store2 = CanonicalStore()
    for i, price in enumerate(["700,000,000", "720,000,000", "740,000,000",
                               "760,000,000", "3,000,000,000"]):
        make_listing(store2, f"b{i}", brand="پژو", model="206", year=1398, price=price)
    result = estimate_market_range(
        store2, brand="پژو", model="206", year=1398, trim_outliers=True
    )
    assert result.max == 3_000_000_000  # 5 samples → no trimming


def test_compare_endpoint_assembles_ranges(tmp_path, monkeypatch):
    monkeypatch.setenv("ICMI_DATA_DIR", str(tmp_path / "data"))
    from fastapi.testclient import TestClient

    from icmi.api.app import create_app

    store = CanonicalStore()
    make_listing(store, "c1", brand="پژو", model="206", year=1398, price="750,000,000")
    make_listing(store, "c2", brand="پژو", model="206", year=1399, price="850,000,000")
    app = create_app(search_service=None, store=store)
    client = TestClient(app)

    body = client.get("/compare", params={"keys": "bama:c1,bama:c2"}).json()
    assert len(body["vehicles"]) == 2
    for v in body["vehicles"]:
        assert v["listing"]["brand_raw"] == "پژو"
        assert v["market_range"]["count"] >= 1
        assert v["market_range"]["median"] is not None
    assert client.get("/compare", params={"keys": "bama:c1"}).status_code == 400
    assert client.get("/compare", params={"keys": "bama:c1,nope"}).status_code == 404


def test_stats_includes_mileage_distribution(tmp_path, monkeypatch):
    monkeypatch.setenv("ICMI_DATA_DIR", str(tmp_path / "data"))
    from icmi.analytics.stats import market_stats

    store = CanonicalStore()
    make_listing(store, "m1", brand="پژو", year=1398, mileage=20000, price="750,000,000")
    make_listing(store, "m2", brand="پژو", year=1398, mileage=45000, price="700,000,000")
    make_listing(store, "m3", brand="پژو", year=1398, mileage=120000, price="600,000,000")
    result = market_stats(store)
    buckets = set(result["by_mileage"]["mileage_bucket"])
    assert "0-30k" in buckets and "30-60k" in buckets and "100-150k" in buckets


def test_browse_mode_empty_query_with_filters(tmp_path, monkeypatch):
    """Filter-only browsing: empty query + filters returns filtered listings
    ordered by the sort field — no semantic ranking, filters still hard."""
    monkeypatch.setenv("ICMI_DATA_DIR", str(tmp_path / "data3"))
    import numpy as np

    from icmi.retrieval.search import SearchFilters, SearchService
    from tests.fakes import HashingEncoder

    store = CanonicalStore()
    make_listing(store, "x1", brand="پژو", model="206", year=1398, price="700,000,000")
    make_listing(store, "x2", brand="پژو", model="206", year=1399, price="900,000,000")
    make_listing(store, "x3", brand="تویوتا", model="کرولا", year=2015, price="2,000,000,000")
    from icmi.retrieval.index import build_index_metadata
    chunks = build_index_metadata(store)
    encoder = HashingEncoder()
    vectors = np.asarray(encoder.encode_passages([c.normalized_text for c in chunks]))
    import pandas as pd
    metadata = pd.DataFrame([vars(c) for c in chunks])
    service = SearchService(vectors, metadata, encoder)

    response = service.search("", SearchFilters(brands=["پژو"]), top_k=10)
    assert response.candidates_after_filters == 2
    keys = [r.listing_key for r in response.results]
    # browse order (same-second timestamps → either order valid)
    assert sorted(keys) == ["bama:x1", "bama:x2"]
    # deterministic ordering via explicit sort:
    priced = service.search("", SearchFilters(brands=["پژو"]), top_k=10, sort="price_desc")
    assert [r.listing_key for r in priced.results] == ["bama:x2", "bama:x1"]


def test_incremental_invalidation_matrix(tmp_path, monkeypatch):
    """H4 audit: incremental build must re-encode changed content, carry over
    unchanged, drop deleted, and NEVER reuse vectors across models."""

    from icmi.retrieval.incremental import incremental_index_build
    from tests.fakes import HashingEncoder

    monkeypatch.setenv("ICMI_DATA_DIR", str(tmp_path / "data"))
    store = CanonicalStore()
    make_listing(store, "i1", brand="پژو", model="206", year=1398, price="750,000,000")
    make_listing(store, "i2", brand="پژو", model="پارس", year=1400, price="1,900,000,000")

    enc = HashingEncoder()
    result1 = incremental_index_build(
        store, model_id="fake-hashing", out_dir=tmp_path / "idx",
        full_rebuild=True, encoder=enc,
    )
    assert result1["total_chunks"] == 2 and result1["newly_encoded"] == 2

    # 1) no changes → all carried
    result2 = incremental_index_build(
        store, model_id="fake-hashing", out_dir=tmp_path / "idx", encoder=enc
    )
    assert result2["carried_over"] == 2 and result2["newly_encoded"] == 0

    # 2) changed description (content change) → re-encoded
    conn = store._conn
    conn.execute("UPDATE listings SET description = 'تغییر کرده' WHERE listing_key = 'bama:i1'")
    conn.commit()
    result3 = incremental_index_build(
        store, model_id="fake-hashing", out_dir=tmp_path / "idx", encoder=enc
    )
    assert result3["newly_encoded"] >= 1

    # 3) deleted listing → chunk dropped
    conn.execute("DELETE FROM listings WHERE listing_key = 'bama:i2'")
    conn.commit()
    result4 = incremental_index_build(
        store, model_id="fake-hashing", out_dir=tmp_path / "idx", encoder=enc
    )
    assert result4["total_chunks"] == 1

    # 4) model change → full re-encode (never reuse incompatible vectors)
    result5 = incremental_index_build(
        store, model_id="other-model", out_dir=tmp_path / "idx", encoder=enc
    )
    assert result5["newly_encoded"] == result5["total_chunks"]


def test_incremental_requires_model_match_in_saved_index(tmp_path, monkeypatch):
    """A saved index built with model A must not silently feed model B carries."""
    import numpy as np

    from icmi.retrieval.index import build_index_metadata, load_index, save_index
    from tests.fakes import HashingEncoder

    monkeypatch.setenv("ICMI_DATA_DIR", str(tmp_path / "data"))
    store = CanonicalStore()
    make_listing(store, "s1", brand="پژو", model="206", year=1398, price="750,000,000")
    chunks = build_index_metadata(store)
    enc = HashingEncoder()
    vectors = np.asarray(enc.encode_passages([c.normalized_text for c in chunks]))
    save_index(vectors, chunks, "model-A", out_dir=tmp_path / "idx")

    _, meta, info = load_index(tmp_path / "idx")
    assert info["model_id"] == "model-A"
    # a build with model B must not carry model-A vectors
    from icmi.retrieval.incremental import incremental_index_build

    incremental_index_build(
        store, model_id="model-B", out_dir=tmp_path / "idx", encoder=enc
    )
    _, _, info2 = load_index(tmp_path / "idx")
    assert info2["model_id"] == "model-B"

