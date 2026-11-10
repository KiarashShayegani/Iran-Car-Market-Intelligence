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
