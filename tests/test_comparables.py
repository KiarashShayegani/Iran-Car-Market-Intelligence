"""Tests for the comparables engine (retrieval-based price estimation)."""

import pytest

from icmi.analytics.comparables import estimate_market_range
from icmi.core.records import CanonicalRecord, RawPayloadRef, utc_now
from icmi.pipeline.store import CanonicalStore


def make_listing(store, lid, *, brand, model=None, year=None, mileage=None, price):
    record = CanonicalRecord(
        source_platform="bama",
        source_listing_id=lid,
        original_url=f"https://example.com/{lid}",
        title=f"{brand} {model or ''} {year or ''}".strip(),
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
    # five 206s around 1398 (shamsi) with varied prices
    make_listing(store, "a1", brand="پژو", model="206", year=1398,
                 mileage=45000, price="750,000,000")
    make_listing(store, "a2", brand="پژو", model="206", year=1399,
                 mileage=40000, price="800,000,000")
    make_listing(store, "a3", brand="پژو", model="206", year=1397,
                 mileage=60000, price="700,000,000")
    make_listing(store, "a4", brand="پژو", model="206", year=1398,
                 mileage=50000, price="780,000,000")
    make_listing(store, "a5", brand="پژو", model="206", year=1399,
                 mileage=35000, price="820,000,000")
    # a gregorian-year Peugeot that must NEVER mix into shamsi comparables
    make_listing(store, "g1", brand="پژو", model="206", year=1998,
                 mileage=45000, price="1,500,000,000")
    # an expensive different-brand car
    make_listing(store, "b1", brand="تویوتا", model="کرولا", year=1399,
                 mileage=30000, price="3,000,000,000")
    # unpriced Peugeot — excluded from estimates
    make_listing(store, "n1", brand="پژو", model="206", year=1398,
                 mileage=45000, price=None)
    return store


def test_estimate_returns_observed_range_and_excludes_other_calendar(store):
    result = estimate_market_range(store, brand="پژو", model="206", year=1398, mileage_km=45000)
    assert result.quality == "ok"
    assert result.count == 5  # gregorian 1998 + unpriced n1 never join
    assert result.median == 780_000_000
    assert result.p25 <= result.median <= result.p75
    assert result.min == 700_000_000 and result.max == 820_000_000
    assert all(c.year_calendar == "shamsi" for c in result.comparables)
    assert all(c.listing_key != "bama:g1" for c in result.comparables)
    assert all(c.listing_key != "bama:n1" for c in result.comparables)


def test_estimate_orders_comparables_by_year_proximity(store):
    result = estimate_market_range(store, brand="پژو", model="206", year=1398)
    years = [c.year_num for c in result.comparables if c.year_num is not None]
    assert years[0] == 1398  # nearest year first


def test_estimate_low_sample_flagged_honestly(store):
    result = estimate_market_range(store, brand="تویوتا", model="کرولا", year=1399)
    assert result.quality == "low_sample"
    assert result.count == 1
    assert "نمونه کوچک" in result.quality_note


def test_estimate_no_data_is_honest(store):
    result = estimate_market_range(store, brand="فراری", model="sf90")
    assert result.quality == "no_data"
    assert result.median is None and result.comparables == []


def test_estimate_unclassifiable_year_refuses(store):
    result = estimate_market_range(store, brand="پژو", model="206", year=1700)
    assert result.quality == "no_data"
    assert "قابل‌طبقه‌بندی نیست" in result.quality_note


def test_estimate_requires_brand(store):
    with pytest.raises(ValueError, match="brand is required"):
        estimate_market_range(store, brand="   ")
