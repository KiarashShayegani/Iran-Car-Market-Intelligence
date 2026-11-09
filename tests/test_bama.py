"""Tests for the Bama collector extraction (no network)."""

from icmi.collectors.bama import (
    BamaCollector,
    bama_public_url,
    iter_bama_ads,
    listing_id_of,
)
from icmi.core.config import PipelineConfig
from icmi.core.records import RawPayloadRef
from tests.fixtures.bama_payloads import bama_ad, bama_page


def _bare_collector() -> BamaCollector:
    """BamaCollector without __init__ (no HTTP session / writer needed)."""
    collector = BamaCollector.__new__(BamaCollector)
    collector.source_key = "bama"
    collector.pipeline = PipelineConfig(
        extraction_version="test-v1",
        request_timeout_seconds=5,
        request_delay_seconds=0,
        request_jitter_seconds=0,
        max_retries=1,
        user_agent="test-agent",
    )
    return collector


def test_iter_ads_filters_non_ads_and_duplicates():
    ad1 = bama_ad("aaa")
    ad2 = bama_ad("bbb")
    page = bama_page(
        ad1,
        ad1,  # exact duplicate node
        ad2,
        extra_nodes=[
            {"type": "banner", "detail": {"code": "zzz"}},  # not an ad
            {"type": "ad", "detail": {"type": "moto", "code": "mmm"}},  # not a car
            {"type": "ad"},  # no detail
        ],
    )
    ads = iter_bama_ads(page)
    assert [listing_id_of(a) for a in ads] == ["aaa", "bbb"]


def test_public_url_prefers_canonical_then_builds_relative():
    ad = bama_ad("aaa")
    assert bama_public_url(ad) == "https://bama.ir/car/پژو-206/aaa"

    no_canonical = bama_ad("bbb", **{})
    no_canonical["metadata"] = {}
    assert bama_public_url(no_canonical) == "https://bama.ir/car/پژو-206/bbb"

    ad_no_urls = bama_ad("ccc")
    ad_no_urls["metadata"] = {}
    ad_no_urls["detail"]["url"] = None
    assert bama_public_url(ad_no_urls) is None


def _parse(collector: BamaCollector, ad: dict):
    ref = RawPayloadRef(
        run_id="r", platform="bama", file="records.jsonl", key="k"
    )
    return collector.parse_listing(ad, ref)


def test_parse_listing_maps_canonical_fields():
    collector = _bare_collector()
    record = _parse(collector, bama_ad("aaa"))

    assert record.source_platform == "bama"
    assert record.source_listing_id == "aaa"
    assert record.original_url == "https://bama.ir/car/پژو-206/aaa"
    assert record.title == "پژو 206، مدل 1398، تیپ 2"
    assert record.description and "بیمه کامل" in record.description
    assert record.brand_raw == "پژو"
    assert record.model_raw == "پژو 206"
    assert record.trim_raw == "تیپ 2"
    assert record.year_raw == 1398
    assert record.mileage_raw == "45,000 km"
    assert record.price_raw == 750_000_000
    assert record.price_currency_raw == "تومان"
    assert record.price_details["type"] == "price"
    assert record.location_raw == "تهران"
    assert record.fuel_raw == "بنزینی"
    assert record.transmission_raw == "دنده ای"
    assert record.body_type_raw == "هاچبک"
    assert record.body_condition_raw == "سالم و بی خط و خش"
    assert record.color_raw == "سفید"
    assert record.seller_type == "personal"
    assert record.seller_id == "12345"
    assert record.image_urls and record.image_urls[0].startswith("https://cdn.bama.ir")
    assert record.image_count == 8
    assert record.source_published_at == "لحظات پیش"
    assert record.source_attributes["specs"] == {"gearbox": "5-speed"}
    assert record.raw_payload_ref.key == "k"
    assert record.extraction_errors == {}


def test_parse_listing_keeps_negotiable_price_raw_and_flags_missing_fields():
    collector = _bare_collector()
    ad = bama_ad(
        "neg",
        price={"price": 0, "type": "negotiable"},
        description=None,
        mileage=None,
        fuel=None,
    )
    record = _parse(collector, ad)
    # Raw value preserved verbatim — never zero-filled or dropped.
    assert record.price_raw == 0
    assert record.price_details["type"] == "negotiable"
    assert record.description is None  # missing_in_ad, not fabricated
    assert record.mileage_raw is None
    assert record.fuel_raw is None
    assert record.extraction_errors == {}


def test_parse_listing_records_extraction_failures(monkeypatch):
    collector = _bare_collector()

    def boom(ad):
        raise RuntimeError("simulated URL parser crash")

    monkeypatch.setattr("icmi.collectors.bama.bama_public_url", boom)
    record = _parse(collector, bama_ad("bad"))
    assert "original_url" in record.extraction_errors
    assert "RuntimeError" in record.extraction_errors["original_url"]
    assert record.original_url is None
    assert record.title == "پژو 206، مدل 1398، تیپ 2"  # other fields unaffected
