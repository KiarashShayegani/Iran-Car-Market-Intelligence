"""Tests for the Sheypoor and Hamrah Mechanic collectors (no network)."""

from icmi.collectors.hamrah_mechanic import (
    HamrahMechanicCollector,
    iter_hamrah_inventory,
)
from icmi.collectors.sheypoor import SheypoorCollector, iter_sheypoor_ads
from icmi.core.config import PipelineConfig, SourceConfig
from icmi.core.records import RawPayloadRef
from tests.fixtures.other_platforms import hamrah_ad, hamrah_page, sheypoor_ad, sheypoor_page


def make_collector(cls, source_key: str):
    collector = cls.__new__(cls)
    collector.source_key = source_key
    collector.pipeline = PipelineConfig(
        extraction_version="test-v1",
        request_timeout_seconds=5,
        request_delay_seconds=0,
        request_jitter_seconds=0,
        max_retries=1,
        user_agent="test-agent",
    )
    collector.skip_reasons = {}
    collector.collection = "full"
    collector.collection_params = {}
    if cls is SheypoorCollector:
        collector._cursor = None  # normally set in __init__
    return collector


def ref_for(key: str) -> RawPayloadRef:
    return RawPayloadRef(run_id="r", platform="p", file="records.jsonl", key=key)


# --- Sheypoor ---------------------------------------------------------------


def test_iter_sheypoor_ads_filters_by_category_and_type():
    car = sheypoor_ad(1)
    native = {"type": "nativeAd", "id": 2, "attributes": {}, "fullAttributes": []}
    non_car = sheypoor_ad(3, categories=[{"name": "موبایل", "id": 9}])
    no_attrs = {"type": "normal", "id": 4}
    ads = iter_sheypoor_ads(sheypoor_page(car, native, non_car, no_attrs))
    assert [a["id"] for a in ads] == [1]


def test_sheypoor_parse_listing_maps_fields():
    collector = make_collector(SheypoorCollector, "sheypoor")
    record = collector.parse_listing(sheypoor_ad(111), ref_for("111"))
    assert record.source_listing_id == "111"
    assert record.original_url == "https://www.sheypoor.com/v/پژو-206-تیپ-5-111.html"
    assert record.title == "پژو 206 تیپ 5 مدل 97"
    assert record.manufacturer_raw == "پژو"
    assert record.model_raw == "پژو 206"
    assert record.trim_raw == "تیپ 5"
    assert record.year_raw == "1397"
    assert record.mileage_raw == "85,000"
    assert record.price_raw == "2,050,000,000"  # comma-formatted string preserved
    assert record.price_currency_raw == "تومان"
    assert record.location_raw == "تهران"
    assert record.fuel_raw == "بنزینی"
    assert record.transmission_raw == "دنده"
    assert record.body_type_raw == "هاچبک"
    assert record.body_condition_raw == "سالم و بی‌خط‌وخش"
    assert record.color_raw == "نقره‌ای"
    assert record.seller_type == "personal"
    assert record.seller_id == "99887"
    assert record.image_urls and "s.sheypoor.com" in record.image_urls[0]
    assert record.image_count == 6
    assert record.source_published_at == "۲ ساعت پیش"
    assert record.extraction_errors == {}


def test_sheypoor_cursor_chaining_and_has_more():
    collector = make_collector(SheypoorCollector, "sheypoor")
    page = sheypoor_page(sheypoor_ad(1))
    assert collector.page_params(0) == {"p": 0}  # fresh chain: no f
    collector.observe_page(page)
    chained = collector.page_params(1)
    assert chained["p"] == 1
    assert chained["f"] == "[1.0, 1794265261245]_469078347n_1794253002.951553"
    assert collector.has_more(page) is True  # 1 * 24 < 44621

    exhausted = sheypoor_page(sheypoor_ad(2), meta_f="next")
    exhausted["data"]["meta"]["total"] = 24
    assert collector.has_more(exhausted) is False


# --- Hamrah Mechanic ----------------------------------------------------------


def test_iter_hamrah_inventory_filters():
    ad = hamrah_ad(1)
    junk = {"foo": "bar"}
    ads = iter_hamrah_inventory(hamrah_page(ad, junk))
    assert [a["orderId"] for a in ads] == [1]


def test_hamrah_parse_listing_maps_fields():
    collector = make_collector(HamrahMechanicCollector, "hamrah_mechanic")
    record = collector.parse_listing(hamrah_ad(3595004), ref_for("3595004"))
    assert record.source_listing_id == "3595004"
    assert record.title == "هیوندای آزرا ۶ سیلندر"
    assert record.original_url == "https://www.hamrah-mechanic.com/cars-for-sale/3595004"
    assert record.manufacturer_raw == "Hyundai"
    assert record.model_raw == "Azera"  # never the full title (deviation from notebook)
    assert record.trim_raw == "سدان"
    assert record.year_raw == 2012  # Gregorian preserved raw (import)
    assert record.mileage_raw == 250000
    assert record.price_raw == 7100000000
    assert record.location_raw == "شیروان"
    assert record.transmission_raw == "اتوماتیک"
    assert record.color_raw == "مشکی"
    assert record.image_urls == ["https://www.hamrah-mechanic.com/img/3595004.jpg"]
    assert record.description is None  # not supported by this platform
    assert record.fuel_raw is None
    assert record.extraction_errors == {}


def test_hamrah_page_params_are_one_based():
    collector = make_collector(HamrahMechanicCollector, "hamrah_mechanic")
    collector.source = SourceConfig(
        display_name="Hamrah Mechanic",
        display_name_fa="همراه مکانیک",
        enabled=True,
        status="experimental",
        base_url="https://www.hamrah-mechanic.com/api/v1/common/newexhibition",
        pagination={"type": "page_number", "page_param": "page", "page_size": 24},
    )
    assert collector.page_params(0) == {"page": 1, "size": 24}
    assert collector.page_params(2) == {"page": 3, "size": 24}


def test_hamrah_sold_listings_skipped_and_counted():
    collector = make_collector(HamrahMechanicCollector, "hamrah_mechanic")
    assert collector.parse_listing(hamrah_ad(1, is_sold=True), ref_for("1")) is None
    assert collector.skip_reasons.get("sold_listing") == 1
