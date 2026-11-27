"""Tests for the Divar collector extraction (no network).

Covers the 2026-11-28 parser corrections: original_url constructed from the
token (was 0% — broke traceability and the «منبع» link), mileage parsed from
the list row's top chip («۳۰۰,۰۰۰ کیلومتر» / «صفر کیلومتر») only when the
text actually is a mileage, and source_published_at honestly left empty
(list rows carry no posting date — the top chip is NOT a date).
"""

from icmi.collectors.divar import DivarCollector
from icmi.core.config import PipelineConfig
from icmi.core.records import RawPayloadRef


def _bare_collector() -> DivarCollector:
    """DivarCollector without __init__ (no HTTP session / writer needed)."""
    collector = DivarCollector.__new__(DivarCollector)
    collector.source_key = "divar"
    collector.pipeline = PipelineConfig(
        extraction_version="test-v1",
        request_timeout_seconds=5,
        request_delay_seconds=0,
        request_jitter_seconds=0,
        max_retries=1,
        user_agent="test-agent",
    )
    return collector


def _ad(**overrides) -> dict:
    """A realistic POST_ROW data node (20261127 run sample shape)."""
    ad = {
        "token": "waCkq-6g",
        "title": "میتسوبیشی اکلیپس کراس",
        "middle_description_text": "۷,۹۹۰,۰۰۰,۰۰۰ تومان",
        "top_description_text": "۰ کیلومتر",
        "bottom_description_text": "نمایشگاه در الهیه",
        "image_url": "https://s100.divarcdn.com/static/photo/x.webp",
        "image_count": 2,
        "has_chat": True,
        "layout_type": "GRID_ITEM",
        "red_text": "پله شده",
        "image_top_left_tag": {"text": "پله"},
    }
    ad.update(overrides)
    return ad


def _parse(ad: dict):
    collector = _bare_collector()
    return collector.parse_listing(
        ad, RawPayloadRef(run_id="r", platform="divar", file="records.jsonl", key="x")
    )


def test_original_url_constructed_from_token():
    record = _parse(_ad())
    assert record.original_url == "https://divar.ir/v/waCkq-6g"


def test_price_raw_captured_and_amount_derived_downstream():
    """CanonicalRecord carries price_raw only — price_amount is derived by
    the store's ingest (the old collector even had a dead price_amount
    set_field that failed on every row)."""
    record = _parse(_ad())
    assert record.price_raw == "۷,۹۹۰,۰۰۰,۰۰۰ تومان"
    from icmi.pipeline.normalize import normalize_price_amount

    assert normalize_price_amount(record.price_raw) == 7_990_000_000


def test_mileage_parsed_from_top_chip():
    zero = _parse(_ad(top_description_text="۰ کیلومتر"))
    assert zero.mileage_raw == "۰ کیلومتر"

    used = _parse(_ad(top_description_text="۳۰۰,۰۰۰ کیلومتر"))
    assert used.mileage_raw == "۳۰۰,۰۰۰ کیلومتر"


def test_mileage_not_fabricated_from_condition_text():
    """Non-mileage chip texts (other layouts) must stay honestly empty."""
    record = _parse(_ad(top_description_text="در حد نو"))
    assert record.mileage_raw is None


def test_source_published_at_stays_empty():
    """The top chip is mileage text, NOT a posting date — the date field
    must stay None (the text survives in source_attributes)."""
    record = _parse(_ad())
    assert record.source_published_at is None
    assert record.source_attributes["top_description_text"] == "۰ کیلومتر"


def test_extraction_errors_empty_on_clean_row():
    record = _parse(_ad())
    assert record.extraction_errors == {}
    assert record.source_listing_id == "waCkq-6g"
    assert record.location_raw == "نمایشگاه در الهیه"
    assert record.image_urls == ["https://s100.divarcdn.com/static/photo/x.webp"]
