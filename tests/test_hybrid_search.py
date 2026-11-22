"""Tests for the hybrid retrieval layer: entity detection + steering,
keyword/semantic fusion, canonical facet values, and browse diversity.

The regression at the heart of these tests (owner report 2026-11-22): the
pure-semantic pipeline ranked «ام جی» (MG) ads first for "AMG" and شاهین
«تیپ G» for "G class" — short entity queries need deterministic steering.
"""

import re

import numpy as np
import pytest

from icmi.core.taxonomy import clean_trim_label, extract_model_from_title
from icmi.core.value_aliases import (
    canonical_facet_values,
    canonical_value,
    variants_of,
)
from icmi.retrieval.entities import detect_entities, model_match_pattern
from icmi.retrieval.search import SearchFilters, SearchService
from tests.fakes import HashingEncoder

# --- entity detection -------------------------------------------------------

VOCAB = {
    "206": {"label": "206", "brands": {"پژو"}, "listing_keys": {"bama:a"}},
    "کلاسg": {"label": "کلاس G", "brands": {"بنز"}, "listing_keys": {"bama:g1"}},
}


@pytest.mark.parametrize(
    ("query", "brands", "models"),
    [
        ("AMG", {"بنز"}, []),
        ("G class", {"بنز"}, ["کلاس G"]),
        ("جی کلاس", {"بنز"}, ["کلاس G"]),
        ("پژو 206", {"پژو"}, ["206"]),
        ("مرسدس بنز", {"بنز"}, []),
        ("ماشین اتوماتیک کم کارکرد", set(), []),
    ],
)
def test_entity_detection(query, brands, models):
    e = detect_entities(query, VOCAB)
    assert e.brands == brands
    assert [m.label for m in e.models] == models


def test_amg_token_stays_a_keyword():
    """«amg» must remain a keyword so ads literally mentioning AMG («53AMG»)
    outrank the rest of the steered بنز subset."""
    e = detect_entities("AMG", VOCAB)
    assert "amg" in e.keyword_tokens


def test_model_pattern_word_boundary():
    """«کلاس G» must not match «کلاس GLC»."""
    pattern = model_match_pattern("کلاس G")
    assert re.search(pattern, "بنز، کلاس g، سال 1404")
    assert not re.search(pattern, "بنز، کلاس glc، سال 1404")


# --- value canonicalization --------------------------------------------------

def test_transmission_groups_merge():
    assert canonical_value("transmission", "اتومات") == "اتوماتیک"
    assert canonical_value("transmission", "دنده ای") == "دنده‌ای"
    assert canonical_value("transmission", "دنده\u200cای") == "دنده‌ای"


def test_transmission_filter_expansion():
    """A canonical selection expands to the raw variants constituting it."""
    expanded = variants_of("transmission", "اتوماتیک")
    assert "اتومات" in expanded and "اتوماتیک" in expanded


def test_unknown_values_pass_through():
    assert canonical_value("fuel", "سوخت اتمی") == "سوخت اتمی"
    assert variants_of("fuel", "سوخت اتمی") == ("سوخت اتمی",)


def test_facet_rollup():
    rolled = canonical_facet_values(
        "transmission",
        [("اتوماتیک", 100), ("اتومات", 10), ("دنده ای", 50), ("دنده\u200cای", 50)],
    )
    labels = {r["label"]: r["count"] for r in rolled}
    assert labels["اتوماتیک"] == 110
    assert labels["دنده‌ای"] == 100


# --- taxonomy extraction ------------------------------------------------------

def test_title_model_extraction():
    assert extract_model_from_title("پژو، 207") == "207"
    assert extract_model_from_title("بنز، کلاس G") == "کلاس G"
    assert extract_model_from_title("باما بدون ویرگول") is None
    assert extract_model_from_title(None) is None


# --- search service fusion + diversity ----------------------------------------

def _service_from_rows(metadata: list[dict], vocab: dict | None = VOCAB) -> SearchService:
    """Build a SearchService over hand-made metadata rows (offline)."""
    import pandas as pd

    defaults = {
        "listing_key": "",
        "source_platform": "bama",
        "text": "",
        "normalized_text": "",
        "original_url": None,
        "raw_payload_ref_json": "{}",
        "chunk_index": 0,
        "chunk_count": 1,
        "extraction_version": "test",
        "brand_raw": None,
        "brand_key": None,
        "model_raw": None,
        "trim_raw": None,
        "fuel_raw": None,
        "transmission_raw": None,
        "body_type_raw": None,
        "body_condition_raw": None,
        "brand_canonical": None,
        "model_key": None,
        "year_raw": None,
        "year_num": None,
        "year_calendar": None,
        "price_amount": None,
        "mileage_km": None,
        "location_raw": None,
        "image_urls_json": None,
        "first_seen_at": "2026-01-01T00:00:00",
    }
    rows = [{**defaults, **row} for row in metadata]
    df = pd.DataFrame(rows)
    encoder = HashingEncoder()
    vectors = np.asarray(
        encoder.encode_passages(df["normalized_text"].tolist()), dtype=np.float32
    )
    return SearchService(
        vectors, df, encoder,
        entity_vocab=vocab,
        brand_canonicals={"بنز", "ام جی", "شاهین", "پژو"},
    )


def test_amg_query_steers_to_benz_not_mg():
    """THE regression: "AMG" must surface بنز rows — never ام جی (MG) ads,
    no matter what the encoder's vectors say."""
    rows = [
        {"listing_key": "bama:mg1", "brand_key": "ام جی", "brand_raw": "ام جی",
         "normalized_text": "ام جی zs سال 1400", "text": "ام جی zs"},
        {"listing_key": "bama:mg2", "brand_key": "ام جی", "brand_raw": "ام جی",
         "normalized_text": "ام جی 5 جدید", "text": "ام جی 5"},
        {"listing_key": "bama:amg", "brand_key": "بنز", "brand_raw": "بنز",
         "normalized_text": "بنز gle کوپه 53amg سال 2026", "text": "بنز gle 53amg"},
        {"listing_key": "bama:c200", "brand_key": "بنز", "brand_raw": "بنز",
         "normalized_text": "بنز کلاس c سال 2013", "text": "بنز کلاس c"},
    ]
    service = _service_from_rows(rows)
    response = service.search("AMG", top_k=10)
    assert response.results, "steered subset is non-empty"
    assert {r.source_platform for r in response.results} == {"bama"}
    assert all("بنز" not in r.text or "ام جی" not in r.text for r in response.results)
    assert response.results[0].listing_key == "bama:amg"  # keyword hit ranks first
    assert "entity" in response.retrieval_mode


def test_g_class_query_surfaces_klasse_g():
    rows = [
        {"listing_key": "bama:shahin", "brand_key": "شاهین", "brand_raw": "شاهین",
         "normalized_text": "شاهین تیپ g سال 1403", "text": "شاهین تیپ G"},
        {"listing_key": "bama:g1", "brand_key": "بنز", "brand_raw": "بنز",
         "normalized_text": "بنز کلاس g سال 2020", "text": "بنز کلاس G",
         "model_raw": "کلاس G"},
        {"listing_key": "bama:gle", "brand_key": "بنز", "brand_raw": "بنز",
         "normalized_text": "بنز کلاس gle سال 2024", "text": "بنز کلاس GLE"},
    ]
    service = _service_from_rows(rows)
    response = service.search("G class", top_k=10)
    top_keys = [r.listing_key for r in response.results]
    assert "bama:g1" in top_keys[:2]
    assert "bama:shahin" not in top_keys


def test_explicit_filters_beat_entity_steering():
    """User brand filter سایپا wins even when the query names بنز."""
    rows = [
        {"listing_key": "sheypoor:s1", "brand_key": "سایپا", "brand_raw": "سایپا",
         "normalized_text": "تیبا دو سال 1399", "text": "تیبا"},
        {"listing_key": "bama:benz", "brand_key": "بنز", "brand_raw": "بنز",
         "normalized_text": "بنز کلاس c", "text": "بنز"},
    ]
    service = _service_from_rows(rows)
    filters = SearchFilters(brands=["سایپا"])
    response = service.search("بنز", filters, top_k=5)
    assert [r.listing_key for r in response.results] == ["sheypoor:s1"]


def test_browse_newest_diversifies_brands():
    """Brand-rotated sweeps show a wall of one model — the newest browse
    order must interleave brands (≤2 per brand) inside the recent window."""
    rows = []
    for i in range(6):
        rows.append(
            {
                "listing_key": f"bama:dena{i}",
                "brand_key": "دنا",
                "brand_raw": "دنا",
                "normalized_text": f"دنا پلاس {i}",
                "text": f"دنا {i}",
                "first_seen_at": f"2026-01-0{i + 1}T00:00:00",
            }
        )
    rows.append(
        {"listing_key": "bama:pride", "brand_key": "پراید", "brand_raw": "پراید",
         "normalized_text": "پراید 131", "text": "پراید",
         "first_seen_at": "2026-01-07T00:00:00"}
    )
    service = _service_from_rows(rows)
    response = service.search("", top_k=10, sort="newest")
    top3 = [r.listing_key for r in response.results[:3]]
    assert "bama:pride" in top3, "the lone پراید ad must surface in the top 3"
    dena_in_top3 = sum(1 for k in top3 if k.startswith("bama:dena"))
    assert dena_in_top3 <= 2


def test_natural_language_falls_back_to_semantic():
    """No entities, no keyword hits → pure semantic order, mode='semantic'."""
    rows = [
        {"listing_key": "bama:a", "brand_key": "پژو", "brand_raw": "پژو",
         "normalized_text": "پژو 206 سفید", "text": "پژو 206"},
        {"listing_key": "bama:b", "brand_key": "سایپا", "brand_raw": "سایپا",
         "normalized_text": "تیبا 2 قرمز", "text": "تیبا"},
    ]
    service = _service_from_rows(rows)
    response = service.search("یک خودروی خانوادگی بی‌نظیر")
    assert response.retrieval_mode == "semantic"
    assert len(response.results) == 2


# --- dashboard (bama-only, title-extracted models, canonical filters) ---------

def _store_with_bama_ads(tmp_path, monkeypatch):
    from icmi.core.records import CanonicalRecord, RawPayloadRef, utc_now
    from icmi.pipeline.store import CanonicalStore

    monkeypatch.setenv("ICMI_DATA_DIR", str(tmp_path / "data"))
    store = CanonicalStore()

    def add(key, title, brand, model_raw, price, year, transmission, km="45,000 km"):
        record = CanonicalRecord(
            source_platform="bama",
            source_listing_id=key,
            original_url=f"https://example.com/{key}",
            title=title,
            brand_raw=brand,
            model_raw=model_raw,
            year_raw=year,
            mileage_raw=km,
            price_raw=str(price),
            transmission_raw=transmission,
            scraped_at=utc_now(),
            extraction_version="test-v1",
            raw_payload_ref=RawPayloadRef(
                run_id="r", platform="bama", file="records.jsonl", key=key
            ),
        )
        store.upsert_listings([record], f"run-{key}")

    add("d1", "سایپا، دنا پلاس", "سایپا", "1403 | توربو", 800_000_000, 1403, "اتومات")
    add("d2", "سایپا، دنا پلاس", "سایپا", "1404 | توربو", 850_000_000, 1404, "اتوماتیک")
    add("p1", "پژو، 207", "پژو", "1404 | اتوماتیک توربو", 1_100_000_000, 1404, "اتومات")
    add("p2", "پژو، 207", "پژو", "1402 | دنده ای TU5P", 950_000_000, 1402, "دنده ای")
    return store


def test_dashboard_bama_default_and_extracted_models(tmp_path, monkeypatch):
    from icmi.analytics.dashboard import AnalyticsFilters, build_dashboard

    store = _store_with_bama_ads(tmp_path, monkeypatch)
    try:
        result = build_dashboard(store, AnalyticsFilters(
            brand_keys=[], models=[], year_min=None, year_max=None,
            year_calendar=None, platforms=[], fuel=[], transmission=[],
            mileage_max_km=None, price_min=None, price_max=None,
        ))
        assert result["total"] == 4
        # by_model uses the TITLE-extracted model, not the trim-like model_raw
        labels = {m["label"] for m in result["by_model"]}
        assert "دنا پلاس" in labels and "207" in labels
        assert not any("|" in label for label in labels), "no «year | trim» composites"
    finally:
        store.close()


def test_dashboard_models_filter_matches_extracted_model(tmp_path, monkeypatch):
    from icmi.analytics.dashboard import AnalyticsFilters, build_dashboard

    store = _store_with_bama_ads(tmp_path, monkeypatch)
    try:
        result = build_dashboard(store, AnalyticsFilters(
            brand_keys=[], models=["207"], year_min=None, year_max=None,
            year_calendar=None, platforms=[], fuel=[], transmission=[],
            mileage_max_km=None, price_min=None, price_max=None,
        ))
        assert result["total"] == 2  # only the two 207 ads
    finally:
        store.close()


def test_dashboard_transmission_canonical_expansion(tmp_path, monkeypatch):
    """Canonical «اتوماتیک» selection matches the raw «اتومات» variant too."""
    from icmi.analytics.dashboard import AnalyticsFilters, build_dashboard
    from icmi.core.value_aliases import variants_of

    store = _store_with_bama_ads(tmp_path, monkeypatch)
    try:
        result = build_dashboard(store, AnalyticsFilters(
            brand_keys=[], models=[], year_min=None, year_max=None,
            year_calendar=None, platforms=[],
            fuel=[], transmission=list(variants_of("transmission", "اتوماتیک")),
            mileage_max_km=None, price_min=None, price_max=None,
        ))
        assert result["total"] == 3  # d1(اتومات) + d2(اتوماتیک) + p1(اتومات)
    finally:
        store.close()


# --- filter audit fixes (owner round 2026-11-22, second pass) ----------------

def test_models_filter_matches_extracted_model_via_vocab():
    """Facets list clean title-extracted models; the filter must match them
    even though bama's model_raw is a year|trim composite with no model in
    it (regression from the facet cleanup: «207» matched nothing)."""
    rows = [
        {"listing_key": "bama:207a", "brand_key": "پژو", "brand_raw": "پژو",
         "model_raw": "1404 | اتوماتیک توربو",
         "normalized_text": "پژو 207 اتوماتیک", "text": "پژو 207"},
        {"listing_key": "bama:other", "brand_key": "پژو", "brand_raw": "پژو",
         "model_raw": "1402 | دنده ای",
         "normalized_text": "پژو پارس دنده ای", "text": "پژو پارس"},
    ]
    vocab = {"207": {"label": "207", "brands": {"پژو"},
                     "listing_keys": {"bama:207a"}}}
    service = _service_from_rows(rows, vocab=vocab)
    filters = SearchFilters(models=["207"])
    response = service.search("", filters, top_k=10)
    assert [r.listing_key for r in response.results] == ["bama:207a"]


def test_models_filter_falls_back_to_model_raw_without_vocab():
    rows = [
        {"listing_key": "sheypoor:x", "brand_key": "پژو", "brand_raw": "پژو",
         "model_raw": "206 sd v20",
         "normalized_text": "پژو 206 sd", "text": "پژو 206 sd"},
        {"listing_key": "sheypoor:y", "brand_key": "پژو", "brand_raw": "پژو",
         "model_raw": "پارس",
         "normalized_text": "پژو پارس", "text": "پژو پارس"},
    ]
    service = _service_from_rows(rows, vocab=None)
    assert not service.entity_vocab  # built without a vocabulary
    response = service.search("", SearchFilters(models=["206"]), top_k=10)
    assert [r.listing_key for r in response.results] == ["sheypoor:x"]


def test_trim_label_strips_year_prefix():
    """Estimate trim options must not embed the model year — the year is a
    separate field («1405 | دنده ای TU5P» → «دنده ای TU5P»; a bare «1402»
    is not a trim at all)."""
    assert clean_trim_label("1405 | دنده ای TU5P") == "دنده ای TU5P"
    assert clean_trim_label("2026 | 2.0 لیتر") == "2.0 لیتر"
    assert clean_trim_label("1402 ") is None
    assert clean_trim_label("اتوماتیک توربو") == "اتوماتیک توربو"
    assert clean_trim_label(None) is None


def test_dashboard_year_filter_normalizes_gregorian(tmp_path, monkeypatch):
    """«از سال 1400» must not silently exclude imported cars whose source
    year is Gregorian (2024 ≈ 1403) — the filter matches the chart's
    model-year normalization."""
    from icmi.analytics.dashboard import AnalyticsFilters, build_dashboard

    monkeypatch.setenv("ICMI_DATA_DIR", str(tmp_path / "data"))
    from icmi.core.records import CanonicalRecord, RawPayloadRef, utc_now
    from icmi.pipeline.store import CanonicalStore

    store = CanonicalStore()
    try:
        for key, year_raw, _year_num, _cal, price in (
            ("g1", 2024, 2024, "gregorian", 900_000_000),
            ("s1", 1404, 1404, "shamsi", 800_000_000),
            ("s0", 1395, 1395, "shamsi", 700_000_000),
        ):
            store.upsert_listings([CanonicalRecord(
                source_platform="bama", source_listing_id=key,
                original_url=f"https://example.com/{key}",
                title=f"تست، {key}", brand_raw="تست", year_raw=year_raw,
                price_raw=str(price), scraped_at=utc_now(),
                extraction_version="test-v1",
                raw_payload_ref=RawPayloadRef(
                    run_id="r", platform="bama", file="records.jsonl", key=key
                ),
            )], f"run-{key}")
        result = build_dashboard(store, AnalyticsFilters(
            brand_keys=[], models=[], year_min=1400, year_max=None,
            year_calendar=None, platforms=[], fuel=[], transmission=[],
            mileage_max_km=None, price_min=None, price_max=None,
        ))
        assert result["total"] == 2  # g1 (2024≈1403) + s1 (1404), not s0
    finally:
        store.close()
