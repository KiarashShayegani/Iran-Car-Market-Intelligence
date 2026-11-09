"""Tests for the search service: pre-filter guarantee, traceability,
explainability, and honest zero-results (fully offline via HashingEncoder)."""

import pytest

from icmi.core.records import CanonicalRecord, RawPayloadRef, utc_now
from icmi.pipeline.store import CanonicalStore
from icmi.retrieval.index import build_index_metadata, load_index, save_index
from icmi.retrieval.search import SearchFilters, SearchService
from tests.fakes import HashingEncoder


def make_listing(store, platform, lid, **f):
    record = CanonicalRecord(
        source_platform=platform,
        source_listing_id=lid,
        original_url=f"https://example.com/{lid}",
        title=f.get("title", "پژو 206 تیپ 2"),
        description=f.get("description", "بدون رنگ، بیمه کامل."),
        brand_raw=f.get("brand_raw", "پژو"),
        model_raw=f.get("model_raw", "پژو 206"),
        year_raw=f.get("year_raw", 1398),
        mileage_raw=f.get("mileage_raw", "45,000 km"),
        price_raw=f.get("price_raw", "750,000,000"),
        location_raw=f.get("location_raw", "تهران"),
        scraped_at=utc_now(),
        extraction_version="test-v1",
        raw_payload_ref=RawPayloadRef(
            run_id="r", platform=platform, file="records.jsonl", key=lid
        ),
    )
    store.upsert_listings([record], "run-test")


@pytest.fixture()
def indexed_service(tmp_path, monkeypatch):
    monkeypatch.setenv("ICMI_DATA_DIR", str(tmp_path / "data"))
    store = CanonicalStore()
    try:
        make_listing(store, "bama", "p1")
        make_listing(store, "bama", "p2", model_raw="پژو پارس",
                     description="پارس خودرو، دنده ای، سفید.")
        make_listing(
            store, "bama", "t1", brand_raw="تویوتا", model_raw="کریسida",
            title="کریسida 2011", year_raw=2011,
            description="تویوتا کریسida وارداتی، اتوماتیک، کم کارکرد.",
        )
        make_listing(
            store, "sheypoor", "s1", brand_raw="سایپا", model_raw="تیبا",
            title="تیبا 2", price_raw="480,000,000",
            description="تیبا دو صفر کیلومتر، ارزان قیمت.",
        )
        chunks = build_index_metadata(store)
    finally:
        store.close()
    encoder = HashingEncoder()
    import numpy as np

    vectors = np.asarray(encoder.encode_passages([c.normalized_text for c in chunks]))
    save_index(vectors, chunks, encoder.model_id, out_dir=tmp_path / "idx")
    vectors2, metadata, info = load_index(tmp_path / "idx")
    assert info["vector_count"] == len(chunks)
    return SearchService(vectors2, metadata, encoder)


def test_semantic_search_returns_traceable_explainable_results(indexed_service):
    response = indexed_service.search("تویوتا اتوماتیک وارداتی", top_k=3)
    assert response.total_corpus == 4
    assert response.filters.is_empty()
    assert response.results, "corpus has a Toyota ad — must not be empty"
    top = response.results[0]
    assert top.listing_key == "bama:t1"
    assert top.source_platform == "bama"
    assert -1.0 <= top.score <= 1.0
    assert top.original_url == "https://example.com/t1"
    assert "کریسida" in top.text  # matching excerpt present
    ref = top.raw_payload_ref_json  # traceability survives retrieval
    assert ref and "t1" in ref
    assert top.extraction_version == "test-v1"


def test_filters_are_genuinely_restrictive(indexed_service):
    """Even when the query is semantically about a Toyota, brand+price filters
    restrict candidates BEFORE ranking — no Toyota can appear."""
    filters = SearchFilters(brands=["سایپا"], price_max=600_000_000)
    response = indexed_service.search("تویوتا اتوماتیک وارداتی لوکس", filters, top_k=5)
    assert response.candidates_after_filters == 1
    assert {r.listing_key for r in response.results} <= {"sheypoor:s1"}
    assert all(r.source_platform == "sheypoor" for r in response.results)


def test_zero_results_are_honest_not_loosened(indexed_service):
    filters = SearchFilters(price_max=100_000_000)  # nothing this cheap
    response = indexed_service.search("پژو", filters, top_k=5)
    assert response.candidates_after_filters == 0
    assert response.results == []


def test_price_filter_excludes_unpriced_listings(indexed_service):
    # add a listing without a price
    store = CanonicalStore()
    try:
        make_listing(store, "bama", "noprice", price_raw=None)
        chunks = build_index_metadata(store)
    finally:
        store.close()
    import numpy as np

    encoder = indexed_service.encoder
    vectors = np.asarray(encoder.encode_passages([c.normalized_text for c in chunks]))
    service = SearchService(vectors, indexed_service.metadata, encoder)
    # rebuild metadata to include the new chunk
    import pandas as pd

    new_row = pd.DataFrame([vars(c) for c in chunks if c.listing_key == "bama:noprice"])
    service.metadata = pd.concat([indexed_service.metadata, new_row], ignore_index=True)
    service.vectors = vectors

    filters = SearchFilters(price_min=1)
    response = service.search("پژو", filters, top_k=10)
    assert "bama:noprice" not in {r.listing_key for r in response.results}
    assert response.candidates_after_filters == len(indexed_service.metadata)


def test_calendar_restricted_year_filter(indexed_service):
    filters = SearchFilters(year_min=2010, year_calendar="gregorian")
    response = indexed_service.search("ماشین", filters, top_k=10)
    assert response.candidates_after_filters == 1  # only the 2011 Toyota


def test_invalid_filter_ranges_rejected():
    with pytest.raises(ValueError, match="price_min exceeds"):
        SearchFilters(price_min=10, price_max=5)
    with pytest.raises(ValueError, match="year_calendar"):
        SearchFilters(year_calendar="jalali")
