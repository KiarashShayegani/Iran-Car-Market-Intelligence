"""Tests for document preparation, market stats, and the benchmark harness
(fully offline; the benchmark runs against a deterministic fake encoder)."""

import json

from icmi.core.records import CanonicalRecord, RawPayloadRef, utc_now
from icmi.pipeline.store import CanonicalStore
from icmi.retrieval.benchmark import BenchmarkResult, run_benchmark
from icmi.retrieval.documents import compose_text, iter_documents


def make_listing(store: CanonicalStore, platform: str, lid: str, **fields) -> None:
    record = CanonicalRecord(
        source_platform=platform,
        source_listing_id=lid,
        original_url=f"https://example.com/{lid}",
        title=fields.get("title", "پژو 206 تیپ 2"),
        description=fields.get("description", "بدون رنگ، بیمه کامل، دو لاستیک نو."),
        brand_raw=fields.get("brand_raw", "پژو"),
        model_raw=fields.get("model_raw", "پژو 206"),
        trim_raw=fields.get("trim_raw"),
        year_raw=fields.get("year_raw", 1398),
        mileage_raw=fields.get("mileage_raw", "45,000 km"),
        price_raw=fields.get("price_raw", "750,000,000"),
        price_currency_raw=None,
        location_raw=fields.get("location_raw", "تهران"),
        fuel_raw="بنزینی",
        transmission_raw="دنده ای",
        color_raw="سفید",
        scraped_at=utc_now(),
        extraction_version="test-v1",
        raw_payload_ref=RawPayloadRef(
            run_id="r", platform=platform, file="records.jsonl", key=lid
        ),
    )
    store.upsert_listings([record], "run-test")


def test_compose_text_includes_facts_and_verbatim_description(tmp_path, monkeypatch):
    monkeypatch.setenv("ICMI_DATA_DIR", str(tmp_path / "data"))
    store = CanonicalStore()
    try:
        make_listing(store, "bama", "a1")
        rows = store.query_df("SELECT * FROM listings")
        text = compose_text(rows.iloc[0])
        assert "پژو" in text and "سال 1398" in text
        assert "بدون رنگ، بیمه کامل، دو لاستیک نو." in text  # description verbatim
        assert text.count("\n") >= 1  # header separated from description
    finally:
        store.close()


def test_documents_carry_traceability_and_normalized_copy(tmp_path, monkeypatch):
    monkeypatch.setenv("ICMI_DATA_DIR", str(tmp_path / "data"))
    store = CanonicalStore()
    try:
        make_listing(store, "bama", "a1")
        docs = list(iter_documents(store))
        assert len(docs) == 1
        doc = docs[0]
        assert doc.listing_key == "bama:a1"
        assert doc.original_url == "https://example.com/a1"
        ref = json.loads(doc.raw_payload_ref_json)
        assert ref["key"] == "a1" and ref["file"] == "records.jsonl"
        assert doc.normalized_text and "می\u200cشود" not in doc.normalized_text
        assert doc.document_version == "doc-v1"
    finally:
        store.close()


def test_long_documents_chunk_with_boundaries(tmp_path, monkeypatch):
    monkeypatch.setenv("ICMI_DATA_DIR", str(tmp_path / "data"))
    long_description = "جمله تست. " * 200  # ~2000 chars
    store = CanonicalStore()
    try:
        make_listing(store, "bama", "a2", description=long_description)
        docs = list(iter_documents(store))
        assert len(docs) > 1  # chunked
        assert all(d.chunk_count == len(docs) for d in docs)
        assert {d.chunk_index for d in docs} == set(range(len(docs)))
    finally:
        store.close()


def test_market_stats_respect_calendar_and_null_safety(tmp_path, monkeypatch):
    monkeypatch.setenv("ICMI_DATA_DIR", str(tmp_path / "data"))
    from icmi.analytics.stats import market_stats

    store = CanonicalStore()
    try:
        make_listing(store, "bama", "a1", year_raw=1398, price_raw="750,000,000")
        make_listing(store, "bama", "a2", year_raw=2012, price_raw="900,000,000",
                     brand_raw="هیوندای", mileage_raw="صفر کیلومتر")
        make_listing(store, "bama", "a3", year_raw="کارکرده", price_raw=None)

        result = market_stats(store)
        assert result["overview"]["count"].sum() == 2  # a3 has no price → excluded
        by_year = result["by_year"]
        assert set(by_year["year_calendar"]) == {"shamsi", "gregorian"}  # never blended
        assert by_year[by_year["year_calendar"] == "gregorian"]["count"].sum() == 1

        # mileage parsing checks via stats side effects (brand medians table)
        assert result["meta"]["price_normalization_version"] == "price-norm-1"
    finally:
        store.close()


class FakeEncoder:
    """Deterministic bag-of-character-bigrams encoder (offline benchmark)."""

    model_id = "fake-bigram"

    def _vector(self, text):
        vec = [0.0] * 128
        for a, b in zip(text, text[1:], strict=False):
            vec[(hash((a, b)) % 128)] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def encode_passages(self, texts):
        return [self._vector(t) for t in texts]

    def encode_query(self, text):
        return self._vector(text)


def test_benchmark_harness_runs_offline_and_reports_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("ICMI_DATA_DIR", str(tmp_path / "data"))
    store = CanonicalStore()
    try:
        for i in range(12):
            make_listing(
                store, "bama", f"id{i}",
                title=f"پژو 206 تیپ {i}",
                model_raw=f"پژو 206 تیپ {i}",
                description=f"توضیحات شماره {i} با جزئیات مختلف برای تست.",
            )
        docs = list(iter_documents(store))
        assert len(docs) == 12
        result = run_benchmark(FakeEncoder(), docs, sample_size=6)
        assert isinstance(result, BenchmarkResult)
        assert result.model_id == "fake-bigram"
        assert 0.0 <= result.mrr_at_10 <= 1.0
        assert 0.0 <= result.top1_accuracy <= 1.0
        assert result.peak_rss_mb > 0
        # A bigram identity query should find the exact doc reliably:
        assert result.top1_accuracy >= 0.5
    finally:
        store.close()
