"""Tests for normalization, the canonical store, and cross-run ETL."""

import json

from icmi.core.records import CanonicalRecord, RawPayloadRef, utc_now
from icmi.pipeline.etl import ingest_run
from icmi.pipeline.normalize import (
    PRICE_NORMALIZATION_VERSION,
    normalize_price_amount,
    normalize_year,
)
from icmi.pipeline.store import CanonicalStore
from tests.fixtures.bama_payloads import bama_ad
from tests.fixtures.other_platforms import hamrah_ad

# --- normalization ------------------------------------------------------------


def test_normalize_price_handles_both_formats_and_digits():
    assert normalize_price_amount("7,100,000,000") == 7_100_000_000
    assert normalize_price_amount(7_100_000_000) == 7_100_000_000
    assert normalize_price_amount("7100000000") == 7_100_000_000  # same price!
    assert normalize_price_amount("۷،۰۰۰،۰۰۰") == 7_000_000  # Persian digits + comma
    assert normalize_price_amount("2,050,000,000") == 2_050_000_000
    assert normalize_price_amount("توافقی") is None
    assert normalize_price_amount("") is None
    assert normalize_price_amount(None) is None
    assert normalize_price_amount(True) is None


def test_normalize_year_preserves_calendar_ambiguity():
    assert normalize_year(1398) == 1398
    assert normalize_year("2012") == 2012
    assert normalize_year(None) is None


# --- store: cross-run identity and price history --------------------------------


def make_record(
    platform: str,
    listing_id: str,
    *,
    price_raw=7_100_000_000,
    currency: str | None = None,
    run: str = "run-1",
) -> CanonicalRecord:
    return CanonicalRecord(
        source_platform=platform,
        source_listing_id=listing_id,
        title="تست",
        price_raw=price_raw,
        price_currency_raw=currency,
        scraped_at=utc_now(),
        extraction_version="test-v1",
        raw_payload_ref=RawPayloadRef(
            run_id=run, platform=platform, file="records.jsonl", key=listing_id
        ),
    )


def test_cross_run_dedup_same_ad_maps_to_one_listing(tmp_path):
    store = CanonicalStore(db_path=tmp_path / "icmi.db")
    try:
        s1 = store.upsert_listings([make_record("bama", "abc")], "run-1")
        s2 = store.upsert_listings([make_record("bama", "abc")], "run-2")
        assert (s1.new_listings, s1.updated_listings) == (1, 0)
        assert (s2.new_listings, s2.updated_listings) == (0, 1)
        assert store.listing_count() == 1  # NOT two rows
        assert store.observation_count("bama:abc") == 1  # unchanged price → one obs
    finally:
        store.close()


def test_platform_scoped_identity(tmp_path):
    store = CanonicalStore(db_path=tmp_path / "icmi.db")
    try:
        # same numeric listing id on different platforms = different listings
        store.upsert_listings([make_record("bama", "123")], "run-1")
        store.upsert_listings([make_record("sheypoor", "123")], "run-1")
        assert store.listing_count() == 2
    finally:
        store.close()


def test_formatting_change_is_not_a_price_change(tmp_path):
    store = CanonicalStore(db_path=tmp_path / "icmi.db")
    try:
        store.upsert_listings(
            [make_record("bama", "abc", price_raw="7,100,000,000")], "run-1"
        )
        stats = store.upsert_listings(
            [make_record("bama", "abc", price_raw=7_100_000_000)], "run-2"
        )
        assert stats.price_observations_appended == 0
        assert stats.price_change_events == 0
        assert store.observation_count("bama:abc") == 1
    finally:
        store.close()


def test_real_price_change_appends_observation(tmp_path):
    store = CanonicalStore(db_path=tmp_path / "icmi.db")
    try:
        store.upsert_listings(
            [make_record("bama", "abc", price_raw="7,100,000,000")], "run-1"
        )
        stats = store.upsert_listings(
            [make_record("bama", "abc", price_raw="7,300,000,000")], "run-2"
        )
        assert stats.price_observations_appended == 1
        assert stats.price_change_events == 1
        assert any("7100000000 -> 7300000000" in d for d in stats.details)
        assert store.observation_count("bama:abc") == 2
    finally:
        store.close()


def test_currency_change_appends_observation(tmp_path):
    store = CanonicalStore(db_path=tmp_path / "icmi.db")
    try:
        store.upsert_listings(
            [make_record("sheypoor", "x", price_raw="2,050,000,000", currency="تومان")],
            "run-1",
        )
        stats = store.upsert_listings(
            [make_record("sheypoor", "x", price_raw="2,050,000,000", currency=None)],
            "run-2",
        )
        assert stats.price_observations_appended == 1  # currency difference matters
        assert stats.price_change_events == 0  # amount itself unchanged
    finally:
        store.close()


def test_missing_price_never_touches_history(tmp_path):
    store = CanonicalStore(db_path=tmp_path / "icmi.db")
    try:
        store.upsert_listings([make_record("bama", "abc")], "run-1")
        stats = store.upsert_listings(
            [make_record("bama", "abc", price_raw=None)], "run-2"
        )
        assert stats.price_observations_appended == 0
        assert store.observation_count("bama:abc") == 1  # history intact
        row = store._conn.execute(
            "SELECT price_amount FROM listings WHERE listing_key='bama:abc'"
        ).fetchone()
        assert row["price_amount"] == 7_100_000_000  # last observation retained
    finally:
        store.close()


def test_records_without_id_are_counted_not_crashed(tmp_path):
    store = CanonicalStore(db_path=tmp_path / "icmi.db")
    try:
        record = make_record("bama", "abc")
        # parse_listing guards ids, but the store defends in depth too:
        record.source_listing_id = ""
        stats = store.upsert_listings([record], "run-1")
        assert stats.records_without_id == 1
        assert store.listing_count() == 0
    finally:
        store.close()


def test_observation_carries_normalization_version(tmp_path):
    store = CanonicalStore(db_path=tmp_path / "icmi.db")
    try:
        store.upsert_listings([make_record("bama", "abc")], "run-1")
        row = store._conn.execute(
            "SELECT price_normalization_version FROM price_observations "
            "WHERE listing_key='bama:abc'"
        ).fetchone()
        assert row["price_normalization_version"] == PRICE_NORMALIZATION_VERSION
    finally:
        store.close()


# --- ETL: raw → canonical via re-extraction --------------------------------------


def test_ingest_run_reextracts_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("ICMI_DATA_DIR", str(tmp_path / "data"))
    from icmi.core.config import load_sources
    from icmi.core.paths import manifests_dir, raw_dir

    run_id = "testrun-bama-local"
    platform = "bama"
    ad = bama_ad("abc123")
    records_path = raw_dir() / run_id / platform / "records.jsonl"
    records_path.parent.mkdir(parents=True)
    records_path.write_text(
        json.dumps({"source_listing_id": "abc123", "raw_payload": ad}) + "\n",
        encoding="utf-8",
    )
    manifest_path = manifests_dir() / f"{run_id}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"run_id": run_id, "platform": platform, "status": "success",
                    "execution_mode": "local", "collection": "full"}),
        encoding="utf-8",
    )

    sources = load_sources()
    store = CanonicalStore()
    try:
        s1 = ingest_run(run_id, platform, store, sources.sources[platform], sources.pipeline)
        assert s1.new_listings == 1
        assert store.listing_count("bama") == 1
        row = store._conn.execute(
            "SELECT title, original_url, raw_payload_ref_json FROM listings "
            "WHERE listing_key='bama:abc123'"
        ).fetchone()
        assert row["title"] == "پژو 206، مدل 1398، تیپ 2"
        assert row["original_url"].startswith("https://bama.ir/")
        ref = json.loads(row["raw_payload_ref_json"])
        assert ref["run_id"] == run_id and ref["file"].startswith("records.jsonl#")

        # Idempotent re-ingest, two layers:
        # 1) run-level fast path — the run is already recorded, skipped entirely
        s2 = ingest_run(run_id, platform, store, sources.sources[platform], sources.pipeline)
        assert s2.skipped_runs == 1
        assert s2.new_listings == 0 and s2.updated_listings == 0
        # 2) record-level safety net — covered by
        #    test_cross_run_dedup_same_ad_maps_to_one_listing (re-upsert of the
        #    same records yields updated=1 and no duplicate observations)
        assert store.observation_count("bama:abc123") == 1

        runs = store._conn.execute("SELECT run_id FROM runs").fetchall()
        assert [r["run_id"] for r in runs] == [run_id]
    finally:
        store.close()


def test_hamrah_record_ingests_without_price_fabrication(tmp_path, monkeypatch):
    monkeypatch.setenv("ICMI_DATA_DIR", str(tmp_path / "data"))
    from icmi.core.config import load_sources
    from icmi.core.paths import raw_dir

    run_id = "testrun-hamrah-local"
    platform = "hamrah_mechanic"
    ad = hamrah_ad(3595004)
    records_path = raw_dir() / run_id / platform / "records.jsonl"
    records_path.parent.mkdir(parents=True)
    records_path.write_text(
        json.dumps({"source_listing_id": "3595004", "raw_payload": ad}) + "\n",
        encoding="utf-8",
    )
    sources = load_sources()
    store = CanonicalStore()
    try:
        # simulate a sold skip during re-extraction: parse_only + isSold
        ad_sold = dict(ad)
        ad_sold["isSold"] = True
        records_path.write_text(
            json.dumps({"source_listing_id": "3595004", "raw_payload": ad_sold}) + "\n",
            encoding="utf-8",
        )
        stats = ingest_run(run_id, platform, store, sources.sources[platform], sources.pipeline)
        assert stats.new_listings == 0  # sold listing skipped by re-extraction
        assert store.listing_count() == 0
    finally:
        store.close()
