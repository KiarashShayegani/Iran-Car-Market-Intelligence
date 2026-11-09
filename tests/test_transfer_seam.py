"""Tests for the F2b transfer seam: analytical export, snapshot publish/import,
run-level fast path, and cross-source dedup into the authoritative local DB.

No network. "Actions" snapshots are simulated with a second, separate data
directory (a different ICMI_DATA_DIR), exactly like a real artifact would be.
"""

import json

import pytest
from scripts.publish_data import publish
from scripts.sync_latest_data import _extract_if_zip

from icmi.core.records import CanonicalRecord, RawPayloadRef, utc_now
from icmi.pipeline.analytical import export_analytics
from icmi.pipeline.store import CanonicalStore


def make_record(platform, listing_id, *, price_raw="7,100,000,000", currency=None):
    return CanonicalRecord(
        source_platform=platform,
        source_listing_id=listing_id,
        title="تست",
        price_raw=price_raw,
        price_currency_raw=currency,
        scraped_at=utc_now(),
        extraction_version="test-v1",
        raw_payload_ref=RawPayloadRef(
            run_id="run-x", platform=platform, file="records.jsonl", key=listing_id
        ),
    )


@pytest.fixture()
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ICMI_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


def test_analytical_export_row_counts(isolated_data_dir):
    store = CanonicalStore()
    try:
        store.upsert_listings([make_record("bama", "a"), make_record("bama", "b")], "run-1")
        result = export_analytics(store)
        assert result["listings_rows"] == 2
        assert result["observation_rows"] == 2
        df = store.query_df("SELECT listing_key, price_amount FROM listings ORDER BY 1")
        assert list(df["listing_key"]) == ["bama:a", "bama:b"]
        assert list(df["price_amount"]) == [7_100_000_000, 7_100_000_000]
    finally:
        store.close()


def _write_run(raw_root, run_id, platform, listing_id, price_raw):
    records = raw_root / run_id / platform / "records.jsonl"
    records.parent.mkdir(parents=True)
    ad = {
        "detail": {
            "code": listing_id,
            "title": "پژو 206، مدل 1398، تیپ 2",
            "url": f"/car/x/{listing_id}",
            "year": 1398,
        },
        "metadata": {},
        "price": {"price": price_raw, "currency": None, "type": "price"},
    }
    records.write_text(
        json.dumps({"source_listing_id": listing_id, "raw_payload": ad}) + "\n",
        encoding="utf-8",
    )


def _ingest_all(platform="bama"):
    from icmi.core.config import load_sources
    from icmi.core.paths import raw_dir
    from icmi.pipeline.etl import ingest_run

    sources = load_sources()
    raw_root = raw_dir()
    store = CanonicalStore()
    try:
        for run_dir in sorted(raw_root.iterdir()):
            if not (run_dir / platform / "records.jsonl").is_file():
                continue
            yield_from = ingest_run(
                run_id=run_dir.name,
                platform=platform,
                store=store,
                source=sources.sources[platform],
                pipeline=sources.pipeline,
            )
            yield yield_from
    finally:
        store.close()


def test_publish_import_round_trip_into_fresh_store(tmp_path, isolated_data_dir):
    # Local machine: collect a run, ingest, export analytics
    _write_run(isolated_data_dir / "raw", "local-run-1", "bama", "abc", "7,100,000,000")
    stats = list(_ingest_all())[0]
    assert stats.new_listings == 1
    export_analytics(CanonicalStore())

    # Publish a snapshot (the artifact Actions would upload)
    snapshot = publish(
        tmp_path / "out", include_raw=True, include_db=True, zip_output=True
    )
    assert snapshot.suffix == ".zip"

    # A "different device": empty data dir — run the real sync entrypoint
    from scripts.sync_latest_data import main as sync_main

    monkey_other = pytest.MonkeyPatch()
    monkey_other.setenv("ICMI_DATA_DIR", str(tmp_path / "other-device-data"))
    try:
        sync_main(["--source", str(snapshot)])

        other_store = CanonicalStore()
        try:
            assert other_store.listing_count("bama") == 1
            row = other_store._conn.execute(
                "SELECT price_amount FROM price_observations WHERE listing_key='bama:abc'"
            ).fetchone()
            assert row["price_amount"] == 7_100_000_000
            assert other_store.ingested_run_ids() == ["local-run-1"]

            # importing the same artifact again changes nothing (idempotence)
            sync_main(["--source", str(snapshot)])
            assert other_store.listing_count("bama") == 1
            assert other_store.observation_count("bama:abc") == 1
        finally:
            other_store.close()
    finally:
        monkey_other.undo()


def test_importing_same_run_twice_is_idempotent(isolated_data_dir):
    _write_run(isolated_data_dir / "raw", "run-1", "bama", "abc", "7,100,000,000")
    store = CanonicalStore()
    try:
        from icmi.core.config import load_sources
        from icmi.pipeline.etl import ingest_run

        sources = load_sources()
        s1 = ingest_run("run-1", "bama", store, sources.sources["bama"], sources.pipeline)
        s2 = ingest_run("run-1", "bama", store, sources.sources["bama"], sources.pipeline)
        assert (s1.new_listings, s1.price_observations_appended) == (1, 1)
        assert s2.skipped_runs == 1
        assert s2.new_listings == 0 and s2.price_observations_appended == 0
        assert store.listing_count() == 1
        assert store.observation_count("bama:abc") == 1
    finally:
        store.close()


def test_cross_source_dedup_local_and_actions_runs(isolated_data_dir):
    """The same ad collected locally (run A) and by Actions (run B) merges into
    one authoritative listing with distinct run provenance on observations."""
    from icmi.core.config import load_sources
    from icmi.pipeline.etl import ingest_run

    sources = load_sources()
    store = CanonicalStore()
    try:
        _write_run(isolated_data_dir / "raw", "local-run", "bama", "abc", "7,100,000,000")
        _write_run(isolated_data_dir / "raw", "actions-run", "bama", "abc", "7,100,000,000")

        s_local = ingest_run(
            "local-run", "bama", store, sources.sources["bama"], sources.pipeline
        )
        s_actions = ingest_run(
            "actions-run", "bama", store, sources.sources["bama"], sources.pipeline
        )

        assert s_local.new_listings == 1
        assert s_actions.new_listings == 0  # same ad → same listing (cross-run dedup)
        assert s_actions.updated_listings == 1
        assert store.listing_count() == 1
        # price unchanged across runs → still a single observation
        assert store.observation_count("bama:abc") == 1

        # runs table records both provenances
        runs = store.ingested_run_ids()
        assert "local-run" in runs and "actions-run" in runs
    finally:
        store.close()


def test_actions_price_change_flows_into_authoritative_history(isolated_data_dir):
    from icmi.core.config import load_sources
    from icmi.pipeline.etl import ingest_run

    sources = load_sources()
    store = CanonicalStore()
    try:
        _write_run(isolated_data_dir / "raw", "run-1", "bama", "abc", "7,100,000,000")
        _write_run(isolated_data_dir / "raw", "run-2", "bama", "abc", "7,300,000,000")
        ingest_run("run-1", "bama", store, sources.sources["bama"], sources.pipeline)
        ingest_run("run-2", "bama", store, sources.sources["bama"], sources.pipeline)
        assert store.observation_count("bama:abc") == 2
        row = store._conn.execute(
            "SELECT price_amount FROM price_observations WHERE listing_key='bama:abc' "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row["price_amount"] == 7_300_000_000
    finally:
        store.close()


def test_zip_extraction_rejects_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        _extract_if_zip(tmp_path / "nope.zip", tmp_path)
