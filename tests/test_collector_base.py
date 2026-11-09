"""Tests for the collector base loop (dedup, stop conditions, persistence).

No network: _fetch_json is replaced with prepared page payloads.
"""

import json

import pytest

from icmi.collectors.base import BaseCollector
from icmi.core.config import PipelineConfig, SourceConfig
from tests.fixtures.bama_payloads import bama_ad


def make_source(**overrides) -> SourceConfig:
    defaults = dict(
        display_name="Bama",
        display_name_fa="باما",
        enabled=True,
        status="experimental",
        base_url="https://bama.ir/cad/api/search",
        field_support=[
            "source_listing_id", "original_url", "title", "description",
            "year_raw", "mileage_raw", "price_raw",  # trim_raw intentionally NOT supported
        ],
    )
    defaults.update(overrides)
    return SourceConfig(**defaults)


def make_pipeline(**overrides) -> PipelineConfig:
    defaults = dict(
        extraction_version="test-v1",
        request_timeout_seconds=5,
        request_delay_seconds=0,
        request_jitter_seconds=0,
        max_retries=1,
        user_agent="test-agent",
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


class FakeCollector(BaseCollector):
    """BaseCollector wired to canned pages, exercising the shared loop."""

    source_key = "fake"

    def __init__(self, pages, **kwargs):
        super().__init__(**kwargs)
        self._pages = list(pages)

    def _fetch_json(self, params):
        # Same contract as BaseCollector._fetch_json minus HTTP/retry.
        # Beyond the prepared pages, behave like an exhausted feed (empty page).
        self.pages["requested"] += 1
        self.pages["fetched"] += 1
        return self._pages.pop(0) if self._pages else {"ads": []}

    def page_params(self, page_index):
        return {"pageIndex": page_index}

    def iter_ads(self, page_payload):
        return page_payload.get("ads", [])

    def listing_id_of(self, ad):
        code = ad["detail"]["code"]
        return str(code) if code else None

    def parse_listing(self, ad, payload_ref):
        detail = ad["detail"]
        return CanonicalRecordForTest(detail, payload_ref)


from icmi.core.records import CanonicalRecord, utc_now  # noqa: E402


def CanonicalRecordForTest(detail, payload_ref):
    return CanonicalRecord(
        source_platform="fake",
        source_listing_id=str(detail["code"]),
        title=detail.get("title"),
        description=detail.get("description"),
        year_raw=detail.get("year"),
        scraped_at=utc_now(),
        extraction_version="test-v1",
        raw_payload_ref=payload_ref,
    )


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ICMI_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


def run_collector(pages, data_dir, **kwargs):
    monkey_source = make_source()
    collector = FakeCollector(
        pages=pages,
        source=monkey_source,
        pipeline=make_pipeline(),
        run_id="run-1",
        execution_mode="local",
    )
    return collector.collect(**kwargs)


def test_collect_dedups_within_run_and_stops_at_target(data_dir):
    ad_a = bama_ad("aaa")
    pages = [
        {"ads": [ad_a, bama_ad("aaa")]},  # duplicate of aaa on same page
        {"ads": [bama_ad("bbb")]},
        {"ads": [bama_ad("ccc")]},  # must not be reached (target=2)
    ]
    result = run_collector(pages, data_dir, target_listings=2, max_pages=5)

    assert len(result.records) == 2
    assert [r.source_listing_id for r in result.records] == ["aaa", "bbb"]
    assert result.manifest.records["duplicates_within_run"] == 1
    assert result.manifest.pages["requested"] == 2
    assert result.manifest.status == "success"


def test_collect_stops_after_three_empty_pages(data_dir):
    pages = [
        {"ads": [bama_ad("aaa")]},
        {"ads": []},
        {"ads": []},
        {"ads": []},                # 3rd consecutive empty -> stop
        {"ads": [bama_ad("bbb")]},  # never requested
    ]
    result = run_collector(pages, data_dir, target_listings=10, max_pages=5)
    assert len(result.records) == 1
    assert result.manifest.pages["empty"] == 3
    assert result.manifest.pages["requested"] == 4  # page 0 + 3 empties


def test_collect_persists_raw_records_and_manifest(data_dir):
    pages = [{"ads": [bama_ad("aaa"), bama_ad("bbb")]}]
    run_collector(pages, data_dir, target_listings=10, max_pages=3)

    records_path = data_dir / "raw" / "run-1" / "fake" / "records.jsonl"
    assert records_path.is_file()
    lines = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    assert [entry["source_listing_id"] for entry in lines] == ["aaa", "bbb"]
    assert lines[0]["raw_payload"]["detail"]["code"] == "aaa"

    page_path = data_dir / "raw" / "run-1" / "fake" / "pages" / "page_0000.json"
    assert page_path.is_file()

    manifest_path = data_dir / "manifests" / "run-1.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "run-1"
    assert manifest["platform"] == "fake"
    assert manifest["execution_mode"] == "local"
    assert manifest["quality"]["record_count"] == 2


def test_quality_report_missing_states(data_dir):
    ad_full = bama_ad("full")
    ad_sparse = bama_ad("sparse", description=None, year=None, trim="غیرمنتظره")
    pages = [{"ads": [ad_full, ad_sparse]}]
    result = run_collector(pages, data_dir, target_listings=10, max_pages=1)
    q = {f.field: f for f in result.manifest.quality.fields}

    # Supported and present on both ads -> populated
    assert q["source_listing_id"].missing_state == "populated"
    assert q["source_listing_id"].coverage_pct == 100.0

    # Supported but missing on one ad -> missing_in_ad
    assert q["description"].populated == 1
    assert q["description"].missing_in_ad == 1
    assert q["description"].missing_state == "missing_in_ad"

    # Supported but missing on both -> missing_in_ad (platform supports mileage via detail)
    assert q["mileage_raw"].missing_in_ad == 2

    # NOT in field_support -> not_supported_by_platform regardless of content
    assert q["trim_raw"].supported_by_platform is False
    assert q["trim_raw"].missing_state == "not_supported_by_platform"

    # P0/P1 rollups present
    assert result.manifest.quality.p0_coverage_pct > 0
    assert result.manifest.quality.p1_coverage_pct >= 0
