"""Tests for the grounded-QA module and the API (fully offline; fake LLM)."""

import json

import pytest
from fastapi.testclient import TestClient

from icmi.api.app import create_app
from icmi.core.records import CanonicalRecord, RawPayloadRef, utc_now
from icmi.pipeline.store import CanonicalStore
from icmi.qa.grounded import (
    LLMError,
    answer_question,
    format_user_prompt,
    parse_citations,
)
from icmi.retrieval.index import build_index_metadata, save_index
from icmi.retrieval.search import SearchService
from tests.fakes import HashingEncoder


class FakeLLM:
    """Deterministic fake OpenAI-compatible client."""

    model = "fake-model"

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.last_prompts: tuple[str, str] | None = None

    def complete(self, system: str, user: str) -> str:
        self.last_prompts = (system, user)
        return self.answer


def make_listing(store, platform, lid, **f):
    record = CanonicalRecord(
        source_platform=platform,
        source_listing_id=lid,
        original_url=f.get("original_url", f"https://example.com/{lid}"),
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
def env(tmp_path, monkeypatch):
    """Fixture corpus + index + components; returns a namespace-like dict."""
    monkeypatch.setenv("ICMI_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    store = CanonicalStore()
    try:
        make_listing(store, "bama", "t1", brand_raw="تویوتا", model_raw="کریسida",
                     title="کریسida 2011", year_raw=2011,
                     description="تویوتا کریسida اتوماتیک، بیمه کامل.")
        make_listing(store, "bama", "p1")
        chunks = build_index_metadata(store)
    finally:
        store.close()
    encoder = HashingEncoder()

    import numpy as np

    vectors = np.asarray(encoder.encode_passages([c.normalized_text for c in chunks]))
    save_index(vectors, chunks, encoder.model_id, out_dir=tmp_path / "idx")
    from icmi.retrieval.index import load_index

    vectors2, metadata, _ = load_index(tmp_path / "idx")
    service = SearchService(vectors2, metadata, encoder)
    store2 = CanonicalStore()
    app = create_app(search_service=service, store=store2)
    return {
        "client": TestClient(app),
        "store": store2,
        "service": service,
        "index_dir": tmp_path / "idx",
    }


@pytest.fixture()
def client(env):
    return env["client"]


# --- grounded QA unit tests ----------------------------------------------------


def test_format_user_prompt_delimits_evidence():
    from icmi.qa.grounded import build_evidence
    from icmi.retrieval.search import SearchFilters, SearchResponse, SearchResult

    response = SearchResponse(
        query="تست",
        filters=SearchFilters(),
        total_corpus=1,
        candidates_after_filters=1,
        results=[
            SearchResult(
                listing_key="bama:t1", source_platform="bama", score=0.9,
                text="تویوتا کریسida اتوماتیک، بیمه کامل.",
                original_url="https://example.com/t1",
                raw_payload_ref_json='{"key":"t1"}',
                chunk_index=0, chunk_count=1, extraction_version="test-v1",
            )
        ],
    )
    evidence = build_evidence(response)
    prompt = format_user_prompt("بیمه ماشین چطوره؟", evidence)
    assert "شواهد [1]" in prompt
    assert "https://example.com/t1" in prompt
    assert "بیمه کامل" in prompt  # source content verbatim


def test_parse_citations_maps_and_falls_back():
    from icmi.qa.grounded import Evidence

    evidence = [
        Evidence(1, "bama:t1", "bama", "text1", "u1", 0.9),
        Evidence(2, "bama:p1", "bama", "text2", "u2", 0.8),
    ]
    cited = parse_citations("قیمت طبق [2] مناسب است.", evidence)
    assert [e.index for e in cited] == [2]
    # no citations → all evidence (explicit fallback)
    assert len(parse_citations("بدون ارجاع", evidence)) == 2


def test_answer_question_with_fake_llm():
    from icmi.retrieval.search import SearchFilters, SearchResponse, SearchResult

    response = SearchResponse(
        query="q", filters=SearchFilters(), total_corpus=1, candidates_after_filters=1,
        results=[
            SearchResult(
                listing_key="bama:t1", source_platform="bama", score=0.9,
                text="سال 2011", original_url="https://example.com/t1",
                raw_payload_ref_json="{}", chunk_index=0, chunk_count=1,
                extraction_version="v",
            )
        ],
    )
    llm = FakeLLM("این ماشین سال [1] 2011 است.")
    answer = answer_question("سال ماشین؟", response, llm)
    assert answer.answer == "این ماشین سال [1] 2011 است."
    assert answer.grounded is True
    assert [c.listing_key for c in answer.citations] == ["bama:t1"]
    system, user = llm.last_prompts
    assert "فقط بر اساس" in system  # grounding contract in the Persian prompt


def test_answer_question_requires_client():
    from icmi.retrieval.search import SearchFilters, SearchResponse

    empty = SearchResponse("q", SearchFilters(), 0, 0, [])
    with pytest.raises(LLMError, match="not configured"):
        answer_question("q", empty, None)


# --- API tests -------------------------------------------------------------------


def test_health_reports_corpus_and_llm_state(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["listings"] == 2
    assert body["llm_configured"] is False


def test_search_endpoint_returns_traceable_results(client):
    body = client.get("/search", params={"q": "تویوتا اتوماتیک"}).json()
    assert body["mode"] == "semantic"
    assert body["results"], "Toyota ad exists — must be found"
    top = body["results"][0]
    assert top["listing_key"] == "bama:t1"
    assert top["original_url"].startswith("https://")
    assert json.loads(top["raw_payload_ref_json"])["key"] == "t1"


def test_search_endpoint_hybrid_filters_restrict(client):
    body = client.get(
        "/search",
        params={"q": "تویوتا لوکس", "brands": "سایپا", "price_max": 600_000_000},
    ).json()
    # no Saipa in corpus → honest zero, never loosened
    assert body["candidates_after_filters"] == 0
    assert body["results"] == []


def test_listing_endpoint_and_404(client):
    body = client.get("/listings/bama/t1").json()
    assert body["listing_key"] == "bama:t1"
    assert body["year_calendar"] == "gregorian"
    assert client.get("/listings/bama/nope").status_code == 404


def test_stats_endpoint_groups_separate_calendars(client):
    body = client.get("/stats").json()
    assert body["meta"]["price_normalization_version"] == "price-norm-1"
    calendars = {row["year_calendar"] for row in body["by_year"]}
    assert calendars  # grouped, not blended


def test_ask_returns_503_without_llm_config(client):
    response = client.post("/ask", json={"question": "سال ماشین چیه؟"})
    assert response.status_code == 503
    assert "LLM" in response.json()["detail"]


def test_ask_with_injected_fake_llm_is_grounded(env):
    fake = FakeLLM("کریسida سال [1] 2011 با بیمه کامل.")
    app = create_app(search_service=env["service"], store=env["store"], qa_client=fake)
    body = TestClient(app).post("/ask", json={"question": "سال ماشین؟"}).json()
    assert body["grounded"] is True
    assert body["model_id"] == "fake-model"
    # citation [1] maps to the FIRST evidence item — whichever the (fake)
    # encoder ranked there; the mapping mechanism is what we're testing.
    assert body["citations"][0]["index"] == 1
    assert body["citations"][0]["listing_key"] in {"bama:t1", "bama:p1"}
    assert body["evidence_count"] == 2
    assert "فقط بر اساس" in fake.last_prompts[0]


def test_ask_llm_failure_is_502(env):
    class BrokenLLM:
        model = "broken"

        def complete(self, system, user):
            raise LLMError("endpoint unreachable")

    app = create_app(search_service=env["service"], store=env["store"], qa_client=BrokenLLM())
    response = TestClient(app).post("/ask", json={"question": "سال ماشین؟"})
    assert response.status_code == 502
    assert "unreachable" in response.json()["detail"]
