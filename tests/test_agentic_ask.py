"""Controlled agentic /ask flow: intent parsing, filter steering, budget,
grounding contract, and structured payload — with a scripted mock LLM."""

import json

import pandas as pd
import pytest

from icmi.qa.agentic import MAX_SEARCH_CALLS, answer_agentic, parse_intent_plan
from icmi.qa.grounded import LLMError


class MockClient:
    """Scripted LLM: intent call returns a plan; answer call returns grounded text."""

    def __init__(self, plan_json: str, answer: str):
        self.plan_json = plan_json
        self.answer = answer
        self.model = "mock-model"
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system[:40], user[:80]))
        if "برنامهٔ جستجو" in system:
            return self.plan_json
        return self.answer



class FakeService:
    def __init__(self, results_per_call: list[int]):
        self.results_per_call = results_per_call
        self.calls = 0
        # metadata frame with card fields (mirrors the real service contract)
        rows = []
        for i in range(20):
            rows.append({
                "listing_key": f"bama:{100 + i}",
                "image_urls_json": f'["https://img.example/{100 + i}.jpg"]',
                "price_amount": 700_000_000 + i * 10_000_000,
                "year_num": 1395 + (i % 5),
                "year_calendar": "shamsi",
                "mileage_km": 50_000 + i * 1_000,
                "location_raw": "تهران",
                "brand_raw": "پژو",
                "model_raw": "206 تیپ 5",
            })
        self.metadata = pd.DataFrame(rows)

    def search(self, query, filters=None, top_k=6, offset=0, sort="relevance"):
        n = self.results_per_call[min(self.calls, len(self.results_per_call) - 1)]
        self.calls += 1

        class R:
            results = [
                type(
                    "Row",
                    (),
                    {
                        "listing_key": f"bama:{100 + i}",
                        "source_platform": "bama",
                        "score": 0.9,
                        "text": f"آگهی نمونهٔ {i}",
                        "original_url": "https://bama.ir/car/detail-x",
                        "raw_payload_ref_json": "{}",
                        "chunk_index": 0,
                        "chunk_count": 1,
                        "extraction_version": "icmi-v3-extraction-1",
                    },
                )()
                for i in range(n)
            ]
            candidates_after_filters = n
            mode = "semantic"

            def __init__(self, n):
                self.results = [
                    type(
                        "Row",
                        (),
                        {
                            "listing_key": f"bama:{100 + i}",
                            "source_platform": "bama",
                            "score": 0.9,
                            "text": f"آگهی نمونهٔ {i}",
                            "original_url": "https://bama.ir/car/detail-x",
                            "raw_payload_ref_json": "{}",
                            "chunk_index": 0,
                            "chunk_count": 1,
                            "extraction_version": "icmi-v3-extraction-1",
                        },
                    )()
                    for i in range(n)
                ]

        return R(n)


def test_intent_parse_and_filter_mapping():
    plan = parse_intent_plan(
        '{"refined_query": "پژو 206", "brand": "پژو", "year_max": 1400, "intent": "value"}',
        "q",
    )
    assert plan.parse_ok and plan.brand == "پژو" and plan.intent == "value"
    filters = __import__(
        "icmi.qa.agentic", fromlist=["plan_to_filters"]
    ).plan_to_filters(plan, [])
    assert filters.brands == ["پژو"] and filters.year_max == 1400


def test_agentic_happy_path_uses_filters_and_returns_structured_payload():
    plan = json.dumps(
        {
            "refined_query": "پژو 206 اتوماتیک",
            "brand": "پژو",
            "price_max": 1000000000,
            "intent": "search",
        }
    )
    client = MockClient(plan, "پاسخ آزمایشی [1]")
    service = FakeService([6])  # first call rich → no refinement
    answer, trace = answer_agentic("بهترین 206 زیر یک میلیارد؟", service, client)

    assert service.calls == 1  # rich first call → within budget
    assert trace.search_calls == 1
    assert trace.filters_used.get("brand") == "پژو"
    assert answer.mode == "agentic"
    # curated semantics: listing_keys = the LLM's cited picks only ([1]),
    # not all evidence — and evidence is enriched with card fields
    assert answer.listing_keys == ["bama:100"]
    assert answer.grounded and not answer.insufficient_evidence
    assert answer.citations[0].image_url == "https://img.example/100.jpg"
    assert answer.citations[0].price_amount == 700_000_000
    assert answer.citations[0].year_num == 1395
    assert answer.citations[0].location_raw == "تهران"


def test_agentic_sparse_triggers_one_refinement_within_budget():
    plan = json.dumps({"refined_query": "پراید", "brand": "پراید", "intent": "search"})
    client = MockClient(plan, "پاسخ [1]")
    service = FakeService([2, 6])  # sparse with filters → refine without them
    answer, trace = answer_agentic("پراید؟", service, client)
    assert service.calls == 2
    assert trace.search_calls <= MAX_SEARCH_CALLS
    # citations only (the mock answer cites [1]) — curated picks
    assert len(answer.listing_keys) == 1


def test_agentic_intent_failure_falls_back_to_plain_search():
    client = MockClient("Absolutely not JSON", "پاسخ [1]")
    service = FakeService([5])
    answer, trace = answer_agentic("پراید؟", service, client)
    assert any(s.get("step") == "intent" and not s.get("ok") for s in trace.steps)
    # the mock answer cites only [1] — citations map to cited picks
    assert len(answer.citations) == 1
    assert answer.citations[0].image_url is not None  # enrichment applied


def test_agentic_no_client_raises():
    with pytest.raises(LLMError):
        answer_agentic("q", FakeService([1]), None)


def test_models_filter_prefix_matching():
    """Regression (2026-11-18): the agentic path extracts short model
    references ('206') from user questions — the corpus stores full trim
    variants ('206 sd v20'). Prefix matching must bridge them."""
    import pandas as pd

    from icmi.retrieval.search import SearchFilters, _apply_filters

    frame = pd.DataFrame({
        "listing_key": ["a", "b", "c"],
        "model_raw": ["206 sd v20", "206 (تیپ2)", "پارس"],
        "brand_key": ["پژو", "پژو", "پژو"],
        "price_amount": [900_000_000, 800_000_000, 1_500_000_000],
    })
    filters = SearchFilters(models=["206"], price_max=1_000_000_000)
    out = _apply_filters(frame, filters)
    assert set(out["listing_key"]) == {"a", "b"}
