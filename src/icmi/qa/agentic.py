"""Controlled agentic question answering (GUI evolution, 2026-11-17).

Two-step, budget-bounded workflow — agentic flexibility without autonomy:

1. **Intent step** (one LLM call): parse the user's question into a strict
   JSON plan {refined_query, brand, model, year_min, year_max, price_min,
   price_max, city, intent}. Schema-validated; any failure falls back to a
   plain semantic search with the original question. The plan never reaches
   the user verbatim — it only steers retrieval.
2. **Retrieval**: semantic search with the plan's structured filters
   (SearchFilters). One refinement allowed when the first pass is sparse
   (<3 results): drop the filters, widen the query. Total budget: 2 search
   calls.
3. **Answer step**: the existing hard grounding contract (evidence-only,
   [n] citations, insufficient-evidence marker) — unchanged from
   qa/grounded.py.

The LLM never touches SQL, never invents listings, and every cited listing
maps back to a real listing_key. The response adds `mode: "agentic"`,
`filters_used`, and `steps` (trace) for the frontend's methodology
disclosure. Prompt-injection resistance: untrusted ad text is confined to
delimited evidence blocks in the final step only — the intent step sees the
user question alone.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from icmi.retrieval.normalization import normalize_for_key
from icmi.qa.grounded import (
    INSUFFICIENT_MARKER,
    GroundedAnswer,
    LLMError,
)
from icmi.retrieval.search import SearchFilters, SearchResponse, SearchService

MAX_SEARCH_CALLS = 2
SPARSE_RESULTS = 3

INTENT_SYSTEM = """\
تو تحلیل‌گر پرسش‌های خودرویی هستی. پرسش کاربر را به یک برنامهٔ جستجو تبدیل کن
و «فقط» یک JSON معتبر برگردان، بدون هیچ متن اضافه:
{
  "refined_query": "عبارت جستجوی معنایی (فارسی، خلاصه و مؤثر)",
  "brand": "برند فارسی یا null",
  "model": "مدل یا null",
  "year_min": عدد یا null,
  "year_max": عدد یا null,
  "price_min": عدد تومان یا null,
  "price_max": عدد تومان یا null,
  "city": "شهر یا null",
  "intent": "search | compare | value | trend"
}
قواعد: سال‌ها شمسی هستند. قیمت‌ها به تومان. هیچ فیلدی را از خودت نساز — فقط
از پرسش کاربر استخراج کن. اگر چیزی در پرسش نیست، null بگذار."""


@dataclass
class IntentPlan:
    refined_query: str
    brand: str | None = None
    model: str | None = None
    year_min: int | None = None
    year_max: int | None = None
    price_min: int | None = None
    price_max: int | None = None
    city: str | None = None
    intent: str = "search"
    parse_ok: bool = True


@dataclass
class AgenticTrace:
    steps: list[dict] = field(default_factory=list)
    filters_used: dict = field(default_factory=dict)
    search_calls: int = 0


def parse_intent_plan(raw: str, question: str) -> IntentPlan:
    """Parse the intent JSON; any failure → safe fallback plan."""
    fallback = IntentPlan(refined_query=question, parse_ok=False)
    try:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return fallback
        d = json.loads(m.group(0))
    except (ValueError, TypeError):
        return fallback
    if not isinstance(d, dict):
        return fallback

    def _num(v) -> int | None:
        try:
            return int(v) if v is not None else None
        except (ValueError, TypeError):
            return None

    refined = str(d.get("refined_query") or question).strip()
    return IntentPlan(
        refined_query=refined[:200] or question,
        brand=(str(d["brand"]).strip() or None) if d.get("brand") else None,
        model=(str(d["model"]).strip() or None) if d.get("model") else None,
        year_min=_num(d.get("year_min")),
        year_max=_num(d.get("year_max")),
        price_min=_num(d.get("price_min")),
        price_max=_num(d.get("price_max")),
        city=(str(d["city"]).strip() or None) if d.get("city") else None,
        intent=str(d.get("intent") or "search"),
        parse_ok=True,
    )


def plan_to_filters(plan: IntentPlan, platforms: list[str]) -> SearchFilters:
    """Map the intent plan to SearchFilters (normalized-key brand/model)."""
    return SearchFilters(
        platforms=platforms,
        brands=[normalize_for_key(plan.brand)] if plan.brand else [],
        models=[normalize_for_key(plan.model)] if plan.model else [],
        year_min=plan.year_min,
        year_max=plan.year_max,
        price_min=plan.price_min,
        price_max=plan.price_max,
        location_contains=plan.city,
    )


def answer_agentic(
    question: str,
    service: SearchService,
    client,
    platforms: list[str] | None = None,
    top_k: int = 6,
) -> tuple[GroundedAnswer, AgenticTrace]:
    """Controlled agentic flow: intent → retrieval (≤2 calls) → grounded answer.

    Returns (GroundedAnswer, AgenticTrace). Raises LLMError when the endpoint
    fails; retrieval-only fallbacks stay inside this function.
    """
    if client is None:
        raise LLMError("LLM is not configured (set LLM_BASE_URL)")
    trace = AgenticTrace()
    platforms = platforms or []

    # --- step 1: intent plan (user question only — no untrusted ad text) ---
    steps_entry = {"step": "intent", "ok": False}
    try:
        raw = client.complete(INTENT_SYSTEM, f"پرسش کاربر: {question}")
        plan = parse_intent_plan(raw, question)
        steps_entry.update(ok=plan.parse_ok, intent=plan.intent,
                           refined_query=plan.refined_query)
    except LLMError:
        plan = IntentPlan(refined_query=question, parse_ok=False)
        steps_entry.update(ok=False, note="intent call failed — plain search")
    trace.steps.append(steps_entry)

    # --- step 2: retrieval with plan filters (call 1) ---
    filters = plan_to_filters(plan, platforms)
    trace.filters_used = {
        k: v for k, v in {
            "brand": plan.brand, "model": plan.model,
            "year_min": plan.year_min, "year_max": plan.year_max,
            "price_min": plan.price_min, "price_max": plan.price_max,
            "city": plan.city, "platforms": platforms or None,
        }.items() if v
    }
    response: SearchResponse | None = None
    try:
        response = service.search(
            plan.refined_query, filters=filters, top_k=top_k, sort="relevance"
        )
        trace.search_calls += 1
        trace.steps.append({
            "step": "search", "call": 1, "results": len(response.results),
            "mode": response.mode,
        })
    except Exception as exc:  # noqa: BLE001 — retrieval failure → fallback below
        trace.steps.append({"step": "search", "call": 1, "error": str(exc)})

    # --- step 2b: refinement when sparse (call 2: drop filters) ---
    if (response is None or len(response.results) < SPARSE_RESULTS) and filters != SearchFilters():
        try:
            response = service.search(question, filters=None, top_k=top_k,
                                      sort="relevance")
            trace.search_calls += 1
            trace.steps.append({
                "step": "search_refine", "call": 2,
                "results": len(response.results),
                "note": "filters dropped for wider recall",
            })
        except Exception as exc:  # noqa: BLE001
            trace.steps.append({"step": "search_refine", "error": str(exc)})

    if response is None:
        raise LLMError("retrieval failed — no search results available")

    # --- step 3: grounded answer (unchanged hard contract) ---
    from icmi.qa.grounded import answer_question, build_evidence

    evidence = build_evidence(response)
    answer = answer_question(question, response, client)
    if INSUFFICIENT_MARKER in answer.answer:
        trace.steps.append({"step": "answer", "insufficient_evidence": True})
    else:
        trace.steps.append({
            "step": "answer", "citations": len(answer.citations),
            "evidence": len(evidence),
        })
    answer.mode = "agentic"
    answer.listing_keys = [ev.listing_key for ev in evidence]
    trace.filters_used = trace.filters_used or {"query": plan.refined_query}
    answer.filters_used = dict(trace.filters_used)
    answer.steps = trace.steps
    return answer, trace
