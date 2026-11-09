"""ICMI API: search, listing detail, market stats, grounded QA.

Built with an app factory so tests inject fakes (offline); production uses
repo-local artifacts. Persian-first API contract; every response keeps
canonical identity + source links per the traceability requirement.

LLM is optional by design: /ask returns a clean 503 with a Persian message
when LLM_BASE_URL is unset — search and stats keep working.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from icmi import __version__
from icmi.analytics.stats import market_stats
from icmi.core.paths import index_dir
from icmi.pipeline.normalize import normalize_year
from icmi.pipeline.store import CanonicalStore
from icmi.qa.grounded import LLMError, answer_question, client_from_env
from icmi.retrieval.encoder import DEFAULT_MODEL_ID, get_encoder
from icmi.retrieval.index import INDEX_FORMAT_VERSION, load_index
from icmi.retrieval.search import SearchFilters, SearchService


class AskRequest(BaseModel):
    question: str = Field(min_length=2)
    platforms: list[str] = Field(default_factory=list)
    price_max: int | None = None
    price_min: int | None = None
    year_min: int | None = None
    year_max: int | None = None
    year_calendar: str | None = None
    mileage_max_km: int | None = None
    top_k: int = Field(default=6, ge=1, le=20)


def _to_filters(request_or_params) -> SearchFilters:
    if isinstance(request_or_params, dict):
        return SearchFilters(**{k: v for k, v in request_or_params.items() if v})
    return SearchFilters()


def create_app(
    search_service: SearchService | None = None,
    store: CanonicalStore | None = None,
    qa_client: Any | None = None,
) -> FastAPI:
    """App factory. Injected services win (tests); otherwise artifacts are
    loaded from the repo data layer (index must be built first)."""
    app = FastAPI(
        title="ICMI API",
        description="Iranian Car Market Intelligence — search, stats, grounded QA",
        version=__version__,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    state: dict[str, Any] = {
        "search_service": search_service,
        "store": store,
        "qa_client": qa_client,
        "qa_client_resolved": qa_client is not None,
    }

    def get_store() -> CanonicalStore:
        if state["store"] is None:
            state["store"] = CanonicalStore()
        return state["store"]

    def get_search_service() -> SearchService:
        if state["search_service"] is None:
            vectors, metadata, info = load_index(index_dir())
            if info.get("format_version") != INDEX_FORMAT_VERSION:
                raise HTTPException(
                    status_code=503,
                    detail="Index format outdated — rebuild with scripts/build_index.py",
                )
            state["search_service"] = SearchService(
                vectors, metadata, get_encoder(DEFAULT_MODEL_ID)
            )
        return state["search_service"]

    def get_qa_client():
        # resolve lazily so .env changes apply without restart in dev
        if not state["qa_client_resolved"]:
            state["qa_client"] = client_from_env()
            state["qa_client_resolved"] = True
        return state["qa_client"]

    @app.get("/health")
    def health() -> dict:
        store = get_store()
        corpus = store.listing_count()
        return {
            "status": "ok",
            "version": __version__,
            "listings": corpus,
            "llm_configured": get_qa_client() is not None,
        }

    @app.get("/search")
    def search(
        q: str = Query(min_length=1),
        top_k: int = Query(default=10, ge=1, le=50),
        platforms: str | None = None,
        brands: str | None = None,
        price_min: int | None = None,
        price_max: int | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        year_calendar: str | None = None,
        mileage_max_km: int | None = None,
        location_contains: str | None = None,
    ) -> dict:
        filters = SearchFilters(
            platforms=[p.strip() for p in platforms.split(",") if p.strip()] if platforms else [],
            brands=[b.strip() for b in brands.split(",") if b.strip()] if brands else [],
            price_min=price_min,
            price_max=price_max,
            year_min=year_min,
            year_max=year_max,
            year_calendar=year_calendar,
            mileage_max_km=mileage_max_km,
            location_contains=location_contains,
        )
        response = get_search_service().search(q, filters, top_k=top_k)
        return {
            "query": response.query,
            "mode": "hybrid" if not response.filters.is_empty() else "semantic",
            "total_corpus": response.total_corpus,
            "candidates_after_filters": response.candidates_after_filters,
            "results": [
                {
                    "listing_key": r.listing_key,
                    "source_platform": r.source_platform,
                    "score": r.score,
                    "excerpt": r.text,
                    "original_url": r.original_url,
                    "raw_payload_ref_json": r.raw_payload_ref_json,
                    "chunk_index": r.chunk_index,
                    "chunk_count": r.chunk_count,
                }
                for r in response.results
            ],
        }

    @app.get("/listings/{platform}/{listing_id}")
    def listing(platform: str, listing_id: str) -> dict:
        store = get_store()
        key = f"{platform}:{listing_id}"
        df = store.query_df("SELECT * FROM listings WHERE listing_key = ?", (key,))
        if df.empty:
            raise HTTPException(status_code=404, detail="Listing not found")
        record = df.iloc[0].to_dict()
        # keep years as delivered (may be Shamsi or Gregorian; raw preserved)
        record["year_calendar"] = (
            None if record.get("year_raw") is None
            else _calendar_of(record.get("year_raw"))
        )
        return _jsonable(record)

    @app.get("/stats")
    def stats() -> dict:
        result = market_stats(get_store())
        return {
            "meta": result["meta"],
            "overview": result["overview"].to_dict(orient="records"),
            "by_brand": result["by_brand"].to_dict(orient="records"),
            "by_year": result["by_year"].to_dict(orient="records"),
        }

    @app.post("/ask")
    def ask(request: AskRequest) -> dict:
        client = get_qa_client()
        if client is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "LLM پیکربندی نشده است — متغیرهای LLM_BASE_URL و LLM_MODEL را "
                    "در فایل .env تنظیم کنید. جستجو بدون LLM کار می‌کند."
                ),
            )
        filters = SearchFilters(
            platforms=request.platforms,
            price_min=request.price_min,
            price_max=request.price_max,
            year_min=request.year_min,
            year_max=request.year_max,
            year_calendar=request.year_calendar,
            mileage_max_km=request.mileage_max_km,
        )
        response = get_search_service().search(request.question, filters, top_k=request.top_k)
        try:
            answer = answer_question(request.question, response, client)
        except LLMError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from None
        return {
            "question": answer.question,
            "answer": answer.answer,
            "grounded": answer.grounded,
            "insufficient_evidence": answer.insufficient_evidence,
            "model_id": answer.model_id,
            "evidence_count": answer.evidence_count,
            "candidates_after_filters": answer.candidates_after_filters,
            "citations": [
                {
                    "index": c.index,
                    "listing_key": c.listing_key,
                    "source_platform": c.source_platform,
                    "original_url": c.original_url,
                    "score": c.score,
                }
                for c in answer.citations
            ],
        }

    return app


def _calendar_of(year_raw) -> str | None:
    from icmi.pipeline.normalize import classify_year_calendar

    return classify_year_calendar(normalize_year(year_raw))


def _jsonable(record: dict) -> dict:
    out = {}
    for key, value in record.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            out[key] = value
        else:
            out[key] = str(value)
    return out
