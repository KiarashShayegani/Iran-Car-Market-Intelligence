"""ICMI API: search, listing detail, market stats, grounded QA.

Built with an app factory so tests inject fakes (offline); production uses
repo-local artifacts. Persian-first API contract; every response keeps
canonical identity + source links per the traceability requirement.

LLM is optional by design: /ask returns a clean 503 with a Persian message
when LLM_BASE_URL is unset — search and stats keep working.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from icmi import __version__
from icmi.analytics.comparables import estimate_market_range
from icmi.analytics.stats import market_stats
from icmi.api.image_proxy import CachedImage, ImageCache, ImageProxyError
from icmi.core.config import load_sources
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


def _load_repo_env() -> None:
    """Load the repo-local .env (same semantics as the dev wrappers: values
    override). Keeps `uv run uvicorn icmi.api.app:app` fully configured
    without the wrapper. UV_* keys are skipped — uv resolves those itself."""
    import os

    from icmi.core.paths import repo_root

    env_path = repo_root() / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        if name.startswith("UV_"):
            continue
        os.environ[name] = value.strip()


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

    sources_cfg = load_sources()
    image_cache = ImageCache(
        allow_hosts=sources_cfg.image_cache.allow_hosts,
        max_cache_mb=sources_cfg.image_cache.max_cache_mb,
        ttl_days=sources_cfg.image_cache.ttl_days,
    )

    state: dict[str, Any] = {
        "search_service": search_service,
        "store": store,
        "qa_client": qa_client,
        "qa_client_resolved": qa_client is not None,
        "image_cache": image_cache,
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

    @app.get("/images/proxy")
    def image_proxy(url: str = Query(min_length=1)) -> Any:
        """Cache-aside fallback for CDN images (direct rendering is default)."""
        from fastapi import Response

        try:
            cached: CachedImage = state["image_cache"].get(url)
        except ImageProxyError as exc:
            status = 400 if exc.kind == "disallowed" else 502
            raise HTTPException(status_code=status, detail=str(exc)) from None
        except Exception as exc:  # noqa: BLE001 — upstream/network failures
            raise HTTPException(status_code=502, detail=f"image fetch failed: {exc}") from None
        return Response(
            content=cached.path.read_bytes(),
            media_type=cached.content_type,
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.get("/latest")
    def latest(limit: int = Query(default=12, ge=1, le=50)) -> dict:
        """Most recently collected listings — homepage inventory surface."""
        df = get_store().query_df(
            """
            SELECT listing_key, source_platform, original_url, title, brand_raw,
                   model_raw, price_amount, year_num, year_calendar, mileage_km,
                   location_raw, image_urls_json, first_seen_at
            FROM listings ORDER BY first_seen_at DESC, listing_key LIMIT ?
            """,
            (limit,),
        )
        return {"listings": [_listing_card(row) for _, row in df.iterrows()]}

    @app.get("/facets")
    def facets() -> dict:
        """Database-driven filter facets: brands with nested models, spec
        values, top locations, platform counts, freshness. Never invents
        options that are not present in the data."""
        store = get_store()
        brands = store.query_df(
            """
            SELECT brand_key, brand_raw, COUNT(*) AS count
            FROM listings WHERE brand_key IS NOT NULL
            GROUP BY brand_key ORDER BY count DESC
            """
        )
        models = store.query_df(
            """
            SELECT brand_key, model_raw, COUNT(*) AS count
            FROM listings WHERE brand_key IS NOT NULL AND model_raw IS NOT NULL
            GROUP BY brand_key, model_raw ORDER BY count DESC
            """
        )
        models_by_brand: dict[str, list] = {}
        for _, r in models.iterrows():
            models_by_brand.setdefault(r["brand_key"], []).append(
                {"label": r["model_raw"], "count": int(r["count"])}
            )
        platforms = store.query_df(
            "SELECT source_platform, COUNT(*) AS count FROM listings GROUP BY source_platform"
        )

        def distinct(column: str) -> list[str]:
            rows = store.query_df(
                f"SELECT DISTINCT {column} AS v FROM listings WHERE {column} IS NOT NULL ORDER BY v"
            )
            return [str(r["v"]) for _, r in rows.iterrows() if r["v"]]

        locations = store.query_df(
            """
            SELECT location_raw, COUNT(*) AS count
            FROM listings WHERE location_raw IS NOT NULL
            GROUP BY location_raw ORDER BY count DESC LIMIT 20
            """
        )
        bounds = store.query_df(
            """
            SELECT MIN(first_seen_at) AS oldest, MAX(first_seen_at) AS newest,
                   SUM(CASE WHEN price_amount IS NOT NULL THEN 1 ELSE 0 END) AS priced,
                   COUNT(*) AS total,
                   MIN(price_amount) AS min_price, MAX(price_amount) AS max_price,
                   MIN(year_num) AS min_year, MAX(year_num) AS max_year
            FROM listings
            """
        ).iloc[0]
        return {
            "brands": [
                {
                    "key": r["brand_key"],
                    "label": r["brand_raw"],
                    "count": int(r["count"]),
                    "models": models_by_brand.get(r["brand_key"], []),
                }
                for _, r in brands.iterrows()
            ],
            "fuel": distinct("fuel_raw"),
            "transmission": distinct("transmission_raw"),
            "body_type": distinct("body_type_raw"),
            "locations": [
                {"label": r["location_raw"], "count": int(r["count"])}
                for _, r in locations.iterrows()
            ],
            "platforms": [
                {"platform": r["source_platform"], "count": int(r["count"])}
                for _, r in platforms.iterrows()
            ],
            "bounds": {
                "min_price": _int_safe(bounds["min_price"]),
                "max_price": _int_safe(bounds["max_price"]),
                "min_year": _int_safe(bounds["min_year"]),
                "max_year": _int_safe(bounds["max_year"]),
            },
            "freshness": {
                "oldest_collection": bounds["oldest"],
                "newest_collection": bounds["newest"],
                "total_listings": int(bounds["total"]),
                "priced_listings": int(bounds["priced"]),
            },
        }

    @app.get("/compare")
    def compare(
        keys: str = Query(min_length=1), limit: int = Query(default=6, ge=1, le=20)
    ) -> dict:
        """Side-by-side comparison of 2-3 listings, each with its own
        retrieval-based market range (auditable, from comparables)."""
        key_list = [k.strip() for k in keys.split(",") if k.strip()]
        if len(key_list) < 2 or len(key_list) > 3:
            raise HTTPException(status_code=400, detail="compare needs 2-3 listing keys")
        store = get_store()
        vehicles = []
        for key in key_list:
            df = store.query_df(
                """
                SELECT listing_key, source_platform, original_url, title, brand_raw,
                       model_raw, trim_raw, year_raw, year_calendar, mileage_raw,
                       mileage_km, price_raw, price_amount, price_currency_raw,
                       location_raw, fuel_raw, transmission_raw, body_type_raw,
                       body_condition_raw, color_raw, image_urls_json, description,
                       first_seen_at
                FROM listings WHERE listing_key = ?
                """,
                (key,),
            )
            if df.empty:
                raise HTTPException(status_code=404, detail=f"not found: {key}")
            row = df.iloc[0]
            estimate = estimate_market_range(
                store,
                brand=row["brand_raw"] or "",
                model=row["model_raw"],
                year=row["year_raw"],
                mileage_km=_int_safe(row["mileage_km"]),
                trim_outliers=False,
            )
            vehicles.append(
                {
                    "listing": _jsonable(row.to_dict()),
                    "image_url": _first_image(row["image_urls_json"]),
                    "market_range": {
                        "count": estimate.count,
                        "median": estimate.median,
                        "p25": estimate.p25,
                        "p75": estimate.p75,
                        "quality": estimate.quality,
                        "quality_note": estimate.quality_note,
                    },
                }
            )
        return {"vehicles": vehicles}

    @app.get("/ops-status")
    def ops_status() -> dict:
        """Operational health: collection runs, DB, index freshness, backups."""
        from icmi.ops.status import build_ops_status

        return build_ops_status(get_store()._conn)

    @app.get("/search")
    def search(
        q: str = Query(default=""),
        top_k: int = Query(default=12, ge=1, le=50),
        offset: int = Query(default=0, ge=0),
        sort: str = Query(default="relevance"),
        platforms: str | None = None,
        brands: str | None = None,
        models: str | None = None,
        fuel: str | None = None,
        transmission: str | None = None,
        body_type: str | None = None,
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
            models=[m.strip() for m in models.split(",") if m.strip()] if models else [],
            fuel=[f.strip() for f in fuel.split(",") if f.strip()] if fuel else [],
            transmission=(
                [t.strip() for t in transmission.split(",") if t.strip()]
                if transmission
                else []
            ),
            body_type=(
                [b.strip() for b in body_type.split(",") if b.strip()]
                if body_type
                else []
            ),
            price_min=price_min,
            price_max=price_max,
            year_min=year_min,
            year_max=year_max,
            year_calendar=year_calendar,
            mileage_max_km=mileage_max_km,
            location_contains=location_contains,
        )
        if sort not in ("relevance", "price_asc", "price_desc", "year_desc", "newest"):
            raise HTTPException(status_code=400, detail=f"unknown sort: {sort}")
        service = get_search_service()
        response = service.search(q, filters, top_k=top_k, offset=offset, sort=sort)
        meta_by_key = service.metadata.set_index("listing_key")
        results = []
        for r in response.results:
            meta = meta_by_key.loc[r.listing_key] if r.listing_key in meta_by_key.index else None
            results.append(
                {
                    "listing_key": r.listing_key,
                    "source_platform": r.source_platform,
                    "score": r.score,
                    "excerpt": r.text,
                    "original_url": r.original_url,
                    "raw_payload_ref_json": r.raw_payload_ref_json,
                    "chunk_index": r.chunk_index,
                    "chunk_count": r.chunk_count,
                    "brand_raw": None if meta is None else _nan_safe(meta["brand_raw"]),
                    "model_raw": None if meta is None else _nan_safe(meta["model_raw"]),
                    "trim_raw": None if meta is None else _nan_safe(meta["trim_raw"]),
                    "fuel_raw": None if meta is None else _nan_safe(meta["fuel_raw"]),
                    "transmission_raw": (
                        None if meta is None else _nan_safe(meta["transmission_raw"])
                    ),
                    "body_type_raw": None if meta is None else _nan_safe(meta["body_type_raw"]),
                    "year_num": None if meta is None else _int_safe(meta["year_num"]),
                    "year_calendar": None if meta is None else _nan_safe(meta["year_calendar"]),
                    "price_amount": None if meta is None else _int_safe(meta["price_amount"]),
                    "mileage_km": None if meta is None else _int_safe(meta["mileage_km"]),
                    "location_raw": None if meta is None else _nan_safe(meta["location_raw"]),
                }
            )
            if meta is not None and meta.get("image_urls_json"):
                results[-1]["image_url"] = _first_image(meta["image_urls_json"])
        mode = (
            "browse" if not response.query.strip()
            else "hybrid" if not response.filters.is_empty()
            else "semantic"
        )
        return {
            "query": response.query,
            "mode": mode,
            "total_corpus": response.total_corpus,
            "candidates_after_filters": response.candidates_after_filters,
            "total_matched": response.total_matched,
            "offset": response.offset,
            "limit": response.limit,
            "sort": response.sort,
            "results": results,
        }

    @app.get("/estimate")
    def estimate(
        brand: str = Query(min_length=1),
        model: str | None = None,
        year: str | None = None,
        mileage_km: int | None = None,
        trim_outliers: bool = Query(default=False),
    ) -> dict:
        """Observed market range from comparable ads (retrieval-based, no ML)."""
        result = estimate_market_range(
            get_store(),
            brand=brand,
            model=model,
            year=year,
            mileage_km=mileage_km,
            trim_outliers=trim_outliers,
        )
        return {
            **result.__dict__,
            "comparables": [c.__dict__ for c in result.comparables],
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
        observations = store.query_df(
            """
            SELECT observed_at, price_raw, price_amount, price_currency_raw, run_id
            FROM price_observations WHERE listing_key = ? ORDER BY id
            """,
            (key,),
        )
        record["price_history"] = observations.to_dict(orient="records")
        record["image_urls"] = (
            json.loads(record["image_urls_json"])
            if record.get("image_urls_json")
            else []
        )
        return _jsonable(record)

    @app.get("/listings/{platform}/{listing_id}/similar")
    def similar(platform: str, listing_id: str, limit: int = Query(default=6, ge=1, le=20)) -> dict:
        """Retrieval-based similar ads: reuse the listing's own index vector —
        no re-encoding, same embeddings, traceable results."""
        key = f"{platform}:{listing_id}"
        service = get_search_service()
        matches = service.metadata.index[service.metadata["listing_key"] == key]
        if len(matches) == 0:
            raise HTTPException(status_code=404, detail="Listing not in index")
        position = int(service.metadata.index.get_loc(matches[0]))
        vector = service.vectors[position].tolist()
        from icmi.retrieval.search import rank_subset

        ranked = rank_subset(
            vector, service.vectors, service.metadata.index.to_numpy(), top_k=limit + 1
        )
        results = []
        for pos, score in ranked:
            row = service.metadata.iloc[pos]
            if row["listing_key"] == key:
                continue
            results.append(
                {
                    "listing_key": row["listing_key"],
                    "source_platform": row["source_platform"],
                    "score": round(score, 4),
                    "text": row["text"],
                    "original_url": row["original_url"],
                    "brand_raw": _nan_safe(row["brand_raw"]),
                    "model_raw": _nan_safe(row["model_raw"]),
                    "year_num": _int_safe(row["year_num"]),
                    "price_amount": _int_safe(row["price_amount"]),
                    "mileage_km": _int_safe(row["mileage_km"]),
                    "image_url": (
                        _first_image(row["image_urls_json"])
                        if "image_urls_json" in row
                        else None
                    ),
                }
            )
            if len(results) >= limit:
                break
        return {"listing_key": key, "results": results}

    @app.get("/stats")
    def stats() -> dict:
        result = market_stats(get_store())
        return {
            "meta": result["meta"],
            "overview": result["overview"].to_dict(orient="records"),
            "by_brand": result["by_brand"].to_dict(orient="records"),
            "by_year": result["by_year"].to_dict(orient="records"),
            "by_mileage": result["by_mileage"].to_dict(orient="records"),
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


def _first_image(image_urls_json) -> str | None:
    if not image_urls_json:
        return None
    try:
        urls = json.loads(image_urls_json)
        return urls[0] if isinstance(urls, list) and urls else None
    except (ValueError, TypeError):
        return None


def _listing_card(row) -> dict:
    return {
        "listing_key": row["listing_key"],
        "source_platform": row["source_platform"],
        "original_url": row["original_url"],
        "title": row["title"],
        "brand_raw": row["brand_raw"],
        "model_raw": row["model_raw"],
        "price_amount": _int_safe(row["price_amount"]),
        "year_num": _int_safe(row["year_num"]),
        "year_calendar": row["year_calendar"],
        "mileage_km": _int_safe(row["mileage_km"]),
        "location_raw": row["location_raw"],
        "image_url": _first_image(row["image_urls_json"]),
        "first_seen_at": row["first_seen_at"],
    }


def _nan_safe(value) -> str | None:
    import math

    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def _int_safe(value) -> int | None:
    import math

    if value is None:
        return None
    if isinstance(value, float):
        return None if math.isnan(value) else int(value)
    return int(value)


def _calendar_of(year_raw) -> str | None:
    from icmi.pipeline.normalize import classify_year_calendar

    return classify_year_calendar(normalize_year(year_raw))


def _jsonable(record: dict) -> dict:
    """JSON-safe conversion — lists/dicts pass through recursively (parsed
    price_history and image_urls must arrive as arrays, not string blobs)."""

    def conv(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            return [conv(v) for v in value]
        if isinstance(value, dict):
            return {k: conv(v) for k, v in value.items()}
        return str(value)

    return {key: conv(value) for key, value in record.items()}


_load_repo_env()
# Module-level ASGI app: enables the standard `uv run uvicorn
# icmi.api.app:app --reload --port 8000` workflow. Services resolve lazily
# on first request; tests inject fakes via create_app(...) instead.
app = create_app()
