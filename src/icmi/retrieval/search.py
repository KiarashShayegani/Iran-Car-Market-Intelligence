"""Search service: semantic and hybrid search over the vector index.

Contract (owner requirements, F3c):
- Filters are GENUINELY restrictive: they select a subset of the whole corpus
  metadata first; ranking runs only inside that subset. Zero matches return an
  honest empty list — filters are never loosened to fill results.
- Results are explainable: similarity score, matching excerpt (the indexed
  document text, containing the source description verbatim), source platform,
  and the original advertisement link.
- Traceability survives retrieval: listing_key + raw_payload_ref on every hit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from icmi.retrieval.normalization import normalize_for_key


class SearchFilters(BaseModel):
    """Structured filters, applied pre-ranking over the full corpus metadata.

    Restrictive semantics: fields with no value (e.g. a listing without a
    parsed price) are EXCLUDED whenever that field's filter is active —
    matching nothing beats pretending.

    Calendar semantics (learned live in F3c eval): year_min/year_max are
    NUMERIC ranges on the parsed year — Iranian listings mix Shamsi and
    Gregorian calendars, so an unpinned range admits both (a "1997" Gregorian
    passes year_min=1398). Intent like «مدل ۹۸» is Shamsi: pin
    year_calendar explicitly when the query implies a calendar.
    """

    platforms: list[str] = Field(default_factory=list)
    brands: list[str] = Field(default_factory=list)  # normalized-key match
    models: list[str] = Field(default_factory=list)  # normalized-key match
    price_min: int | None = None
    price_max: int | None = None
    year_min: int | None = None
    year_max: int | None = None
    year_calendar: str | None = None  # "shamsi" | "gregorian" | "unknown"
    mileage_max_km: int | None = None
    location_contains: str | None = None

    @model_validator(mode="after")
    def _validate_ranges(self) -> SearchFilters:
        if self.price_min is not None and self.price_max is not None:
            if self.price_min > self.price_max:
                raise ValueError("price_min exceeds price_max")
        if self.year_min is not None and self.year_max is not None:
            if self.year_min > self.year_max:
                raise ValueError("year_min exceeds year_max")
        if self.year_calendar is not None and self.year_calendar not in (
            "shamsi", "gregorian", "unknown",
        ):
            raise ValueError("year_calendar must be shamsi|gregorian|unknown")
        return self

    def is_empty(self) -> bool:
        return not any(
            [
                self.platforms,
                self.brands,
                self.models,
                self.price_min is not None,
                self.price_max is not None,
                self.year_min is not None,
                self.year_max is not None,
                self.year_calendar is not None,
                self.mileage_max_km is not None,
                self.location_contains,
            ]
        )


@dataclass
class SearchResult:
    listing_key: str
    source_platform: str
    score: float  # cosine similarity, [-1, 1]
    text: str  # indexed excerpt (contains source description verbatim)
    original_url: str | None
    raw_payload_ref_json: str
    chunk_index: int
    chunk_count: int
    extraction_version: str

    def summary_line(self) -> str:
        url = self.original_url or "(no url)"
        excerpt = self.text.replace("\n", " | ")[:110]
        return (
            f"{self.score:+.4f}  {self.listing_key}  [{self.source_platform}]  "
            f"{excerpt}  {url}"
        )


@dataclass
class SearchResponse:
    query: str
    filters: SearchFilters
    total_corpus: int
    candidates_after_filters: int
    results: list[SearchResult]
    total_matched: int = 0  # results passing filters (before pagination)
    offset: int = 0
    limit: int | None = None
    sort: str = "relevance"

    def summary_lines(self) -> list[str]:
        mode = "hybrid" if not self.filters.is_empty() else "semantic"
        lines = [
            f"query: {self.query!r}  mode: {mode}  "
            f"corpus={self.total_corpus}  after-filters={self.candidates_after_filters}"
        ]
        for rank, result in enumerate(self.results, start=1):
            lines.append(f"  {rank}. {result.summary_line()}")
        if not self.results:
            lines.append("  (no results — filters matched nothing; filters NOT loosened)")
        return lines


def _apply_filters(metadata: pd.DataFrame, filters: SearchFilters) -> pd.DataFrame:
    """Pre-ranking filter pass over the ENTIRE corpus metadata."""
    mask = pd.Series(True, index=metadata.index)
    if filters.platforms:
        mask &= metadata["source_platform"].isin(filters.platforms)
    if filters.brands:
        wanted = {normalize_for_key(b) for b in filters.brands}
        mask &= metadata["brand_key"].isin(wanted)
    if filters.models:
        wanted = {normalize_for_key(m) for m in filters.models}
        mask &= metadata["model_key"].isin(wanted)
    if filters.price_min is not None:
        mask &= metadata["price_amount"].notna() & (metadata["price_amount"] >= filters.price_min)
    if filters.price_max is not None:
        mask &= metadata["price_amount"].notna() & (metadata["price_amount"] <= filters.price_max)
    if filters.year_min is not None:
        mask &= metadata["year_num"].notna() & (metadata["year_num"] >= filters.year_min)
    if filters.year_max is not None:
        mask &= metadata["year_num"].notna() & (metadata["year_num"] <= filters.year_max)
    if filters.year_calendar is not None:
        mask &= metadata["year_calendar"] == filters.year_calendar
    if filters.mileage_max_km is not None:
        mask &= metadata["mileage_km"].notna() & (metadata["mileage_km"] <= filters.mileage_max_km)
    if filters.location_contains:
        needle = normalize_for_key(filters.location_contains)
        mask &= metadata["location_raw"].map(
            lambda v: bool(v) and needle in normalize_for_key(v)
        )
    return metadata[mask]


def rank_subset(
    query_vector: list[float],
    vectors: np.ndarray,
    subset_positions: np.ndarray,
    top_k: int | None = None,
) -> list[tuple[int, float]]:
    """Cosine-rank the filtered subset (ALL of it — pagination slices later).
    Returns (position, score) pairs, best first."""
    if len(subset_positions) == 0:
        return []
    q = np.asarray(query_vector, dtype=np.float32)
    q_norm = np.linalg.norm(q) or 1.0
    subset = vectors[subset_positions].astype(np.float32)
    norms = np.linalg.norm(subset, axis=1)
    norms[norms == 0] = 1.0
    sims = (subset @ q) / (norms * q_norm)
    order = np.argsort(-sims)
    if top_k is not None:
        order = order[:top_k]
    return [(int(subset_positions[i]), float(sims[i])) for i in order]


class SearchService:
    """Semantic + hybrid search over a persisted index."""

    def __init__(self, vectors: np.ndarray, metadata: pd.DataFrame, encoder) -> None:
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.metadata = metadata.reset_index(drop=True)
        self.encoder = encoder

    def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        top_k: int = 10,
        offset: int = 0,
        sort: str = "relevance",
    ) -> SearchResponse:
        """Rank the ENTIRE filtered subset, then paginate.

        Pagination never loosens filters and never re-ranks per page — page 2
        is a true continuation of page 1. `sort` may reorder the ranked subset
        by structured fields (relevance | price_asc | price_desc | year_desc).
        """
        filters = filters or SearchFilters()
        filtered = (
            _apply_filters(self.metadata, filters)
            if not filters.is_empty()
            else self.metadata
        )
        subset_positions = filtered.index.to_numpy()

        query_vector = self.encoder.encode_query(query)
        ranked = rank_subset(query_vector, self.vectors, subset_positions)
        total_matched = len(ranked)

        if sort in ("price_asc", "price_desc", "year_desc"):
            def sort_key(item: tuple[int, float]):
                row = self.metadata.iloc[item[0]]
                if sort == "price_asc":
                    return (
                        row["price_amount"]
                        if row["price_amount"] == row["price_amount"]
                        else float("inf")
                    )
                if sort == "price_desc":
                    return (
                        -(row["price_amount"])
                        if row["price_amount"] == row["price_amount"]
                        else float("inf")
                    )
                return (
                    -(row["year_num"])
                    if row["year_num"] == row["year_num"]
                    else float("inf")
                )

            ranked = sorted(ranked, key=sort_key)

        page = ranked[offset : offset + top_k]

        results: list[SearchResult] = []
        for position, score in page:
            row = self.metadata.iloc[position]
            results.append(
                SearchResult(
                    listing_key=row["listing_key"],
                    source_platform=row["source_platform"],
                    score=round(score, 4),
                    text=row["text"],
                    original_url=row["original_url"],
                    raw_payload_ref_json=row["raw_payload_ref_json"],
                    chunk_index=int(row["chunk_index"]),
                    chunk_count=int(row["chunk_count"]),
                    extraction_version=row["extraction_version"],
                )
            )
        return SearchResponse(
            query=query,
            filters=filters,
            total_corpus=len(self.metadata),
            candidates_after_filters=len(filtered),
            results=results,
            total_matched=total_matched,
            offset=offset,
            limit=top_k,
            sort=sort,
        )
