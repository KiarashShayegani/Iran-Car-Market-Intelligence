"""Search service: entity-steered hybrid retrieval over the vector index.

Contract (owner requirements, F3c):
- Filters are GENUINELY restrictive: they select a subset of the whole corpus
  metadata first; ranking runs only inside that subset. Zero matches return an
  honest empty list — filters are never loosened to fill results.
- Query-understanding steering (brands/models detected in the QUERY text)
  narrows candidates the same way, with graceful fallback when the steered
  subset is empty — that is intent parsing, not filter loosening.
- Results are explainable: similarity score, matching excerpt (the indexed
  document text, containing the source description verbatim), source platform,
  and the original advertisement link.
- Traceability survives retrieval: listing_key + raw_payload_ref on every hit.

Retrieval modes (why hybrid): pure-semantic ranking misplaces short entity
queries — e5-small ranks «ام جی» (MG) first for "AMG" and شاهین «تیپ G» for
"G class". The pipeline is now: detect entities (brands/models) → steer the
candidate subset → keyword-score the subset → tiered fusion (entity matches
lead, keyword hits next, pure-semantic remainder). Natural-language queries
with no entity/keyword hits fall back to pure semantic ranking unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from icmi.retrieval.entities import (
    QueryEntities,
    detect_entities,
    model_match_pattern,
)
from icmi.retrieval.normalization import normalize_for_key

# explicit-sort re-ranking (newest/price/year over a query result) applies to
# this fused/semantic top pool only — re-sorting the entire corpus by recency
# would destroy relevance beyond the first page
SORT_POOL_CAP = 300

# browse diversity: within this recency window at most DIVERSE_PER_BRAND ads
# of the same brand surface first (brand-rotation sweeps ingest up to ~750
# same-brand ads in one batch, so the window must span several sweeps);
# the window remainder follows in recency order
DIVERSE_WINDOW = 1500
DIVERSE_PER_BRAND = 2


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
    fuel: list[str] = Field(default_factory=list)
    transmission: list[str] = Field(default_factory=list)
    body_type: list[str] = Field(default_factory=list)

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
                self.fuel,
                self.transmission,
                self.body_type,
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
    fused_score: float | None = None  # RRF fusion score when hybrid ranking ran

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
    retrieval_mode: str = "semantic"  # browse|semantic|keyword+semantic|entity-steered hybrid

    def summary_lines(self) -> list[str]:
        mode = "hybrid" if not self.filters.is_empty() else "semantic"
        lines = [
            f"query: {self.query!r}  mode: {mode} ({self.retrieval_mode})  "
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
        # contains-match on the normalized raw model: a user/agentic model
        # reference ("206") must match every trim variant in the corpus
        # ("206 sd v20", "206 (تیپ2)") — model_key is a composite key
        # (year-prefixed), so exact/prefix matching on it fails.
        wanted = [normalize_for_key(m) for m in filters.models if m]
        mask &= metadata["model_raw"].apply(
            lambda raw: any(w in normalize_for_key(str(raw)) for w in wanted if w)
            if raw is not None else False
        )
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
    if filters.fuel:
        wanted = {f.strip() for f in filters.fuel}
        mask &= metadata["fuel_raw"].map(lambda v: bool(v) and v.strip() in wanted)
    if filters.transmission:
        wanted = {t.strip() for t in filters.transmission}
        mask &= metadata["transmission_raw"].map(lambda v: bool(v) and v.strip() in wanted)
    if filters.body_type:
        wanted = {b.strip() for b in filters.body_type}
        mask &= metadata["body_type_raw"].map(lambda v: bool(v) and v.strip() in wanted)
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


def _browse_order(
    metadata: pd.DataFrame,
    subset_positions: np.ndarray,
    sort: str,
) -> list[tuple[int, float]]:
    """Filter-only browsing: order the subset without semantic ranking.

    newest adds brand diversity: at most DIVERSE_PER_BRAND ads per brand may
    surface inside the DIVERSE_WINDOW newest — collection sweeps are
    brand-rotated (bama brand presets), so raw recency shows a wall of one
    model. The rest of the window and everything older follow in recency
    order, so deeper pages stay complete.
    """
    df = metadata.iloc[subset_positions] if len(subset_positions) else metadata.iloc[0:0]
    if sort == "price_asc":
        ordered = df.sort_values("price_amount", ascending=True, na_position="last")
    elif sort == "price_desc":
        ordered = df.sort_values("price_amount", ascending=False, na_position="last")
    elif sort == "year_desc":
        ordered = df.sort_values("year_num", ascending=False, na_position="last")
    else:  # newest (default browse order)
        if "first_seen_at" in df.columns:
            ordered = df.sort_values("first_seen_at", ascending=False, na_position="last")
        else:
            ordered = df.sort_index(ascending=False)

    if sort == "newest" and len(ordered) > DIVERSE_PER_BRAND:
        window = ordered.iloc[:DIVERSE_WINDOW]
        rest = ordered.iloc[DIVERSE_WINDOW:]
        brand_col = (
            window["brand_key"].fillna(window["source_platform"])
            if "brand_key" in window.columns
            else window["source_platform"]
        )
        picked, counts = [], {}
        # equal lengths by construction: both lists come from the same frame
        for pos, brand in zip(window.index.tolist(), brand_col.tolist(), strict=False):
            counts[brand] = counts.get(brand, 0) + 1
            if counts[brand] <= DIVERSE_PER_BRAND:
                picked.append(pos)
        deferred = [p for p in window.index.tolist() if p not in set(picked)]
        ordered_positions = picked + deferred + rest.index.tolist()
        return [(int(pos), 0.0) for pos in ordered_positions]

    return [(int(pos), 0.0) for pos in ordered.index]


def _keyword_scores(
    metadata: pd.DataFrame,
    positions: np.ndarray,
    entities: QueryEntities,
) -> pd.Series:
    """Token-overlap scores over the normalized document text of the subset.

    Weights: token-in-text +1, full-phrase hit +2. Model-entity matches are
    tiered separately (see _model_hits) — they outrank plain tokens."""
    df = metadata.iloc[positions]
    scores = pd.Series(0.0, index=df.index)
    texts = df["normalized_text"].fillna("")

    for tok in entities.keyword_tokens:
        if len(tok) < 2:
            continue
        scores += texts.str.contains(tok, regex=False).astype(float)

    if entities.phrase and len(entities.phrase) >= 4:
        scores += texts.str.contains(entities.phrase, regex=False).astype(float) * 2.0
    return scores


def _model_hits(
    metadata: pd.DataFrame,
    positions: np.ndarray,
    entities: QueryEntities,
) -> set[int]:
    """Positions matching a detected model entity: corpus-vocabulary
    membership (listing-level, covers bama titles that are NOT in the doc
    text) or a word-boundary regex on the normalized text (covers other
    platforms' title/description mentions)."""
    hits: set[int] = set()
    if not entities.models:
        return hits
    df = metadata.iloc[positions]
    patterns: dict[str, str] = {}
    for entity in entities.models:
        if entity.listing_keys:
            hits.update(df.index[df["listing_key"].isin(entity.listing_keys)].tolist())
        if entity.label:
            pattern = patterns.setdefault(
                entity.label, model_match_pattern(entity.label)
            )
            if pattern:
                matched = df["normalized_text"].str.contains(pattern, regex=True)
                hits.update(df.index[matched].tolist())
    return hits


def _fuse(
    semantic_ranked: list[tuple[int, float]],
    kw_scores: pd.Series,
    model_hit_positions: set[int] | None = None,
) -> list[tuple[int, float, float]]:
    """Tiered fusion of entity matches, keyword hits, and semantic order.

    Tier 0: model-entity matches (corpus membership or title regex) — what
    the user literally named always leads.
    Tier 1: keyword-hit rows (token overlap).
    Tier 2: pure-semantic remainder.
    Within each tier the semantic order is preserved (stable tiers, honest
    ranking inside each). RRF was tried and rejected: it is too soft — a
    sem-rank-1 row with a stray token hit outranked literal entity matches.
    """
    model_hit_positions = model_hit_positions or set()
    tiers: dict[int, list[tuple[int, float, float]]] = {0: [], 1: [], 2: []}
    for pos, sim in semantic_ranked:
        kw = float(kw_scores.get(pos, 0.0))
        tier = 0 if pos in model_hit_positions else 1 if kw > 0 else 2
        fused = (3 - tier) + sim / 100.0
        tiers[tier].append((pos, sim, fused))
    fused_list: list[tuple[int, float, float]] = []
    for tier in (0, 1, 2):
        fused_list.extend(tiers[tier])
    return fused_list


class SearchService:
    """Semantic + keyword hybrid search over a persisted index."""

    def __init__(
        self,
        vectors: np.ndarray,
        metadata: pd.DataFrame,
        encoder,
        entity_vocab: dict[str, dict] | None = None,
        brand_canonicals: set[str] | None = None,
    ) -> None:
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.metadata = metadata.reset_index(drop=True)
        self.encoder = encoder
        # corpus-derived entity vocabulary (model → brands + listing keys),
        # injected by the API layer; detection degrades gracefully without it
        self.entity_vocab = entity_vocab or {}
        self.brand_canonicals = brand_canonicals or set()

    def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        top_k: int = 10,
        offset: int = 0,
        sort: str = "relevance",
    ) -> SearchResponse:
        """Entity-steered hybrid search, or filter-only browsing when the
        query is empty. Ranking always covers the ENTIRE (steered) subset;
        pagination slices afterwards — page 2 is a true continuation of
        page 1.

        sort: relevance | price_asc | price_desc | year_desc | newest.
        Empty query → browse mode (no semantic ranking; newest by default)."""
        filters = filters or SearchFilters()
        filtered = (
            _apply_filters(self.metadata, filters)
            if not filters.is_empty()
            else self.metadata
        )
        subset_positions = filtered.index.to_numpy()
        total_matched = len(subset_positions)

        query_stripped = (query or "").strip()
        if not query_stripped:
            ranked = _browse_order(self.metadata, subset_positions, sort)
            return self._respond(
                query, filters, filtered, total_matched, ranked,
                offset, top_k, sort, retrieval_mode="browse",
            )

        entities = detect_entities(query_stripped, self.entity_vocab, self.brand_canonicals)
        steering_applied = False

        # entity steering — only when the user gave no explicit brand filter
        # (explicit filters always win over intent parsing)
        if entities.brands and not filters.brands:
            keys = {normalize_for_key(b) for b in entities.brands}
            brand_mask = self.metadata["brand_key"].isin(keys)
            steered = self.metadata[brand_mask]
            steered = steered.loc[steered.index.isin(subset_positions)]
            if len(steered):
                subset_positions = steered.index.to_numpy()
                total_matched = len(subset_positions)
                steering_applied = True

        # model references BOOST (regex + corpus-vocabulary membership) rather
        # than hard-narrow: query understanding must not hide inventory —
        # "G class" surfaces «کلاس G» rows on top while the steered brand
        # subset stays browsable below
        query_vector = self.encoder.encode_query(query_stripped)
        semantic_ranked = rank_subset(query_vector, self.vectors, subset_positions)

        kw = _keyword_scores(self.metadata, subset_positions, entities)
        model_hits = _model_hits(self.metadata, subset_positions, entities)
        if (kw > 0).any() or model_hits or steering_applied:
            fused = _fuse(semantic_ranked, kw, model_hits)
            ranked = [(pos, sim) for pos, sim, _ in fused]
            fused_map = {pos: fs for pos, _, fs in fused}
            mode_parts = []
            if steering_applied:
                mode_parts.append("entity-steered")
            mode_parts.append("keyword+semantic fusion")
            retrieval_mode = " · ".join(mode_parts)
        else:
            ranked = semantic_ranked
            fused_map = {}
            retrieval_mode = "semantic"

        if sort != "relevance":
            ranked = self._resort_pool(ranked, sort)

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
                    fused_score=(
                        round(fused_map[position], 6) if position in fused_map else None
                    ),
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
            retrieval_mode=retrieval_mode,
        )

    def _resort_pool(self, ranked: list[tuple[int, float]], sort: str) -> list[tuple[int, float]]:
        """Explicit user sort over a query result: re-order only the top
        SORT_POOL_CAP pool (re-sorting the entire corpus by price/recency
        would push all relevant results beyond the first page)."""
        pool = ranked[:SORT_POOL_CAP]
        tail = ranked[SORT_POOL_CAP:]

        def value_of(pos: int) -> float:
            row = self.metadata.iloc[pos]
            if sort == "newest":
                v = row["first_seen_at"]
                return float("inf") if pd.isna(v) else -pd.Timestamp(v).value
            col = "price_amount" if sort in ("price_asc", "price_desc") else "year_num"
            v = row[col]
            return float("inf") if pd.isna(v) else float(v)

        sign = -1.0 if sort in ("price_desc", "year_desc") else 1.0
        ordered = sorted(pool, key=lambda item: sign * value_of(item[0]))
        return ordered + tail

    def _respond(
        self,
        query: str,
        filters: SearchFilters,
        filtered: pd.DataFrame,
        total_matched: int,
        ranked: list[tuple[int, float]],
        offset: int,
        top_k: int,
        sort: str,
        retrieval_mode: str,
    ) -> SearchResponse:
        results: list[SearchResult] = []
        for position, score in ranked[offset : offset + top_k]:
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
            retrieval_mode=retrieval_mode,
        )
