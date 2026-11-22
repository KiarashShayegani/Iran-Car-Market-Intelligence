"""Query entity detection: brands and models referenced in a search query.

Why: short entity queries break pure-semantic retrieval — the e5-small
embedding ranks «ام جی» (MG) first for "AMG" (phonetic proximity in the
vector space) and شاهین «تیپ G» for "G class". Marketplace-grade search
needs a deterministic layer: detect the brand/model the user named, steer
the candidate subset to it, and only then let semantics rank.

Sources of truth, in order:
1. config/brand_aliases.yaml — brand variants (Latin slugs, spellings) →
   canonical Persian brand (already used for facet normalization).
2. config/entity_aliases.yaml — curated Latin/transliterated model refs
   ("amg", "gclass", "c200", …) → brand + optional bama model label.
3. corpus vocabulary (build_model_vocab) — bama titles give every Persian
   model name + the brand(s) offering it + its listing keys.

Detection is EXACT-key only (no substring) so «ام جی» never fires on
«ام جی 5» noise and single letters are ignored. Detection steers — it never
overrides explicit user filters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from icmi.retrieval.normalization import normalize_for_key, normalize_for_matching

_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "entity_aliases.yaml"

# Persian query filler that neither names a vehicle nor helps keyword ranking
STOP_TOKENS = frozenset(
    {
        "زیر", "بالای", "بین", "با", "برای", "و", "در", "از", "که", "را",
        "ماشین", "خودرو", "سواری", "اتومبیل", "چه", "چی", "کدام", "هست",
        "بهترین", "ارزان", "ارزانترین", "گران", "گرانترین", "کم", "زیاد",
        "میخوام", "میخواهم", "دنبال", "یک", "یه", "تا", "هزار", "میلیون",
        "میلیارد", "تومان", "قیمت", "مدل", "سال", "کارکرد", "کیلومتر",
    }
)


@dataclass
class ModelEntity:
    label: str  # display label as found in bama titles («کلاس G», «207»)
    brands: set[str] = field(default_factory=set)  # canonical brand labels
    listing_keys: set[str] = field(default_factory=set)  # corpus membership


@dataclass
class QueryEntities:
    brands: set[str] = field(default_factory=set)  # canonical Persian labels
    models: list[ModelEntity] = field(default_factory=list)
    keyword_tokens: list[str] = field(default_factory=list)
    phrase: str = ""  # matching-normalized full query (phrase-bonus scoring)

    @property
    def steered(self) -> bool:
        return bool(self.brands or self.models)


@lru_cache(maxsize=1)
def load_entity_aliases(path: str | None = None) -> dict[str, dict]:
    """normalized variant -> {brand?, model?} from the curated YAML."""
    p = Path(path) if path else _DEFAULT_PATH
    if not p.is_file():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {
        normalize_for_key(variant): spec
        for variant, spec in (raw.get("entities", {}) or {}).items()
        if spec
    }


def detect_entities(
    query: str,
    model_vocab: dict[str, dict] | None = None,
    brand_canonicals: set[str] | None = None,
) -> QueryEntities:
    """Detect brands/models in a query; leftover tokens become keywords.

    model_vocab: corpus-derived vocabulary (key → {label, brands,
    listing_keys}). brand_canonicals: the canonical brand labels present in
    the corpus, for direct Persian mentions («بنز», «ام جی»).
    """
    entities = QueryEntities()
    entities.phrase = normalize_for_key(query).lower()

    from icmi.core.brand_aliases import load_aliases

    brand_map = load_aliases()  # normalized variant -> canonical Persian
    alias_keys = set(brand_map.keys())
    alias_values = {normalize_for_key(v) for v in brand_map.values()}
    if brand_canonicals:
        alias_values |= {normalize_for_key(v) for v in brand_canonicals}

    vocab = model_vocab or {}
    curated = load_entity_aliases()

    # candidate keys: single tokens + adjacent token pairs ("g class" is two
    # tokens; normalize_for_key of the pair yields "gclass"). Tokens come
    # from normalize_for_matching (spaces PRESERVED — normalize_for_key
    # strips them, which would fuse «مرسدس بنز G» into one token).
    tokens = [
        t for t in normalize_for_matching(query).lower().split(" ") if t
    ]
    candidates: list[str] = [normalize_for_key(t) for t in tokens if t]
    candidates += [
        normalize_for_key(f"{a} {b}")
        for a, b in zip(tokens, tokens[1:], strict=False)
        if normalize_for_key(f"{a} {b}")
    ]
    candidates = [c for c in candidates if len(c) >= 2]

    consumed: set[str] = set()
    for cand in candidates:
        if cand in consumed:
            continue
        hit: ModelEntity | None = None
        if cand in curated:
            spec = curated[cand]
            brands = {spec["brand"]} if spec.get("brand") else set()
            # vocab keys are normalized (no spaces) — normalize the curated
            # model label the same way before the lookup
            model_key = normalize_for_key(spec["model"]) if spec.get("model") else None
            if model_key and model_key in vocab:
                m = vocab[model_key]
                hit = ModelEntity(
                    label=m["label"],
                    brands=brands | m["brands"],
                    listing_keys=m["listing_keys"],
                )
            elif brands or spec.get("model"):
                hit = ModelEntity(
                    label=spec.get("model") or "", brands=brands,
                    listing_keys=set(),
                )
        elif cand in vocab:
            m = vocab[cand]
            hit = ModelEntity(
                label=m["label"], brands=set(m["brands"]),
                listing_keys=set(m["listing_keys"]),
            )
        elif cand in alias_keys:
            entities.brands.add(brand_map[cand])
            consumed.add(cand)
            continue
        elif cand in alias_values:
            entities.brands.add(cand)  # already canonical
            consumed.add(cand)
            continue
        if hit is not None and (hit.label or hit.listing_keys):
            entities.models.append(hit)
            entities.brands |= {b for b in hit.brands if b}
            if hit.label:
                consumed.add(normalize_for_key(hit.label))
            consumed.add(cand)
        elif hit is not None and hit.brands:
            # brand-only curated ref («amg») — brand steers, the token stays
            # a keyword so rows literally mentioning it rank higher
            entities.brands |= {b for b in hit.brands if b}

    # keyword tokens: single tokens not consumed by entity matching
    entities.keyword_tokens = [
        t for t in tokens if t not in STOP_TOKENS and normalize_for_key(t) not in consumed
    ]
    return entities


def model_match_pattern(model_label: str) -> str:
    r"""Regex matching a model label as whole words inside normalized text.

    «کلاس G» → \bکلاس\s+g\b — the trailing word-boundary stops «کلاس G»
    from matching «کلاس GLC». Persian letters are \w in Python re, so
    boundaries work in both scripts.
    """
    import re

    body = normalize_for_matching(model_label)
    parts = [re.escape(p) for p in body.split(" ") if p]
    if not parts:
        return ""
    return r"\b" + r"\s+".join(parts) + r"\b"
