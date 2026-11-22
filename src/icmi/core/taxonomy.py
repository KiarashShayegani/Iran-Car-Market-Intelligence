"""Vehicle taxonomy derived from Bama titles (the structured source).

Bama titles follow «برند، مدل [| تیپ …]» — the clean model name lives in the
second comma-separated token, while model_raw is a spec/trim composite
(«1405 | 1.5 لیتر توربو», «TU5 | 1404») that clutters model filters. This
module is the SINGLE extraction point shared by the estimate taxonomy, the
dashboard facets, and the entity index — one rule, everywhere.

Read-time derivation by design: the extraction is a pure function of the
title, so future ingests need no backfill and the taxonomy can never drift
from the data. Raw titles stay verbatim in the store.
"""

from __future__ import annotations

import pandas as pd

from icmi.pipeline.store import CanonicalStore
from icmi.retrieval.normalization import normalize_for_key


def extract_model_from_title(title: str | None) -> str | None:
    """Model token from a bama-style title («پژو، 207» → «207»).

    Rule: the second comma-separated token, cut at the pipe separator.
    Returns None for titles without the «برند، مدل» shape — honest missing.
    """
    if not title or "،" not in title:
        return None
    tok = title.split("،")[1].split("|")[0].strip()
    return tok or None


def add_model_label(df: pd.DataFrame, title_col: str = "title") -> pd.DataFrame:
    """Return a copy with a ``model_label`` column: the title-extracted model
    when the title carries one, else the platform's own model_raw."""
    extracted = (
        df[title_col]
        .fillna("")
        .astype(str)
        .map(extract_model_from_title)
        if title_col in df.columns
        else pd.Series([None] * len(df), index=df.index)
    )
    out = df.copy()
    fallback = (
        out["model_raw"]
        if "model_raw" in out.columns
        else pd.Series([None] * len(out), index=out.index)
    )
    out["model_label"] = extracted.where(
        extracted.notna() & (extracted != ""), fallback
    )
    return out


def build_model_vocab(
    store: CanonicalStore, min_count: int = 1
) -> dict[str, dict]:
    """Corpus-derived model vocabulary for query entity detection.

    Scans bama titles → normalized-key → {label, brands, listing_keys}.
    A model key maps to every brand that offers it (a Latin model token such
    as «207» always belongs to پژو in this corpus; «امگرند» to جیلی — the
    co-occurrence decides, no hardcoding).
    """
    df = store.query_df(
        """
        SELECT listing_key, title, brand_canonical
        FROM listings
        WHERE source_platform = 'bama' AND title IS NOT NULL
          AND brand_canonical IS NOT NULL
        """
    )
    vocab: dict[str, dict] = {}
    for _, row in df.iterrows():
        model = extract_model_from_title(row["title"])
        if not model:
            continue
        key = normalize_for_key(model)
        if len(key) < 2:  # single-letter model tokens are noise for detection
            continue
        entry = vocab.setdefault(
            key, {"label": model, "brands": set(), "listing_keys": set()}
        )
        entry["brands"].add(row["brand_canonical"])
        entry["listing_keys"].add(row["listing_key"])
    return {
        key: {
            "label": e["label"],
            "brands": e["brands"],
            "listing_keys": e["listing_keys"],
        }
        for key, e in vocab.items()
        if len(e["listing_keys"]) >= min_count
    }


def build_bama_taxonomy(store: CanonicalStore) -> dict:
    """Guided-flow taxonomy (estimate page): brand → models → trims, plus the
    categorical option lists, all bama-only (most structured source)."""
    df = store.query_df(
        """
        SELECT brand_canonical AS brand, title, model_raw,
               year_num, body_type_raw, body_condition_raw,
               transmission_raw, fuel_raw
        FROM listings
        WHERE source_platform = 'bama'
        """
    )
    df = add_model_label(df)

    models: dict[str, dict[str, int]] = {}
    trims_by_model: dict[tuple[str, str], set] = {}
    years: set[int] = set()
    body_conditions: set[str] = set()
    body_types: set[str] = set()
    transmissions: set[str] = set()
    fuels: set[str] = set()

    for _, r in df.iterrows():
        brand = r["brand"]
        model = r["model_label"]
        if brand and model:
            m = models.setdefault(brand, {})
            m[model] = m.get(model, 0) + 1
            trim = r["model_raw"]  # bama's trim/variant composite
            if trim:
                trims_by_model.setdefault((brand, model), set()).add(trim)
        if r["year_num"] and 1330 <= r["year_num"] <= 1406:
            years.add(int(r["year_num"]))
        for col, bag in (
            ("body_condition_raw", body_conditions),
            ("body_type_raw", body_types),
            ("transmission_raw", transmissions),
            ("fuel_raw", fuels),
        ):
            v = r[col]
            if v:
                bag.add(str(v))

    brand_list = [
        {
            "brand": b,
            "count": sum(models[b].values()),
            "models": sorted(models[b].keys()),
        }
        for b in sorted(models.keys(), key=lambda b: -sum(models[b].values()))
    ]
    return {
        "source": "bama",
        "brands": brand_list,
        "trims": {
            f"{b}||{m}": sorted(t) for (b, m), t in trims_by_model.items()
        },
        "years": sorted(years, reverse=True),
        "body_conditions": sorted(body_conditions),
        "body_types": sorted(body_types),
        "transmissions": sorted(transmissions),
        "fuels": sorted(fuels),
    }
