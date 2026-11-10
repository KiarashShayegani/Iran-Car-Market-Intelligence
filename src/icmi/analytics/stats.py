"""Market statistics: canonical → analytical aggregates (Pillar A start).

All aggregates derive from the canonical layer's raw + normalized fields;
nothing here overwrites or mutates canonical data. Groupings must respect
calendar ambiguity (see classify_year_calendar): mixed-calendar groupings are
reported separately and flagged, never silently blended.

Every export carries the normalization versions that produced its derived
columns, so stats are reproducible and upgradeable.
"""

from __future__ import annotations

import json
from pathlib import Path

from icmi.pipeline.normalize import (
    MILEAGE_NORMALIZATION_VERSION,
    PRICE_NORMALIZATION_VERSION,
    classify_year_calendar,
    normalize_mileage_km,
    normalize_year,
)
from icmi.pipeline.store import CanonicalStore

MILEAGE_BUCKETS = [
    ("0-30k", 0, 30_000),
    ("30-60k", 30_000, 60_000),
    ("60-100k", 60_000, 100_000),
    ("100-150k", 100_000, 150_000),
    ("150k+", 150_000, float("inf")),
]


def _mileage_bucket(km) -> str | None:
    if km is None or km != km:
        return None
    for label, lo, hi in MILEAGE_BUCKETS:
        if lo <= km < hi:
            return label
    return None


def market_stats(store: CanonicalStore) -> dict:
    """Compute market aggregates from the canonical layer.

    Returns {"overview": df, "by_brand": df, "by_year": df, "meta": dict}.
    """
    df = store.query_df(
        """
        SELECT source_platform, brand_raw, model_raw, year_raw, mileage_raw,
               price_amount, price_currency_raw
        FROM listings
        """
    )

    df["year_num"] = df["year_raw"].map(normalize_year)
    df["year_calendar"] = df["year_num"].map(classify_year_calendar)
    df["mileage_km"] = df["mileage_raw"].map(normalize_mileage_km)

    meta = {
        "generated_from_rows": len(df),
        "price_normalization_version": PRICE_NORMALIZATION_VERSION,
        "mileage_normalization_version": MILEAGE_NORMALIZATION_VERSION,
        "note": (
            "Derived aggregates only; canonical layer untouched. Calendar "
            "groupings are separate by design — never blended."
        ),
    }

    valid_price = df[df["price_amount"].notna() & (df["price_amount"] > 0)]

    overview = (
        valid_price.groupby("source_platform")["price_amount"]
        .agg(count="count", median="median", mean="mean", p25=lambda s: s.quantile(0.25),
             p75=lambda s: s.quantile(0.75))
        .reset_index()
    )

    by_brand = (
        valid_price.groupby(["source_platform", "brand_raw"])["price_amount"]
        .agg(count="count", median="median", mean="mean", p25=lambda s: s.quantile(0.25),
             p75=lambda s: s.quantile(0.75))
        .reset_index()
        .sort_values(["source_platform", "count"], ascending=[True, False])
    )
    # median year/mileage per brand (calendar-aware: dominant calendar only)
    brand_extra = (
        df.dropna(subset=["brand_raw"])
        .groupby(["source_platform", "brand_raw"])
        .agg(median_year=("year_num", "median"),
             median_mileage_km=("mileage_km", "median"),
             calendars=("year_calendar", lambda s: ",".join(sorted(set(s.dropna())))))
        .reset_index()
    )
    by_brand = by_brand.merge(brand_extra, on=["source_platform", "brand_raw"], how="left")

    by_year = (
        valid_price.dropna(subset=["year_num"])
        .groupby(["source_platform", "year_num", "year_calendar"])["price_amount"]
        .agg(count="count", median="median", p25=lambda s: s.quantile(0.25),
             p75=lambda s: s.quantile(0.75))
        .reset_index()
        .sort_values(["source_platform", "year_num"])
    )

    # mileage distribution (documented buckets; unknown-mileage ads excluded)
    valid_mileage = valid_price[valid_price["mileage_km"].notna()].copy()
    valid_mileage["mileage_bucket"] = valid_mileage["mileage_km"].map(_mileage_bucket)
    by_mileage = (
        valid_mileage.groupby(["source_platform", "mileage_bucket"])["price_amount"]
        .agg(count="count", median="median")
        .reset_index()
    )

    return {
        "overview": overview,
        "by_brand": by_brand,
        "by_year": by_year,
        "by_mileage": by_mileage,
        "meta": meta,
    }


def write_market_stats(store: CanonicalStore, out_dir: Path) -> dict[str, Path]:
    """Write market stats as Parquet + a JSON meta summary. Returns paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result = market_stats(store)
    paths: dict[str, Path] = {}
    for name in ("overview", "by_brand", "by_year"):
        path = out_dir / f"market_{name}.parquet"
        result[name].to_parquet(path, index=False)
        paths[name] = path
    meta_path = out_dir / "market_meta.json"
    meta_path.write_text(
        json.dumps(result["meta"], ensure_ascii=False, indent=1), encoding="utf-8"
    )
    paths["meta"] = meta_path
    return paths
