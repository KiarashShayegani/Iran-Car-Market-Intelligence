"""Market-analytics aggregations for the interactive dashboard.

One endpoint (`/analytics`) assembles every dashboard section from filtered
SQL aggregations — nothing is loaded into the browser to compute statistics.
All filters are validated and parameterized; aggregation happens in SQLite.

Honesty rules (owner brief): statistics describe ADVERTISEMENT counts, not
unique vehicles (cross-platform entity resolution is future work); price
"trends" are per-collection-run series, not daily market history; sections
with insufficient data are reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass

from icmi.pipeline.store import CanonicalStore

MILEAGE_BUCKETS = [
    ("0-30k", 0, 30_000),
    ("30-60k", 30_000, 60_000),
    ("60-100k", 60_000, 100_000),
    ("100-150k", 100_000, 150_000),
    ("150k+", 150_000, float("inf")),
]


@dataclass
class AnalyticsFilters:
    brand_keys: list[str]
    models: list[str]
    year_min: int | None
    year_max: int | None
    year_calendar: str | None
    platforms: list[str]
    fuel: list[str]
    transmission: list[str]
    mileage_max_km: int | None
    price_min: int | None
    price_max: int | None


def _where(f: AnalyticsFilters) -> tuple[str, list]:
    """Build a WHERE clause over listings from validated filters."""
    clauses: list[str] = ["l.price_amount IS NOT NULL", "l.price_amount > 0"]
    params: list = []

    def add_in(column: str, values: list[str]) -> None:
        if values:
            marks = ",".join("?" for _ in values)
            clauses.append(f"l.{column} IN ({marks})")
            params.extend(values)

    add_in("brand_key", f.brand_keys)
    if f.models:
        marks = ",".join("?" for _ in f.models)
        clauses.append(f"l.model_raw IN ({marks})")
        params.extend(f.models)
    if f.year_min is not None:
        clauses.append("l.year_num >= ?")
        params.append(f.year_min)
    if f.year_max is not None:
        clauses.append("l.year_num <= ?")
        params.append(f.year_max)
    if f.year_calendar:
        clauses.append("l.year_calendar = ?")
        params.append(f.year_calendar)
    add_in("source_platform", f.platforms)
    add_in("fuel_raw", f.fuel)
    add_in("transmission_raw", f.transmission)
    if f.mileage_max_km is not None:
        clauses.append("l.mileage_km IS NOT NULL AND l.mileage_km <= ?")
        params.append(f.mileage_max_km)
    if f.price_min is not None:
        clauses.append("l.price_amount >= ?")
        params.append(f.price_min)
    if f.price_max is not None:
        clauses.append("l.price_amount <= ?")
        params.append(f.price_max)
    return " AND ".join(clauses), params


def _pct(values: list[int], p: float) -> int | None:
    if not values:
        return None
    rows = sorted(values)
    return rows[min(int(round(p * (len(rows) - 1))), len(rows) - 1)]


def build_dashboard(store: CanonicalStore, f: AnalyticsFilters) -> dict:
    where, params = _where(f)

    base = store.query_df(
        f"""
        SELECT l.listing_key, l.source_platform, l.brand_raw, l.brand_key,
               l.model_raw, l.year_num, l.year_calendar, l.mileage_km,
               l.price_amount, l.first_seen_at
        FROM listings l WHERE {where}
        """,
        params,
    )

    total = len(base)
    if total == 0:
        return {
            "total": 0,
            "insufficient": "هیچ آگهی دارای قیمت با این فیلترها یافت نشد.",
            "kpis": {},
            "by_brand": [], "by_model": [], "by_year": [], "by_platform": [],
            "by_mileage": [], "price_trend": [],
        }

    prices = [int(v) for v in base["price_amount"].tolist()]
    mile = base[base["mileage_km"].notna()]

    brands = base.groupby("brand_raw").size().sort_values(ascending=False)
    models = base.groupby(["brand_raw", "model_raw"]).size().sort_values(ascending=False)
    platforms = base.groupby("source_platform").size()
    years = (
        base.dropna(subset=["year_num"])
        .groupby(["year_num", "year_calendar"])["price_amount"]
        .agg(["count", "median"])
        .reset_index()
    )

    mileage_buckets = []
    for label, lo, hi in MILEAGE_BUCKETS:
        in_bucket = mile[(mile["mileage_km"] >= lo) & (mile["mileage_km"] < hi)]
        if len(in_bucket):
            mileage_buckets.append(
                {"bucket": label, "count": int(len(in_bucket)),
                 "median": int(in_bucket["price_amount"].median())}
            )

    # price trend per collection run (honest: run-series, not daily history)
    trend = store.query_df(
        f"""
        SELECT substr(r.started_at, 1, 10) AS run_date, po.price_amount
        FROM price_observations po
        JOIN listings l ON po.listing_key = l.listing_key
        JOIN runs r ON po.run_id = r.run_id
        WHERE {where}
        """,
        params,
    )
    price_trend = (
        trend.groupby("run_date")["price_amount"]
        .agg(count="count", median="median",
             p25=lambda s: s.quantile(0.25), p75=lambda s: s.quantile(0.75))
        .reset_index()
        .sort_values("run_date")
        .to_dict(orient="records")
    )

    newest_day = str(base["first_seen_at"].max())[:10]
    new_listings = int((base["first_seen_at"].str.slice(0, 10) == newest_day).sum())

    def _int_or_none(v):
        if v is None or v != v:
            return None
        return int(v)

    return {
        "total": total,
        "insufficient": (
            None if total >= 5
            else f"نمونهٔ محدود ({total} آگهی) — آمار با احتیاط تفسیر شود."
        ),
        "kpis": {
            "total": total,
            "new_listings": new_listings,
            "median": _pct(prices, 0.5),
            "p25": _pct(prices, 0.25),
            "p75": _pct(prices, 0.75),
            "brands": int(base["brand_raw"].nunique()),
            "models": int(base.groupby(["brand_raw", "model_raw"]).ngroups),
            "median_mileage": (
                int(mile["mileage_km"].median()) if len(mile) else None
            ),
        },
        "by_brand": [
            {"label": str(k), "count": int(v)} for k, v in brands.head(12).items()
        ],
        "by_model": [
            {"brand": str(k[0]), "label": str(k[1]), "count": int(v)}
            for k, v in models.head(12).items()
        ],
        "by_platform": [
            {"platform": str(k), "count": int(v)} for k, v in platforms.items()
        ],
        "by_year": [
            {"year": int(r["year_num"]), "calendar": r["year_calendar"],
             "count": int(r["count"]), "median": _int_or_none(r["median"])}
            for _, r in years.iterrows()
        ],
        "by_mileage": mileage_buckets,
        "price_trend": [
            {"date": r["run_date"], "count": int(r["count"]),
             "median": _int_or_none(r["median"]),
             "p25": _int_or_none(r["p25"]), "p75": _int_or_none(r["p75"])}
            for _, r in enumerate(price_trend)
        ],
    }


def listing_lifecycle(store: CanonicalStore) -> dict:
    """Honest listing-lifecycle overview (count-based, no premature marking):
    how many listings were last observed in each collection run. Absence from
    one rotating/partial feed is NOT treated as disappearance."""
    df = store.query_df(
        """
        SELECT l.listing_key, l.first_seen_at,
               MAX(substr(po.observed_at, 1, 10)) AS last_observed
        FROM listings l
        LEFT JOIN price_observations po ON po.listing_key = l.listing_key
        GROUP BY l.listing_key
        """
    )
    by_day = df.groupby("last_observed").size().sort_index(ascending=False)
    return {
        "total": len(df),
        "last_observed_distribution": [
            {"day": str(day), "count": int(count)} for day, count in by_day.items()
        ],
        "note": (
            "حذف شدن از یک فیدِ چرخشی به معنی فروش یا حذف آگهی نیست — وضعیت "
            "چرخهٔ حیات فقط پس از جمع‌آوری‌های کامل و مکرر معنادار می‌شود."
        ),
    }
