"""Retrieval-based price estimation (replaces ML prediction, owner decision).

Approach: find COMPARABLE listings in the canonical layer — same brand (and
optionally model), similar year (calendar-aware: shamsi years are only
compared with shamsi years), optionally similar mileage — and report the
observed market range (p25–p75) and median of their stated prices.

Design rules inherited from the project:
- Only listings with a stated, normalized price participate (negotiable/0
  prices are already NULL — they can never contaminate the estimate).
- The estimate is OBSERVED DATA (percentiles of real ads), clearly
  distinguished from any derived opinion; the response carries the sample
  size and the comparables themselves so the user can audit every input.
- Small samples are flagged honestly (quality: "low_sample") — never
  silently presented as a confident range.
"""

from __future__ import annotations

from dataclasses import dataclass

from icmi.pipeline.normalize import classify_year_calendar, normalize_year
from icmi.pipeline.store import CanonicalStore
from icmi.retrieval.normalization import normalize_for_key

YEAR_TOLERANCE = 2  # ±2 model-years for comparables
MILEAGE_BAND_KM = 30_000  # comparable mileage band around the target


@dataclass
class Comparable:
    listing_key: str
    source_platform: str
    title: str | None
    brand_raw: str | None
    model_raw: str | None
    year_num: int | None
    year_calendar: str | None
    mileage_km: int | None
    price_amount: int
    original_url: str | None


@dataclass
class EstimateResult:
    brand_key: str
    model_key: str | None
    year_num: int | None
    year_calendar: str | None
    mileage_km: int | None
    count: int
    median: int | None
    p25: int | None
    p75: int | None
    min: int | None
    max: int | None
    quality: str  # "ok" | "low_sample" | "no_data"
    quality_note: str
    comparables: list[Comparable]


def estimate_market_range(
    store: CanonicalStore,
    brand: str,
    model: str | None = None,
    year: int | str | None = None,
    mileage_km: int | None = None,
    max_comparables: int = 8,
    trim_outliers: bool = False,
) -> EstimateResult:
    """Estimate the observed market range for a vehicle description.

    All criteria are AND-ed; with no model given, the whole brand is the pool.
    """
    brand_key = normalize_for_key(brand)
    if not brand_key:
        raise ValueError("brand is required")
    model_key = normalize_for_key(model) if model and model.strip() else None
    year_num = normalize_year(year)
    calendar = None
    if year_num is not None:
        calendar = classify_year_calendar(year_num)
        if calendar == "unknown":
            # never mix calendars on an unclassifiable year — refuse honestly
            return EstimateResult(
                brand_key, model_key, year_num, "unknown", mileage_km,
                0, None, None, None, None, None,
                "no_data",
                "سال قابل‌طبقه‌بندی نیست (نه شمسی و نه میلادی) — مقایسه تقویم‌ها ممنوع است.",
                [],
            )

    df = store.query_df(
        """
        SELECT listing_key, source_platform, title, brand_raw, model_raw,
               year_num, year_calendar, mileage_km, price_amount, original_url
        FROM listings
        WHERE price_amount IS NOT NULL AND price_amount > 0
        """
    )
    df["brand_key"] = df["brand_raw"].map(lambda v: normalize_for_key(v) or None)
    pool = df[df["brand_key"] == brand_key]
    if model_key:
        pool = pool[pool["model_raw"].map(lambda v: normalize_for_key(v)) == model_key]

    if calendar:
        pool = pool[pool["year_calendar"] == calendar]
        if year_num is not None:
            pool = pool[
                pool["year_num"].notna()
                & (pool["year_num"] >= year_num - YEAR_TOLERANCE)
                & (pool["year_num"] <= year_num + YEAR_TOLERANCE)
            ]

    if mileage_km is not None:
        pool = pool[
            pool["mileage_km"].notna()
            & (pool["mileage_km"] <= mileage_km + MILEAGE_BAND_KM)
        ]

    def _int_or_none(value) -> int | None:
        if value is None:
            return None
        try:
            if value != value:  # NaN (pandas null in mixed int columns)
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    comparables = [
        Comparable(
            listing_key=str(row["listing_key"]),
            source_platform=str(row["source_platform"]),
            title=row["title"],
            brand_raw=row["brand_raw"],
            model_raw=row["model_raw"],
            year_num=_int_or_none(row["year_num"]),
            year_calendar=row["year_calendar"],
            mileage_km=_int_or_none(row["mileage_km"]),
            price_amount=int(row["price_amount"]),
            original_url=row["original_url"],
        )
        for _, row in pool.iterrows()
    ]
    # nearest by year first, then mileage — the audit trail is ordered
    comparables.sort(
        key=lambda c: (
            abs(c.year_num - year_num)
            if (year_num is not None and c.year_num is not None)
            else 99,
            abs(c.mileage_km - mileage_km)
            if (mileage_km is not None and c.mileage_km is not None)
            else 99,
            c.price_amount,
        )
    )
    comparables = comparables[:max_comparables]

    prices = sorted(int(row["price_amount"]) for _, row in pool.iterrows())
    count = len(prices)
    if count == 0:
        return EstimateResult(
            brand_key, model_key, year_num, calendar, mileage_km,
            0, None, None, None, None, None,
            "no_data",
            "هیچ آگهی قابل‌مقایسه‌ای با این مشخصات در داده‌های فعلی نیست.",
            comparables,
        )

    def percentile(p: float) -> int:
        idx = min(int(round(p * (count - 1))), count - 1)
        return prices[idx]

    if trim_outliers and count >= 6:
        # IQR-based trimming inside the comparable pool (documented,
        # optional): at least 6 samples required so trimming never eats a
        # small honest sample.
        q1, q3 = percentile(0.25), percentile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        trimmed = [p for p in prices if lo <= p <= hi]
        if len(trimmed) >= 4:  # never trim into meaninglessness
            prices = trimmed
            count = len(prices)

    median = percentile(0.5)
    p25, p75 = percentile(0.25), percentile(0.75)
    quality = "ok" if count >= 5 else "low_sample"
    note = (
        "بر پایه قیمت‌های واقعی آگهی‌های مشابه؛ برای اطمینان بیشتر، نمونه‌ها را ببینید."
        if quality == "ok"
        else f"نمونه کوچک ({count} آگهی) — این بازه تردید بالایی دارد."
    )

    return EstimateResult(
        brand_key=brand_key,
        model_key=model_key,
        year_num=year_num,
        year_calendar=calendar,
        mileage_km=mileage_km,
        count=count,
        median=median,
        p25=p25,
        p75=p75,
        min=prices[0],
        max=prices[-1],
        quality=quality,
        quality_note=note,
        comparables=comparables,
    )
