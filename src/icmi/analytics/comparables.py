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
    source_platform: str | None = None,
    trim: str | None = None,
) -> EstimateResult:
    """Estimate the observed market range for a vehicle description.

    All criteria are AND-ed; with no model given, the whole brand is the pool.
    `source_platform` restricts the comparable pool to one source (the
    Bama-structured estimate uses source_platform="bama"). `trim` narrows
    bama pools further via contains-match on model_raw (the trim composite
    «1405 | دنده ای TU5P» carries the real trim after its year prefix).
    """
    brand_key = normalize_for_key(brand)
    if not brand_key:
        raise ValueError("brand is required")
    model_key = normalize_for_key(model) if model and model.strip() else None
    trim_key = normalize_for_key(trim) if trim and trim.strip() else None
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

    MIN_PLAUSIBLE_PRICE = 20_000_000  # below this, "prices" are parse
    # artifacts (e.g. installment amounts) — never comparable evidence
    df = store.query_df(
        f"""
        SELECT listing_key, source_platform, title, brand_raw, model_raw,
               year_num, year_calendar, mileage_km, price_amount, original_url
        FROM listings
        WHERE price_amount IS NOT NULL AND price_amount >= {MIN_PLAUSIBLE_PRICE}
        """
    )
    df["brand_key"] = df["brand_raw"].map(lambda v: normalize_for_key(v) or None)
    if source_platform:
        df = df[df["source_platform"] == source_platform]
    pool = df[df["brand_key"] == brand_key]
    if model_key:
        # contains-match on the normalized TITLE: bama's model_raw is a
        # trim/variant description (year-prefixed) — the model name lives in
        # the title («پژو، 206 | …»). Contains semantics also cover free-text
        # model references from the agentic/guided flows.
        pool = pool[pool["title"].map(
            lambda v: bool(v) and model_key in normalize_for_key(str(v))
        )]
    if trim_key:
        # the trim composite contains the real trim after its year prefix —
        # contains-match survives both («1405 | دنده ای TU5P» ⊃ «دنده ای tu5p»)
        pool = pool[pool["model_raw"].map(
            lambda v: bool(v) and trim_key in normalize_for_key(str(v))
        )]

    if calendar:
        pool = pool[pool["year_calendar"] == calendar]
        if year_num is not None:
            pool = pool[
                pool["year_num"].notna()
                & (pool["year_num"] >= year_num - YEAR_TOLERANCE)
                & (pool["year_num"] <= year_num + YEAR_TOLERANCE)
            ]

    mileage_note: str | None = None
    if mileage_km is not None:
        # adaptive band: ±MILEAGE_BAND_KM first; widen symmetrically until the
        # pool holds enough comparables (sparse combos must not show empty)
        for band in (MILEAGE_BAND_KM, 60_000, 120_000):
            candidate = pool[
                pool["mileage_km"].notna()
                & (pool["mileage_km"] >= mileage_km - band)
                & (pool["mileage_km"] <= mileage_km + band)
            ]
            if len(candidate) >= 4 or band == 120_000:
                pool = candidate
                if band > MILEAGE_BAND_KM:
                    mileage_note = (
                        f"بازهٔ کارکرد برای رسیدن به نمونهٔ کافی تا ±{band // 1000} "
                        "هزار کیلومتر گسترش داده شد."
                    )
                break

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
    if mileage_note:
        note += f" {mileage_note}"

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
