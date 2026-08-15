"""
Data validation using Pandera.

Design note: a handful of corrupted listings (bad price parse, etc.)
should never abort an entire day's pipeline run. validate_raw() and
validate_processed() now COERCE dtypes automatically and DROP only the
specific rows that fail a data-range check, logging why. If pandera
reports a failure with no row index (a whole-column dtype/shape
problem coercion couldn't fix), that's a real upstream bug and we
still raise.

CHANGELOG (this patch):
  - build_raw_schema() and build_processed_schema() were missing
    several columns the scraper/cleaner actually produce (name, trim,
    model, brand_name_fa, manufacturer, source_url, scraped_at). With
    strict="filter", any column not listed in the schema is silently
    dropped - which meant validate_raw() was stripping name/trim right
    after scraping, and validate_processed() then crashed because it
    required those same columns to be present. Both schemas now list
    every column produced upstream.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml
from loguru import logger

try:
    import pandera.pandas as pa  # pandera >= 0.20
except ImportError:  # pragma: no cover - older pandera versions
    import pandera as pa

from pandera import Check, Column, DataFrameSchema


VALID_BRANDS = [
    "pride", "peugeot", "tara", "shahin", "fownix",
    "reera", "quick", "saina", "dena", "lucano",
    "toyota", "lamari", "kia", "kmc",
]


def _load_validation_config(config_path: str = "config/brands.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("validation", {})


def build_raw_schema(cfg: Optional[Dict[str, Any]] = None) -> DataFrameSchema:
    """Raw schema: what the scraper produces, before cleaning."""
    cfg = cfg or {}
    price_min = cfg.get("price_min", 3_000_000)
    price_max = cfg.get("price_max", 200_000_000_000)
    year_min = cfg.get("min_year", 1340)
    year_max = cfg.get("max_year", 1410)

    return DataFrameSchema(
        {
            "listing_id": Column(str, unique=True, nullable=False),
            "brand_slug": Column(str, Check.isin(VALID_BRANDS), nullable=False),
            "brand_name_fa": Column(str, nullable=True),
            "manufacturer": Column(str, nullable=True),
            "name": Column(str, nullable=False),
            "model": Column(str, nullable=True),
            "trim": Column(str, nullable=False),
            "year": Column(int, Check.in_range(year_min, year_max), nullable=False),
            "mileage": Column(
                float,
                Check.in_range(0, 2_000_000, include_min=True, include_max=True),
                nullable=True,
            ),
            "mileage_unknown": Column(bool, nullable=False),
            "price": Column(int, Check.in_range(price_min, price_max), nullable=False),
            "fuel": Column(
                str,
                Check.isin(["بنزینی", "دوگانه سوز", "پلاگین هیبرید", "برقی"]),
                nullable=False,
            ),
            "transmission": Column(
                str, Check.isin(["دنده ای", "اتوماتیک"]), nullable=False
            ),
            "body_status": Column(str, nullable=False),
            "scraped_at": Column(pa.DateTime, nullable=False),
            "source_url": Column(str, nullable=True),
        },
        strict="filter",  # drop columns not in schema (e.g. first_seen_at)
        coerce=True,  # auto-fix dtype mismatches (e.g. int64 mileage -> float64)
    )


def build_processed_schema(cfg: Optional[Dict[str, Any]] = None) -> DataFrameSchema:
    """Processed schema: after cleaning, right before training."""
    cfg = cfg or {}
    price_min = cfg.get("price_min", 3_000_000)
    price_max = cfg.get("price_max", 200_000_000_000)
    year_min = cfg.get("min_year", 1340)
    year_max = cfg.get("max_year", 1410)

    return DataFrameSchema(
        {
            "listing_id": Column(str, nullable=False),
            "brand_slug": Column(str, nullable=False),
            "brand_name_fa": Column(str, nullable=True),
            "manufacturer": Column(str, nullable=True),
            "name": Column(str, nullable=False),
            "model": Column(str, nullable=True),
            "trim": Column(str, nullable=False),
            "year": Column(int, Check.in_range(year_min, year_max), nullable=False),
            "age": Column(int, Check.in_range(0, 60), nullable=False),
            "mileage": Column(float, Check.in_range(0, 2_000_000), nullable=False),
            "mileage_unknown": Column(int, Check.isin([0, 1]), nullable=False),
            "fuel": Column(str, nullable=False),
            "transmission": Column(str, nullable=False),
            "body_status": Column(str, nullable=False),
            "body_status_ordinal": Column(int, Check.in_range(1, 15), nullable=False),
            "price": Column(int, Check.in_range(price_min, price_max), nullable=False),
            "scraped_at": Column(pa.DateTime, nullable=False),
            "source_url": Column(str, nullable=True),
        },
        strict="filter",
        coerce=True,
    )


def _save_dropped_rows(
    df: pd.DataFrame, path: str = "data/validation_dropped_rows.csv"
) -> None:
    """
    Append the actual dropped rows (with why they failed) to a shared
    diagnostic CSV - the same philosophy as the scraper's
    data/skipped_rows.csv, one stage downstream. Without this, a
    message like "dropping 357 rows | reasons={isin(...): 357}" tells
    you a fuel value was unexpected but not WHICH one.
    """
    if df.empty:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_header = not p.exists()
    df.to_csv(p, mode="a", header=write_header, index=False, encoding="utf-8-sig")
    logger.info("Logged {} dropped rows -> {}", len(df), path)


def _validate_non_fatal(
    df: pd.DataFrame, schema: DataFrameSchema, label: str
) -> pd.DataFrame:
    """
    Validate; on failure, drop only the offending ROWS and continue.

    dtype mismatches are handled by `coerce=True` in the schema itself
    and shouldn't surface here. If pandera reports a failure with no
    row index at all (a whole-column/shape problem coercion couldn't
    fix), that indicates a real upstream bug, so we re-raise instead
    of silently guessing.
    """
    logger.info("Validating {} data | shape={}", label, df.shape)

    try:
        validated = schema.validate(df, lazy=True)
        logger.info("{} validation passed | {} records", label, len(validated))
        return validated
    except pa.errors.SchemaErrors as err:
        failures = err.failure_cases
        bad_indices = set(failures["index"].dropna().astype(int))

        if not bad_indices:
            logger.error(
                "{} validation failed with no recoverable row-level errors", label
            )
            logger.error("Failure cases:\n{}", failures.head(20))
            raise

        reasons = failures["check"].value_counts().to_dict()
        logger.warning(
            "{} validation: dropping {} of {} rows | reasons={}",
            label,
            len(bad_indices),
            len(df),
            reasons,
        )

        present = [i for i in bad_indices if i in df.index]

        # Save the actual offending rows (with which check they failed)
        # so "357 rows failed a fuel check" becomes something you can
        # actually open and read, not just a count.
        reason_by_index = (
            failures.dropna(subset=["index"])
            .astype({"index": int})
            .groupby("index")["check"]
            .apply(lambda s: "; ".join(sorted(set(s.astype(str)))))
        )
        dropped_rows = df.loc[present].copy()
        dropped_rows["validation_stage"] = label
        dropped_rows["failed_check"] = dropped_rows.index.map(reason_by_index)
        _save_dropped_rows(dropped_rows)

        cleaned = df.drop(index=present)
        # Re-validate the cleaned frame - if it still fails, something
        # else is wrong and that should surface loudly, not be masked.
        return schema.validate(cleaned, lazy=True)


def validate_raw(df: pd.DataFrame, config_path: str = "config/brands.yaml") -> pd.DataFrame:
    """
    Validate raw scraped data against schema.

    Args:
        df: Raw DataFrame from scraper.
        config_path: Path to brands.yaml, for price/year bounds.

    Returns:
        Validated (and possibly row-filtered) DataFrame.
    """
    cfg = _load_validation_config(config_path)
    schema = build_raw_schema(cfg)
    return _validate_non_fatal(df, schema, "raw")


def validate_processed(
    df: pd.DataFrame, config_path: str = "config/brands.yaml"
) -> pd.DataFrame:
    """
    Validate cleaned data before training or saving.

    Args:
        df: Cleaned DataFrame.
        config_path: Path to brands.yaml, for price/year bounds.

    Returns:
        Validated (and possibly row-filtered) DataFrame.
    """
    cfg = _load_validation_config(config_path)
    schema = build_processed_schema(cfg)
    return _validate_non_fatal(df, schema, "processed")


def get_validation_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a non-blocking validation report.

    Args:
        df: DataFrame to check.

    Returns:
        Dictionary with pass/fail status and statistics.
    """
    report = {
        "total_rows": len(df),
        "null_counts": df.isnull().sum().to_dict(),
        "price_stats": {
            "min": int(df["price"].min()) if "price" in df.columns else None,
            "max": int(df["price"].max()) if "price" in df.columns else None,
            "mean": float(df["price"].mean()) if "price" in df.columns else None,
        },
        "brands": (
            df["brand_slug"].value_counts().to_dict()
            if "brand_slug" in df.columns
            else {}
        ),
        "year_range": {
            "min": int(df["year"].min()) if "year" in df.columns else None,
            "max": int(df["year"].max()) if "year" in df.columns else None,
        },
    }
    return report
