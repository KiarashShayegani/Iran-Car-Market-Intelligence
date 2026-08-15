"""
Data validation using Pandera.

Design note: a handful of corrupted listings (bad price parse, etc.)
should never abort an entire day's pipeline run. validate_raw() and
validate_processed() now COERCE dtypes automatically and DROP only the
specific rows that fail a data-range check, logging why. If pandera
reports a failure with no row index (a whole-column dtype/shape
problem coercion couldn't fix), that's a real upstream bug and we
still raise.
"""

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
            "name": Column(str, nullable=False),
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
        },
        strict="filter",
        coerce=True,
    )


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

        cleaned = df.drop(index=[i for i in bad_indices if i in df.index])
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
