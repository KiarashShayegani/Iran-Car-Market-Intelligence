"""
Data validation using Pandera.
Data MUST pass these checks before being saved to CSV or database.
"""

from typing import Any, Dict

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema
from loguru import logger


# --- Raw data schema: what the scraper produces ---
RAW_SCHEMA = DataFrameSchema(
    {
        "listing_id": Column(str, unique=True, nullable=False),
        "brand_slug": Column(
            str,
            Check.isin([
                "pride", "peugeot", "tara", "shahin", "fownix",
                "reera", "quick", "saina", "dena", "lucano",
                "toyota", "lamari", "kia", "kmc",
            ]),
            nullable=False,
        ),
        "year": Column(int, Check.in_range(1340, 1410), nullable=False),
        "mileage": Column(
            float,
            Check.in_range(0, 2_000_000, include_min=True, include_max=True),
            nullable=True,
        ),
        "mileage_unknown": Column(bool, nullable=False),
        "price": Column(
            int,
            Check.in_range(10_000_000, 50_000_000_000),
            nullable=False,
        ),
        "fuel": Column(
            str,
            Check.isin(["بنزینی", "دوگانه سوز", "پلاگین هیبرید", "برقی"]),
            nullable=False,
        ),
        "transmission": Column(
            str,
            Check.isin(["دنده ای", "اتوماتیک"]),
            nullable=False,
        ),
        "body_status": Column(str, nullable=False),
        "scraped_at": Column(pa.DateTime, nullable=False),
    },
    strict="filter",  # Drop columns not in schema
)


# --- Processed data schema: after cleaning, before ML ---
PROCESSED_SCHEMA = DataFrameSchema(
    {
        "listing_id": Column(str, nullable=False),
        "brand_slug": Column(str, nullable=False),
        "name": Column(str, nullable=False),
        "trim": Column(str, nullable=False),
        "year": Column(int, Check.in_range(1340, 1410), nullable=False),
        "age": Column(int, Check.in_range(0, 60), nullable=False),
        "mileage": Column(float, Check.in_range(0, 2_000_000), nullable=False),
        "mileage_unknown": Column(int, Check.isin([0, 1]), nullable=False),
        "fuel": Column(str, nullable=False),
        "transmission": Column(str, nullable=False),
        "body_status": Column(str, nullable=False),
        "body_status_ordinal": Column(int, Check.in_range(1, 15), nullable=False),
        "price": Column(
            int,
            Check.in_range(10_000_000, 50_000_000_000),
            nullable=False,
        ),
    },
    strict="filter",
)


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate raw scraped data against schema.

    Args:
        df: Raw DataFrame from scraper.

    Returns:
        Validated DataFrame.

    Raises:
        pa.errors.SchemaErrors: If validation fails with details.
    """
    logger.info("Validating raw data | shape={}", df.shape)

    try:
        validated = RAW_SCHEMA.validate(df, lazy=True)
        logger.info("Raw validation passed | {} records", len(validated))
        return validated
    except pa.errors.SchemaErrors as err:
        logger.error(
            "Raw validation failed | {} failure cases",
            len(err.failure_cases),
        )
        logger.error("Failure cases:\n{}", err.failure_cases.head(20))
        raise


def validate_processed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate cleaned data before training or saving.

    Args:
        df: Cleaned DataFrame.

    Returns:
        Validated DataFrame.

    Raises:
        pa.errors.SchemaErrors: If validation fails.
    """
    logger.info("Validating processed data | shape={}", df.shape)

    try:
        validated = PROCESSED_SCHEMA.validate(df, lazy=True)
        logger.info("Processed validation passed | {} records", len(validated))
        return validated
    except pa.errors.SchemaErrors as err:
        logger.error(
            "Processed validation failed | {} failure cases",
            len(err.failure_cases),
        )
        raise


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
