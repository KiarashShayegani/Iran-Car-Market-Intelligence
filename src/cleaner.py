"""
Data cleaning and feature engineering pipeline.
NO target leakage. NO price-based encoding.

Preprocessing steps:
  1. String cleaning (strip whitespace)
  2. Outlier removal (IQR method, 2.0 * IQR), computed PER BRAND
  3. Mileage imputation (year-group medians, with flag preserved)
  4. Feature engineering:
     - age = current_year - year
     - body_status_ordinal (from hierarchy mapping)
     - fuel_en / transmission_en (normalized labels)
  5. Type coercion and final validation
"""

from pathlib import Path
from typing import Dict

import pandas as pd
import yaml
from loguru import logger


class DataCleaner:
    """
    Idempotent cleaning pipeline for car listing data.

    Transforms raw scraped data into ML-ready features without
    using target variable (price) for any encoding.
    """

    def __init__(self, config_path: str = "config/brands.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.current_year = self.config.get("pipeline", {}).get(
            "current_year_shamsi", 1404
        )

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main cleaning pipeline.

        Steps:
            1. String cleaning
            2. Outlier removal (price, mileage), per brand
            3. Mileage imputation with year-group medians
            4. Feature engineering (age, body_status_ordinal)
            5. Categorical normalization

        Args:
            df: Raw validated DataFrame.

        Returns:
            Cleaned DataFrame ready for ML.
        """
        df = df.copy()
        initial_rows = len(df)

        logger.info("Starting cleaning | {} rows", initial_rows)

        # 1. Clean strings
        df = self._clean_strings(df)

        # 2. Remove outliers
        df = self._remove_outliers(df)

        # 3. Impute missing mileage
        df = self._impute_mileage(df)

        # 4. Feature engineering
        df["age"] = self.current_year - df["year"]
        df["body_status_ordinal"] = (
            df["body_status"]
            .map(self.config.get("body_status_hierarchy", {}))
            .fillna(8)
            .astype(int)
        )

        # 5. Normalize categoricals
        df["fuel_en"] = df["fuel"].map(self.config.get("fuel_mapping", {}))
        df["transmission_en"] = df["transmission"].map(
            self.config.get("transmission_mapping", {})
        )

        # 6. Ensure numeric types
        df["mileage_unknown"] = df["mileage_unknown"].astype(int)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna(subset=["price"])

        final_rows = len(df)
        logger.info(
            "Cleaning complete | {} -> {} rows ({} removed)",
            initial_rows,
            final_rows,
            initial_rows - final_rows,
        )

        return df

    def _clean_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strip whitespace from all string columns."""
        str_cols = df.select_dtypes(include=["object"]).columns
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip()
        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove statistical outliers using IQR method (2.0 * IQR),
        computed PER BRAND when possible.

        A single global price band doesn't make sense across brands
        that span cheap economy cars and expensive imports - grouping
        avoids clipping a legitimately expensive Toyota just because
        it looks like an outlier next to a Pride.
        """
        group_col = "brand_slug" if "brand_slug" in df.columns else None

        for col in ["price", "mileage"]:
            if col not in df.columns or df[col].isna().all():
                continue

            before = len(df)

            if group_col:
                keep_mask = pd.Series(True, index=df.index)
                for _, group_idx in df.groupby(group_col).groups.items():
                    valid = df.loc[group_idx, col].dropna()
                    if len(valid) < 10:
                        continue
                    q1 = valid.quantile(0.25)
                    q3 = valid.quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 2.0 * iqr
                    upper = q3 + 2.0 * iqr
                    in_bounds = df.loc[group_idx, col].isna() | df.loc[
                        group_idx, col
                    ].between(lower, upper)
                    keep_mask.loc[group_idx] &= in_bounds
                df = df[keep_mask]
            else:
                valid = df[col].dropna()
                if len(valid) < 10:
                    continue
                q1 = valid.quantile(0.25)
                q3 = valid.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 2.0 * iqr
                upper = q3 + 2.0 * iqr
                df = df[
                    (df[col].isna()) | ((df[col] >= lower) & (df[col] <= upper))
                ]

            removed = before - len(df)
            logger.info("Outlier removal ({}) | {} removed", col, removed)

        return df

    def _impute_mileage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing mileage using year-group medians.
        Preserves mileage_unknown flag for ML model.
        """
        if "mileage_unknown" not in df.columns:
            df["mileage_unknown"] = df["mileage"].isna()

        missing_count = df["mileage"].isna().sum()
        if missing_count == 0:
            return df

        logger.info("Imputing {} missing mileage values", missing_count)

        def year_group(year: int) -> str:
            if year <= 1380:
                return "old"
            elif year <= 1395:
                return "mid"
            else:
                return "new"

        df["_year_group"] = df["year"].apply(year_group)

        known = df[df["mileage"].notna()]
        if len(known) > 0:
            medians = known.groupby("_year_group")["mileage"].median()
            overall_median = known["mileage"].median()
        else:
            medians = pd.Series({"old": 100000, "mid": 80000, "new": 30000})
            overall_median = 80000

        logger.info("Imputation medians: {}", medians.to_dict())

        for group, median in medians.items():
            mask = (df["_year_group"] == group) & (df["mileage"].isna())
            df.loc[mask, "mileage"] = median

        df["mileage"] = df["mileage"].fillna(overall_median)
        df = df.drop(columns=["_year_group"])

        return df


def process_raw_file(
    raw_path: str, output_dir: str = "data/processed"
) -> str:
    """
    Process a raw Parquet file through the cleaning pipeline.

    Args:
        raw_path: Path to raw Parquet file (ideally the cumulative
            master history, so training data grows over time).
        output_dir: Directory for cleaned output.

    Returns:
        Path to cleaned Parquet file.
    """
    from src.validator import validate_raw, validate_processed

    logger.info("Processing raw file: {}", raw_path)

    # Load
    df = pd.read_parquet(raw_path)

    # Validate raw (drops corrupted rows, doesn't crash the pipeline)
    df = validate_raw(df)

    # Clean
    cleaner = DataCleaner()
    df = cleaner.clean(df)

    # Validate processed
    df = validate_processed(df)

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "processed_latest.parquet"
    df.to_parquet(output_path, index=False)

    logger.info("Saved cleaned data: {} ({} rows)", output_path, len(df))
    return str(output_path)
