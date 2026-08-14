#!/usr/bin/env python3
"""
ICMI v2.0 - End-to-End Pipeline Orchestrator
Scrape → Validate → Clean → Save → Train

Usage:
    python run_pipeline.py

This script runs the full ICMI pipeline:
  1. Scrape all enabled brands from bama.ir
  2. Validate raw data against schema
  3. Clean and engineer features
  4. Save to SQLite database + CSV export
  5. Train unified CatBoost model

All steps are logged to logs/ and console.
"""

import sys
from pathlib import Path
import pandas as pd
from loguru import logger

from src.utils import setup_logging
from src.scraper import MultiBrandScraper
from src.cleaner import process_raw_file
from src.database import CarDatabase
from src.trainer import train_pipeline


def main() -> None:
    """Run the full ICMI pipeline."""
    setup_logging()

    logger.info("=" * 60)
    logger.info("ICMI v2.0 Pipeline Starting")
    logger.info("=" * 60)

    # Step 1: Scrape
    logger.info("STEP 1/4: Scraping data from bama.ir")
    scraper = MultiBrandScraper()
    results = scraper.scrape_all(max_pages_per_brand=10)

    if not results:
        logger.error("No data scraped. Aborting pipeline.")
        sys.exit(1)

    # Step 2: Process (validate + clean)
    logger.info("STEP 2/4: Validating and cleaning data")
    raw_path = "data/raw/combined_latest.parquet"
    if not Path(raw_path).exists():
        logger.error("Combined raw file not found: {}", raw_path)
        sys.exit(1)

    processed_path = process_raw_file(raw_path)

    # Step 3: Save to database
    logger.info("STEP 3/4: Saving to database")
    db = CarDatabase()
    df = pd.read_parquet(processed_path)
    db.upsert_listings(df)
    db.export_to_csv()

    # Step 4: Train model
    logger.info("STEP 4/4: Training model")
    try:
        artifacts = train_pipeline(processed_path)
        logger.info("Model saved: {}", artifacts)
    except Exception as e:
        logger.error("Training failed: {}", e)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    import pandas as pd  # Imported here to avoid circular issues

    main()
