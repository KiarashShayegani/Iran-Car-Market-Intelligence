"""
Database abstraction layer.
Uses SQLite locally (single file, zero setup).
Also exports to CSV for easy inspection.

CHANGELOG (this patch):
  - upsert_listings() used to do `INSERT OR REPLACE INTO listings
    SELECT * FROM temp_listings`, which silently assumes the incoming
    dataframe has exactly the same columns, in the same order, as the
    `listings` table (including the DB-managed `created_at` default).
    It now builds an explicit, matched column list from the table's
    actual schema (via PRAGMA table_info) intersected with whatever
    columns are present in the dataframe, so it's correct regardless
    of column order or a partially-different column set.
"""

import sqlite3
from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger


class CarDatabase:
    """
    SQLite database for car listings with CSV export.

    All writes are validated before execution.
    The database file lives in data/ and can be git-ignored.
    """

    def __init__(self, db_path: str = "data/icmi.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        """Create connection with proper settings."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_tables(self) -> None:
        """Initialize schema if not exists."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS listings (
                    listing_id TEXT PRIMARY KEY,
                    brand_slug TEXT NOT NULL,
                    brand_name_fa TEXT,
                    manufacturer TEXT,
                    name TEXT NOT NULL,
                    model TEXT,
                    trim TEXT,
                    year INTEGER NOT NULL,
                    mileage REAL,
                    mileage_unknown INTEGER DEFAULT 0,
                    fuel TEXT,
                    transmission TEXT,
                    body_status TEXT,
                    body_status_ordinal INTEGER,
                    price INTEGER NOT NULL,
                    scraped_at TEXT,
                    source_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_brand ON listings(brand_slug)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_year ON listings(year)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_price ON listings(price)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scraped ON listings(scraped_at)
            """)

            conn.commit()
        logger.debug("Database initialized: {}", self.db_path)

    def upsert_listings(self, df: pd.DataFrame) -> int:
        """
        Insert or replace listings.
        Uses listing_id as primary key for deduplication.

        Args:
            df: DataFrame with cleaned listings.

        Returns:
            Number of rows written.
        """
        if df.empty:
            logger.warning("Empty DataFrame, nothing to upsert")
            return 0

        # Ensure required columns exist
        required = ["listing_id", "brand_slug", "name", "year", "price"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        with self._get_connection() as conn:
            # Build the write column list from the ACTUAL table schema,
            # intersected with what the dataframe actually has. This
            # keeps the insert correct even if `df` gains/loses columns
            # or the table's column order changes - no positional
            # `SELECT *` assumptions.
            table_cols = [
                row[1]
                for row in conn.execute("PRAGMA table_info(listings)").fetchall()
                if row[1] != "created_at"  # DB-managed default, never supplied by us
            ]
            available = [c for c in table_cols if c in df.columns]
            missing_for_table = [c for c in table_cols if c not in df.columns]
            if missing_for_table:
                logger.warning(
                    "Columns expected by 'listings' but absent from data "
                    "(will be NULL): {}",
                    missing_for_table,
                )

            write_df = df[available].copy()
            write_df.to_sql("temp_listings", conn, if_exists="replace", index=False)

            col_list = ", ".join(available)
            conn.execute(f"""
                INSERT OR REPLACE INTO listings ({col_list})
                SELECT {col_list} FROM temp_listings
                WHERE listing_id IS NOT NULL
                  AND price > 0
                  AND year BETWEEN 1340 AND 1410
            """)

            conn.execute("DROP TABLE temp_listings")
            conn.commit()

            cursor = conn.execute("SELECT changes()")
            changes = cursor.fetchone()[0]

        logger.info("Upserted {} listings to database", changes)
        return changes

    def get_all(self, brand: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve listings, optionally filtered by brand.

        Args:
            brand: Brand slug to filter by.

        Returns:
            DataFrame of listings.
        """
        query = "SELECT * FROM listings"
        params = ()

        if brand:
            query += " WHERE brand_slug = ?"
            params = (brand,)

        query += " ORDER BY scraped_at DESC"

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        return df

    def get_brand_stats(self) -> pd.DataFrame:
        """Get aggregate statistics per brand."""
        query = """
            SELECT
                brand_slug,
                brand_name_fa,
                COUNT(*) as listing_count,
                ROUND(AVG(price), 0) as avg_price,
                ROUND(AVG(year), 0) as avg_year,
                MIN(price) as min_price,
                MAX(price) as max_price
            FROM listings
            GROUP BY brand_slug
            ORDER BY avg_price DESC
        """
        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn)

    def export_to_csv(self, output_path: str = "data/export/listings.csv") -> str:
        """
        Export all listings to CSV for easy inspection.

        Args:
            output_path: Target CSV path.

        Returns:
            Path to exported file.
        """
        df = self.get_all()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info("Exported {} rows to {}", len(df), output_path)
        return output_path

    def get_distinct_values(
        self, column: str, brand: Optional[str] = None
    ) -> List[str]:
        """
        Get distinct values for a column (useful for dropdowns).

        Args:
            column: Column name.
            brand: Optional brand filter.

        Returns:
            List of distinct values.
        """
        query = f"""
            SELECT DISTINCT {column}
            FROM listings
            WHERE {column} IS NOT NULL
        """
        params = ()

        if brand:
            query += " AND brand_slug = ?"
            params = (brand,)

        query += f" ORDER BY {column}"

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        return df[column].tolist()
