"""Canonical SQLite store: listings, append-only price observations, runs.

Cross-run identity: a listing is keyed by ``<source_platform>:<source_listing_id>``
— the same advertisement collected in different runs maps to the same row.

Price history rules (deliberate, tested):
- Observations store the raw value verbatim, the currency, and an explicitly
  normalized integer amount (with PRICE_NORMALIZATION_VERSION).
- A new observation is appended ONLY when there is no previous observation, or
  the normalized amount changed, or the currency changed. Formatting-only
  changes ("7,100,000,000" → 7100000000) are NOT price changes and append
  nothing.
- A missing price on a later sighting never deletes or overwrites history;
  it simply produces no observation.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

# retrieval.normalization is a leaf utility (no project imports) — the
# matching-key function lives there and is safe to use from the pipeline.
from icmi.core.brand_aliases import canonical_brand
from icmi.core.paths import canonical_dir
from icmi.core.records import CanonicalRecord, has_value
from icmi.pipeline.normalize import (
    PRICE_NORMALIZATION_VERSION,
    classify_year_calendar,
    normalize_mileage_km,
    normalize_price_amount,
    normalize_year,
)
from icmi.retrieval.normalization import normalize_for_key


@dataclass
class IngestStats:
    new_listings: int = 0
    updated_listings: int = 0
    price_observations_appended: int = 0
    price_change_events: int = 0  # observations appended due to a real change
    records_without_id: int = 0
    skipped_runs: int = 0
    details: list[str] = field(default_factory=list)


class CanonicalStore:
    """SQLite-backed canonical layer (single file, portable, offline)."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or canonical_dir() / "icmi.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: sync API frameworks run endpoints in a
        # threadpool. Safe at this scale (short serialized statements, single
        # writer in practice); a lock is the upgrade path if concurrency grows.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                platform TEXT NOT NULL,
                execution_mode TEXT,
                collection TEXT,
                status TEXT,
                started_at TEXT,
                completed_at TEXT,
                extraction_version TEXT,
                manifest_json TEXT,
                ingested_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS listings (
                listing_key TEXT PRIMARY KEY,      -- "<platform>:<source_listing_id>"
                source_platform TEXT NOT NULL,
                source_listing_id TEXT NOT NULL,
                original_url TEXT,
                title TEXT,
                description TEXT,
                manufacturer_raw TEXT,
                brand_raw TEXT,
                brand_en_raw TEXT,
                model_raw TEXT,
                trim_raw TEXT,
                year_raw TEXT,                     -- raw as delivered (str or int)
                mileage_raw TEXT,
                price_raw TEXT,                    -- latest observed raw price
                price_currency_raw TEXT,
                price_details_json TEXT,
                price_amount INTEGER,              -- latest normalized amount
                price_normalization_version TEXT,
                location_raw TEXT,
                fuel_raw TEXT,
                transmission_raw TEXT,
                body_type_raw TEXT,
                body_condition_raw TEXT,
                color_raw TEXT,
                seller_type TEXT,
                seller_id TEXT,
                seller_name_raw TEXT,
                image_urls_json TEXT,
                image_count INTEGER,
                source_published_at TEXT,
                source_modified_at TEXT,
                source_attributes_json TEXT,
                raw_payload_ref_json TEXT,
                extraction_errors_json TEXT,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                scraped_at TEXT,
                extraction_version TEXT,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_listings_platform ON listings(source_platform);
            CREATE INDEX IF NOT EXISTS idx_listings_year ON listings(year_raw);
            CREATE INDEX IF NOT EXISTS idx_listings_price ON listings(price_amount);
            CREATE TABLE IF NOT EXISTS price_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                listing_key TEXT NOT NULL REFERENCES listings(listing_key),
                run_id TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                price_raw TEXT,
                price_amount INTEGER,
                price_currency_raw TEXT,
                price_details_json TEXT,
                price_normalization_version TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_obs_listing ON price_observations(listing_key);
            CREATE INDEX IF NOT EXISTS idx_obs_run ON price_observations(run_id);
            """
        )
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Idempotent derived-column migration. New columns are populated on
        upsert; existing rows are backfilled by re-ingesting from raw."""
        existing = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(listings)")
        }
        additions = {
            "year_num": "INTEGER",
            "year_calendar": "TEXT",
            "mileage_km": "INTEGER",
            "brand_key": "TEXT",
            "model_key": "TEXT",
            "brand_canonical": "TEXT",
        }
        for name, col_type in additions.items():
            if name not in existing:
                self._conn.execute(
                    f"ALTER TABLE listings ADD COLUMN {name} {col_type}"
                )
        for name in ("year_calendar", "brand_key", "model_key"):
            self._conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_listings_{name} ON listings({name})"
            )

    # --- runs ----------------------------------------------------------------

    def record_run(self, manifest: dict, run_id: str, platform: str) -> None:
        self._conn.execute(
            """
            INSERT INTO runs (run_id, platform, execution_mode, collection, status,
                              started_at, completed_at, extraction_version,
                              manifest_json, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(run_id) DO UPDATE SET
                manifest_json=excluded.manifest_json,
                ingested_at=excluded.ingested_at
            """,
            (
                run_id,
                platform,
                manifest.get("execution_mode"),
                manifest.get("collection"),
                manifest.get("status"),
                manifest.get("started_at"),
                manifest.get("completed_at"),
                manifest.get("extraction_version"),
                json.dumps(manifest, ensure_ascii=False, default=str),
            ),
        )
        self._conn.commit()

    # --- listings + price observations ----------------------------------------

    def upsert_listings(
        self, records: list[CanonicalRecord], run_id: str
    ) -> IngestStats:
        stats = IngestStats()
        for record in records:
            if not record.source_listing_id or not record.source_listing_id.strip():
                stats.records_without_id += 1
                continue
            key = f"{record.source_platform}:{record.source_listing_id}"
            observed_at = record.scraped_at.isoformat(timespec="seconds")

            amount = (
                normalize_price_amount(record.price_raw)
                if has_value(record.price_raw)
                else None
            )
            # A zero price means "no stated price" (negotiable) on these
            # marketplaces — it must never count as a comparable amount in
            # filters, stats, or history. Raw value stays preserved verbatim.
            if amount is not None and amount <= 0:
                amount = None
            currency = record.price_currency_raw

            existing = self._conn.execute(
                "SELECT listing_key FROM listings WHERE listing_key = ?", (key,)
            ).fetchone()
            is_new = existing is None

            self._conn.execute(
                """
                INSERT INTO listings (
                    listing_key, source_platform, source_listing_id, original_url,
                    title, description, manufacturer_raw, brand_raw, brand_en_raw,
                    model_raw, trim_raw, year_raw, mileage_raw, price_raw,
                    price_currency_raw, price_details_json, price_amount,
                    price_normalization_version, location_raw, fuel_raw,
                    transmission_raw, body_type_raw, body_condition_raw, color_raw,
                    seller_type, seller_id, seller_name_raw, image_urls_json,
                    image_count, source_published_at, source_modified_at,
                    source_attributes_json, raw_payload_ref_json,
                    extraction_errors_json, first_seen_at, last_seen_at,
                    scraped_at, extraction_version, updated_at,
                    year_num, year_calendar, mileage_km, brand_key, model_key,
                    brand_canonical
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT(listing_key) DO UPDATE SET
                    original_url=excluded.original_url,
                    title=excluded.title,
                    description=excluded.description,
                    manufacturer_raw=excluded.manufacturer_raw,
                    brand_raw=excluded.brand_raw,
                    brand_en_raw=excluded.brand_en_raw,
                    model_raw=excluded.model_raw,
                    trim_raw=excluded.trim_raw,
                    year_raw=excluded.year_raw,
                    mileage_raw=excluded.mileage_raw,
                    price_currency_raw=excluded.price_currency_raw,
                    price_details_json=excluded.price_details_json,
                    location_raw=excluded.location_raw,
                    fuel_raw=excluded.fuel_raw,
                    transmission_raw=excluded.transmission_raw,
                    body_type_raw=excluded.body_type_raw,
                    body_condition_raw=excluded.body_condition_raw,
                    color_raw=excluded.color_raw,
                    seller_type=excluded.seller_type,
                    seller_id=excluded.seller_id,
                    seller_name_raw=excluded.seller_name_raw,
                    image_urls_json=excluded.image_urls_json,
                    image_count=excluded.image_count,
                    source_published_at=excluded.source_published_at,
                    source_modified_at=excluded.source_modified_at,
                    source_attributes_json=excluded.source_attributes_json,
                    raw_payload_ref_json=excluded.raw_payload_ref_json,
                    extraction_errors_json=excluded.extraction_errors_json,
                    last_seen_at=excluded.last_seen_at,
                    scraped_at=excluded.scraped_at,
                    extraction_version=excluded.extraction_version,
                    year_num=excluded.year_num,
                    year_calendar=excluded.year_calendar,
                    mileage_km=excluded.mileage_km,
                    brand_key=excluded.brand_key,
                    model_key=excluded.model_key,
                    brand_canonical=excluded.brand_canonical,
                    updated_at=excluded.updated_at
                """,
                self._listing_params(
                    record, key, observed_at, amount, self._derive(record)
                ),
            )
            if is_new:
                stats.new_listings += 1
            else:
                stats.updated_listings += 1

            # Price observation: append ONLY when appropriately warranted.
            if has_value(record.price_raw):
                last = self._conn.execute(
                    """
                    SELECT price_amount, price_currency_raw FROM price_observations
                    WHERE listing_key = ? ORDER BY id DESC LIMIT 1
                    """,
                    (key,),
                ).fetchone()
                amount_changed = last is not None and last["price_amount"] != amount
                currency_changed = (
                    last is not None and last["price_currency_raw"] != currency
                )
                warranted = last is None or amount_changed or currency_changed
                if warranted:
                    self._conn.execute(
                        """
                        INSERT INTO price_observations (
                            listing_key, run_id, observed_at, price_raw,
                            price_amount, price_currency_raw, price_details_json,
                            price_normalization_version
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            key,
                            run_id,
                            observed_at,
                            str(record.price_raw),
                            amount,
                            currency,
                            json.dumps(record.price_details, ensure_ascii=False, default=str)
                            if record.price_details is not None
                            else None,
                            PRICE_NORMALIZATION_VERSION,
                        ),
                    )
                    stats.price_observations_appended += 1
                    if amount_changed:
                        stats.price_change_events += 1
                        stats.details.append(
                            f"price change {key}: {last['price_amount']} -> {amount}"
                        )

            # current price_* columns always reflect the latest observation
            self._refresh_current_price(key)

        self._conn.commit()
        return stats

    def _derive(self, record: CanonicalRecord) -> dict:
        """Derived matching/analytics fields (single derivation point at
        ingest). Raw record values are never modified."""
        year_num = normalize_year(record.year_raw)
        mileage_km = normalize_mileage_km(record.mileage_raw)
        if mileage_km is not None and mileage_km < 0:
            mileage_km = None
        brand_key, brand_canonical = canonical_brand(record.brand_raw)
        return {
            "year_num": year_num,
            "year_calendar": classify_year_calendar(year_num),
            "mileage_km": mileage_km,
            "brand_key": brand_key or None,
            "brand_canonical": brand_canonical or None,
            "model_key": normalize_for_key(record.model_raw) or None,
        }

    def _listing_params(
        self,
        record: CanonicalRecord,
        key: str,
        observed_at: str,
        amount: int | None,
        derived: dict,
    ) -> tuple:
        return (
            key,
            record.source_platform,
            record.source_listing_id,
            record.original_url,
            record.title,
            record.description,
            record.manufacturer_raw,
            record.brand_raw,
            record.brand_en_raw,
            record.model_raw,
            record.trim_raw,
            str(record.year_raw) if record.year_raw is not None else None,
            str(record.mileage_raw) if record.mileage_raw is not None else None,
            str(record.price_raw) if has_value(record.price_raw) else None,
            record.price_currency_raw,
            json.dumps(record.price_details, ensure_ascii=False, default=str)
            if record.price_details is not None
            else None,
            amount,
            PRICE_NORMALIZATION_VERSION if amount is not None else None,
            record.location_raw,
            record.fuel_raw,
            record.transmission_raw,
            record.body_type_raw,
            record.body_condition_raw,
            record.color_raw,
            record.seller_type,
            record.seller_id,
            record.seller_name_raw,
            json.dumps(record.image_urls, ensure_ascii=False) if record.image_urls else None,
            record.image_count,
            record.source_published_at,
            record.source_modified_at,
            json.dumps(record.source_attributes, ensure_ascii=False, default=str),
            record.raw_payload_ref.model_dump_json(),
            json.dumps(record.extraction_errors, ensure_ascii=False),
            observed_at,  # first_seen_at (kept on conflict by DO UPDATE design)
            observed_at,  # last_seen_at
            observed_at,  # scraped_at
            record.extraction_version,
            observed_at,  # updated_at
            derived["year_num"],
            derived["year_calendar"],
            derived["mileage_km"],
            derived["brand_key"],
            derived["model_key"],
            derived["brand_canonical"],
        )

    def _refresh_current_price(self, key: str) -> None:
        last = self._conn.execute(
            """
            SELECT price_raw, price_amount, price_currency_raw,
                   price_normalization_version
            FROM price_observations WHERE listing_key = ?
            ORDER BY id DESC LIMIT 1
            """,
            (key,),
        ).fetchone()
        if last is None:
            return
        self._conn.execute(
            """
            UPDATE listings SET price_raw = ?, price_amount = ?,
                   price_currency_raw = ?, price_normalization_version = ?
            WHERE listing_key = ?
            """,
            (
                last["price_raw"],
                last["price_amount"],
                last["price_currency_raw"],
                last["price_normalization_version"],
                key,
            ),
        )

    # --- queries (used by tests, sync, and later the API) ---------------------

    def query_df(self, sql: str, params: tuple | None = None):
        """Run a read-only query and return a pandas DataFrame."""
        import pandas as pd

        return pd.read_sql_query(sql, self._conn, params=params or ())

    def has_run(self, run_id: str) -> bool:
        """Whether a run is already recorded as ingested (fast-path skip)."""
        row = self._conn.execute(
            "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return row is not None

    def ingested_run_ids(self) -> list[str]:
        rows = self._conn.execute("SELECT run_id FROM runs ORDER BY run_id").fetchall()
        return [r["run_id"] for r in rows]

    def listing_count(self, platform: str | None = None) -> int:
        if platform:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM listings WHERE source_platform = ?", (platform,)
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM listings").fetchone()
        return int(row["n"])

    def observation_count(self, listing_key: str | None = None) -> int:
        if listing_key:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM price_observations WHERE listing_key = ?",
                (listing_key,),
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM price_observations").fetchone()
        return int(row["n"])
