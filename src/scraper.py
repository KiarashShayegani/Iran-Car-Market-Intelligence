"""
Bama.ir scraper with comprehensive error handling, retry logic,
deduplication, and per-record validation before accumulation.
"""

import hashlib
import json
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
import yaml
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class BamaScraper:
    """
    Scraper for bama.ir hidden API.

    Supports multi-brand scraping with automatic retry, deduplication,
    and comprehensive logging of every stage.
    """

    def __init__(self, brand_slug: str, config_path: str = "config/brands.yaml"):
        """
        Initialize scraper for a specific brand.

        Args:
            brand_slug: Brand identifier (e.g., 'pride', 'peugeot').
            config_path: Path to brands configuration file.
        """
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.brand_slug = brand_slug
        self.brand_info = self.config["brands"].get(brand_slug, {})
        self.scrape_cfg = self.config["scraping"]

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.scrape_cfg["user_agent"],
            "Accept": "application/json",
            "Accept-Language": "fa-IR,fa;q=0.9",
        })

        self.records: List[Dict] = []
        self.errors: List[Dict] = []
        self.skip_reasons: Counter = Counter()
        self.logger = logger.bind(brand=brand_slug)

        self.logger.info(
            "Initialized scraper for '{}' ({})",
            brand_slug,
            self.brand_info.get("name_fa", "unknown"),
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type((requests.RequestException, json.JSONDecodeError)),
        reraise=True,
    )
    def _fetch_page(self, page_index: int) -> Dict:
        """
        Fetch a single page with exponential backoff retry.

        Args:
            page_index: Page number to fetch.

        Returns:
            Parsed JSON response.

        Raises:
            requests.RequestException: After all retries exhausted.
        """
        url = (
            f"{self.scrape_cfg['base_url']}"
            f"?vehicle={self.brand_slug}&pageIndex={page_index}"
        )

        self.logger.debug("Fetching page {}: {}", page_index, url)

        response = self.session.get(url, timeout=self.scrape_cfg["request_timeout"])
        response.raise_for_status()
        return response.json()

    def _parse_ad(self, item: Dict) -> Optional[Dict]:
        """
        Extract structured data from a single ad item.

        Performs strict validation at the field level. If any critical
        field is malformed, returns None and logs the reason.

        Args:
            item: Raw ad item from API.

        Returns:
            Parsed record dict or None if invalid.
        """
        try:
            if item.get("type") != "ad":
                self.skip_reasons["not_ad_item"] += 1
                return None

            detail = item.get("detail", {})
            price_data = item.get("price", {})

            # --- Extract title ---
            title_raw = detail.get("title", "")
            title_parts = [p.strip() for p in title_raw.split("،")]

            if len(title_parts) >= 2:
                car_name = title_parts[1]
                car_model = title_parts[0]
            else:
                car_name = title_parts[0] if title_parts else "unknown"
                car_model = title_parts[0] if title_parts else "unknown"

            # --- Parse mileage ---
            raw_mileage = detail.get("mileage", "کارکرده")
            mileage_unknown = False

            if raw_mileage == "صفر کیلومتر":
                mileage = 0
            elif raw_mileage == "کارکرده":
                mileage = None
                mileage_unknown = True
            else:
                mileage_str = str(raw_mileage).replace("km", "").replace(",", "").strip()
                mileage = int(mileage_str) if mileage_str.isdigit() else None
                if mileage is None:
                    mileage_unknown = True

            # --- Parse price ---
            price_str = price_data.get("price", "0")
            if not price_str or str(price_str).strip() == "":
                self.skip_reasons["no_price_negotiable"] += 1
                return None

            price_clean = str(price_str).replace(",", "").strip()
            price = int(price_clean) if price_clean.isdigit() else 0

            # CRITICAL: Skip zero-price listings (توافقی / corrupted)
            if price == 0:
                self.skip_reasons["zero_price"] += 1
                return None

            # --- Parse year ---
            year_raw = detail.get("year", "0")
            year = int(year_raw) if str(year_raw).isdigit() else 0

            # Skip invalid years
            if not (1340 <= year <= 1410):
                self.skip_reasons["invalid_year"] += 1
                return None

            # Generate deterministic ID for deduplication
            ad_code = detail.get("code", json.dumps(item, sort_keys=True))
            listing_id = hashlib.md5(
                f"{self.brand_slug}-{ad_code}".encode()
            ).hexdigest()[:16]

            record = {
                "listing_id": listing_id,
                "brand_slug": self.brand_slug,
                "brand_name_fa": self.brand_info.get("name_fa", ""),
                "manufacturer": self.brand_info.get("manufacturer", ""),
                "name": car_name,
                "model": car_model,
                "trim": detail.get("trim", "ساده") or "ساده",
                "year": year,
                "mileage": float(mileage) if mileage is not None else None,
                "mileage_unknown": mileage_unknown,
                "fuel": detail.get("fuel", "بنزینی") or "بنزینی",
                "transmission": detail.get("transmission", "دنده ای") or "دنده ای",
                "body_status": detail.get("body_status", "کارکرده") or "کارکرده",
                "price": price,
                "scraped_at": datetime.now().isoformat(),
                "source_url": f"https://bama.ir/car/{detail.get('code', '')}",
            }

            return record

        except (KeyError, ValueError, IndexError, TypeError) as e:
            self.skip_reasons["parse_error"] += 1
            self.logger.warning(
                "Parse error: {} | item_keys: {}", e, list(item.keys())
            )
            self.errors.append({"error": str(e), "item_keys": list(item.keys())})
            return None

    def scrape(self, max_pages: Optional[int] = None) -> pd.DataFrame:
        """
        Scrape all pages for this brand until empty pages or max reached.

        Args:
            max_pages: Maximum pages to scrape. None = use config.

        Returns:
            Deduplicated DataFrame of valid records.
        """
        page = 0
        consecutive_empty = 0
        max_empty = self.scrape_cfg["consecutive_empty_threshold"]
        max_pages = max_pages or self.scrape_cfg["max_pages_per_brand"]

        self.logger.info("Starting scrape | max_pages={}", max_pages)

        while page < max_pages:
            try:
                data = self._fetch_page(page)
                ads = data.get("data", {}).get("ads", [])

                if not ads:
                    consecutive_empty += 1
                    self.logger.info(
                        "Page {} empty ({}/{})",
                        page,
                        consecutive_empty,
                        max_empty,
                    )
                    if consecutive_empty >= max_empty:
                        self.logger.info(
                            "Stopping: {} consecutive empty pages", max_empty
                        )
                        break
                    page += 1
                    continue

                consecutive_empty = 0
                parsed = [self._parse_ad(ad) for ad in ads]
                valid = [r for r in parsed if r is not None]
                self.records.extend(valid)

                self.logger.info(
                    "Page {}: {}/{} valid | total: {}",
                    page,
                    len(valid),
                    len(ads),
                    len(self.records),
                )

                jitter = self.scrape_cfg.get("jitter_seconds", 0)
                time.sleep(self.scrape_cfg["rate_limit_delay"] + random.uniform(0, jitter))
                page += 1

            except Exception as e:
                self.logger.error("Fatal error on page {}: {}", page, e)
                break

        df = pd.DataFrame(self.records)

        if self.skip_reasons:
            self.logger.info(
                "Skip reasons for '{}': {}", self.brand_slug, dict(self.skip_reasons)
            )

        if df.empty:
            self.logger.warning("No records scraped for {}", self.brand_slug)
            return df

        # Deduplicate
        before = len(df)
        df = df.drop_duplicates(subset=["listing_id"], keep="last")
        after = len(df)

        self.logger.info(
            "Scrape complete | unique={} | dupes_removed={} | errors={}",
            after,
            before - after,
            len(self.errors),
        )

        return df

    def save_raw(self, df: pd.DataFrame, output_dir: str = "data/raw") -> str:
        """
        Save raw scrape as timestamped Parquet.

        Args:
            df: DataFrame to save.
            output_dir: Target directory.

        Returns:
            Path to saved file.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(output_dir) / f"{self.brand_slug}_{timestamp}.parquet"
        df.to_parquet(path, index=False)
        self.logger.info("Saved raw data: {} ({} rows)", path, len(df))
        return str(path)


def append_to_master(
    df: pd.DataFrame, master_path: str = "data/raw/master_history.parquet"
) -> pd.DataFrame:
    """
    Merge freshly scraped listings into a cumulative master dataset.

    Without this, every pipeline run overwrites `combined_latest.parquet`
    and the model always trains on a single day's snapshot (~2-3k rows).
    This keeps every listing ever seen, updating its latest snapshot
    (price, mileage, etc.) while preserving the first time it appeared,
    so the dataset - and model quality - actually grows day over day.

    Args:
        df: Freshly scraped, deduplicated DataFrame for this run.
        master_path: Path to the cumulative Parquet file.

    Returns:
        The updated master DataFrame.
    """
    path = Path(master_path)
    df = df.copy()
    df["first_seen_at"] = df["scraped_at"]

    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df

    first_seen = combined.groupby("listing_id")["first_seen_at"].min()
    combined = combined.sort_values("scraped_at").drop_duplicates(
        subset="listing_id", keep="last"
    )
    combined["first_seen_at"] = combined["listing_id"].map(first_seen)

    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False)
    logger.info(
        "Master history updated | {} total unique listings ({} scraped this run)",
        len(combined),
        len(df),
    )
    return combined


class MultiBrandScraper:
    """Orchestrate scraping across all enabled brands."""

    def __init__(self, config_path: str = "config/brands.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def scrape_all(
        self, max_pages_per_brand: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Scrape all enabled brands.

        Args:
            max_pages_per_brand: Limit pages per brand for testing.

        Returns:
            Dictionary mapping brand_slug to DataFrame.
        """
        enabled = {
            k: v
            for k, v in self.config["brands"].items()
            if v.get("enabled", True)
        }

        logger.info("Multi-brand scrape | {} brands enabled", len(enabled))

        results: Dict[str, pd.DataFrame] = {}
        cooldown = self.config["scraping"].get("inter_brand_cooldown", 0)

        for brand_slug in enabled:
            try:
                scraper = BamaScraper(brand_slug)
                df = scraper.scrape(max_pages=max_pages_per_brand)

                if len(df) > 0:
                    scraper.save_raw(df)
                    results[brand_slug] = df
                else:
                    logger.warning("No data for {}", brand_slug)

            except Exception as e:
                logger.error("Failed to scrape {}: {}", brand_slug, e)
                continue
            finally:
                # A short pause between brands reduces load on bama.ir's
                # API and lowers the odds of tripping rate-limiting (503s).
                if cooldown:
                    time.sleep(cooldown)

        # Save today's snapshot (kept for debugging/audit) and merge it
        # into the cumulative master history the cleaner/trainer use.
        if results:
            combined = pd.concat(results.values(), ignore_index=True)
            combined_path = Path("data/raw") / "combined_latest.parquet"
            combined.to_parquet(combined_path, index=False)
            logger.info(
                "Combined snapshot: {} records -> {}",
                len(combined),
                combined_path,
            )
            append_to_master(combined)

        return results
