"""Canonical record schema for ICMI.

One record per advertisement, unified across platforms. Raw values are
preserved verbatim (``*_raw`` fields); normalization happens only in derived
fields downstream and never overwrites them. Null means "not available" —
never zero, never fabricated. The three missing-data states are:

- ``not_supported_by_platform`` — the platform never provides this field
  (declared per-source in config/sources.yaml ``field_support``);
- ``missing_in_ad`` — the platform supports the field but this ad lacks it
  (null + no extraction error);
- ``extraction_failed`` — extraction of the field raised (recorded in
  ``extraction_errors``, never silently nulled).

Every record carries collection metadata in UTC and a ``raw_payload_ref``
pointing at the immutable raw payload it was extracted from.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

# P0 — essential identity and content fields.
P0_FIELDS: tuple[str, ...] = (
    "source_listing_id",
    "original_url",
    "title",
    "description",
    "manufacturer_raw",
    "brand_raw",
    "model_raw",
    "trim_raw",
    "year_raw",
    "mileage_raw",
    "price_raw",
    "location_raw",
)

# P1 — important fields, extracted whenever the platform provides them.
P1_FIELDS: tuple[str, ...] = (
    "fuel_raw",
    "transmission_raw",
    "body_type_raw",
    "body_condition_raw",
    "color_raw",
    "seller_type",
    "seller_id",
    "image_urls",
    "image_count",
    "source_published_at",
    "source_modified_at",
)


def utc_now() -> datetime:
    """Current UTC time (collection metadata timestamps must all be UTC)."""
    return datetime.now(tz=UTC)


class RawPayloadRef(BaseModel):
    """Pointer from a canonical record to its immutable raw payload."""

    run_id: str
    platform: str
    file: str  # path relative to the run's raw directory, e.g. "records.jsonl"
    key: str  # record key within the file (source_listing_id for JSONL lines)


class CanonicalRecord(BaseModel):
    """Unified representation of one advertisement, raw values preserved."""

    # --- identity ----------------------------------------------------------
    source_platform: str
    source_listing_id: str  # always a string, even if all digits
    original_url: str | None = None  # public listing page, never an API URL

    # --- content -----------------------------------------------------------
    title: str | None = None
    description: str | None = None

    # --- vehicle identity (source spelling preserved) -----------------------
    manufacturer_raw: str | None = None
    brand_raw: str | None = None
    brand_en_raw: str | None = None
    model_raw: str | None = None
    trim_raw: str | None = None
    year_raw: int | str | None = None
    mileage_raw: int | str | None = None

    # --- price (units/semantics documented per platform before comparing) ---
    price_raw: int | float | str | None = None
    price_currency_raw: str | None = None
    price_details: Any = None  # original price object if the source has one

    location_raw: str | None = None

    # --- P1 ----------------------------------------------------------------
    fuel_raw: str | None = None
    transmission_raw: str | None = None
    body_type_raw: str | None = None
    body_condition_raw: str | None = None
    color_raw: str | None = None
    seller_type: str | None = None
    seller_id: str | None = None  # string even if numeric at the source
    seller_name_raw: str | None = None
    image_urls: list[str] = Field(default_factory=list)
    image_count: int | None = None
    source_published_at: str | None = None  # may be a relative label; keep raw
    source_modified_at: str | None = None

    # --- P2 / source-specific: nothing useful is discarded ------------------
    source_attributes: dict[str, Any] = Field(default_factory=dict)

    # --- collection metadata (all UTC) --------------------------------------
    scraped_at: datetime
    extraction_version: str
    raw_payload_ref: RawPayloadRef

    # --- extraction diagnostics ---------------------------------------------
    extraction_errors: dict[str, str] = Field(default_factory=dict)


def has_value(value: Any) -> bool:
    """True when a field genuinely carries data (ported from the research notebook)."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True
