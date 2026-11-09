"""Hamrah Mechanic collector.

Endpoint: https://www.hamrah-mechanic.com/api/v1/common/newexhibition
(dealer exhibition inventory — not classifieds; experimental).

Verified live (2026-11-09): ``page`` is 1-based and page 1 is the true
first page (zero overlap with page 2); 24 records per page at size=24.

Platform limitations (declared in config field_support): no description,
fuel, body type, body condition, seller identity, image count, or
timestamps in inventory responses. Years/`carYear` may be Gregorian for
imports — preserved raw.

Deliberate deviation from the research notebook: ``model_raw`` is
``modelEnglishName`` only (never the full ``carNamePersian`` title, which
would pollute cross-platform model comparison); Persian name stays in
``title``. Sold vehicles are skipped and counted, not silently dropped.
"""

from __future__ import annotations

from typing import Any

from icmi.collectors.base import BaseCollector
from icmi.core.records import CanonicalRecord, RawPayloadRef, utc_now


def _walk(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def iter_hamrah_inventory(page_payload: dict) -> list[dict]:
    """Vehicle inventory records: orderId + Persian car name."""
    return [
        node
        for node in _walk(page_payload)
        if isinstance(node.get("orderId"), (str, int))
        and node.get("carNamePersian")
    ]


class HamrahMechanicCollector(BaseCollector):
    """Collector for Hamrah Mechanic's exhibition inventory API (1-based page)."""

    source_key = "hamrah_mechanic"

    def page_params(self, page_index: int) -> dict[str, Any]:
        params: dict[str, Any] = {"page": page_index + 1}  # source is 1-based
        if self.source.pagination and self.source.pagination.page_size:
            params["size"] = self.source.pagination.page_size
        return params

    def iter_ads(self, page_payload: dict) -> list[dict]:
        return iter_hamrah_inventory(page_payload)

    def listing_id_of(self, ad: dict) -> str | None:
        value = ad.get("orderId")
        return str(value) if value is not None and str(value).strip() else None

    def parse_listing(self, ad: dict, payload_ref: RawPayloadRef) -> CanonicalRecord | None:
        if ad.get("isSold") is True:
            self.skip_reasons["sold_listing"] = self.skip_reasons.get("sold_listing", 0) + 1
            return None

        errors: dict[str, str] = {}
        record = CanonicalRecord(
            source_platform=self.source_key,
            source_listing_id=str(ad.get("orderId")),
            scraped_at=utc_now(),
            extraction_version=self.pipeline.extraction_version,
            raw_payload_ref=payload_ref,
        )

        def set_field(field: str, extractor) -> None:
            try:
                setattr(record, field, extractor())
            except Exception as exc:  # noqa: BLE001
                errors[field] = f"{type(exc).__name__}: {exc}"

        def first(*values):
            return next((v for v in values if v is not None and v != ""), None)

        def extract_url():
            relative = ad.get("exhibitionDetailUrl")
            if isinstance(relative, str) and relative.startswith("/cars-for-sale/"):
                return "https://www.hamrah-mechanic.com" + relative
            if (
                isinstance(relative, str)
                and relative.startswith("https://www.hamrah-mechanic.com/")
            ):
                return relative
            return None

        def extract_images():
            image = ad.get("imageUrl")
            if isinstance(image, str) and image.startswith("http"):
                return [image]
            return []

        set_field("title", lambda: ad.get("carNamePersian"))
        set_field("original_url", extract_url)
        set_field("manufacturer_raw", lambda: ad.get("brandEnglishName"))
        set_field("brand_raw", lambda: ad.get("brandEnglishName"))
        set_field("model_raw", lambda: ad.get("modelEnglishName"))
        set_field("trim_raw", lambda: ad.get("carTypeName"))
        set_field("year_raw", lambda: ad.get("carYear"))
        set_field("mileage_raw", lambda: ad.get("km"))
        set_field("price_raw", lambda: ad.get("price"))
        set_field("location_raw", lambda: ad.get("carLocation"))
        set_field("transmission_raw", lambda: ad.get("gearBoxPersian"))
        set_field("color_raw", lambda: ad.get("carColorName"))
        set_field("image_urls", extract_images)

        record.source_attributes = {
            "car_id": ad.get("carId"),
            "car_type_id": ad.get("carTypeId"),
            "leasing_option": ad.get("leasingOption"),
            "leasing_section_tag": ad.get("leasingSectionTagEnum"),
            "fast_sell": ad.get("fastSell"),
            "in_auto_center": ad.get("inAutoCenter"),
            "badge_type": ad.get("badgeType"),
            "offer_price": ad.get("offerPrice"),
            "offer_price_difference_pct": ad.get("offerPriceDifferencePercentage"),
            "special_offer_end_time": ad.get("specialOfferEndTime"),
            "coming_soon": ad.get("comingSoon"),
            "available_colors": ad.get("availableColors"),
            "footer_features": ad.get("footerFeatures"),
        }
        record.extraction_errors = errors
        return record
