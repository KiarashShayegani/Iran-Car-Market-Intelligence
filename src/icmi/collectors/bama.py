"""Bama.ir collector.

Endpoint: https://bama.ir/cad/api/search (experimental, undocumented —
verify current behavior on every structural surprise; treat changes as
extraction failures, not crashes).

Extraction ported from the validated research notebook
(docs/research/web_scraping_01_code.ipynb, cell 15):

- Ad nodes carry ``type == "ad"`` with a ``detail`` object; vehicle listings
  have ``detail.type`` of None or "car".
- ``detail.code`` is the stable listing id; the public page URL comes from
  ``metadata.canonical`` or is built from the relative ``detail.url``.
- ``detail.subtitle`` is the model; ``brand_fa``/``brand`` give Persian and
  English brand names. ``specs`` and any extra detail keys are preserved in
  ``source_attributes`` (P2 — nothing useful is discarded).
"""

from __future__ import annotations

import json
from typing import Any

from icmi.collectors.base import BaseCollector
from icmi.core.records import CanonicalRecord, RawPayloadRef, utc_now


def _nonempty(value: Any) -> bool:
    return value is not None and value != ""


def _walk(node: Any):
    """Yield every dict in an arbitrarily nested JSON structure."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def iter_bama_ads(page_payload: dict) -> list[dict]:
    """Extract vehicle ad nodes from a Bama search page response."""
    ads: list[dict] = []
    seen: set[str] = set()
    for node in _walk(page_payload):
        if node.get("type") != "ad" or not isinstance(node.get("detail"), dict):
            continue
        detail = node["detail"]
        if detail.get("type") not in (None, "car"):
            continue
        listing_id = detail.get("code")
        unique_key = str(listing_id) if listing_id else detail.get("url")
        if not unique_key or unique_key in seen:
            continue
        seen.add(unique_key)
        ads.append(node)
    return ads


def listing_id_of(ad: dict) -> str | None:
    detail = ad.get("detail") or {}
    code = detail.get("code")
    return str(code) if code else None


def bama_public_url(ad: dict) -> str | None:
    detail = ad.get("detail") or {}
    metadata = ad.get("metadata") or {}
    canonical = metadata.get("canonical")
    relative = detail.get("url")
    if isinstance(canonical, str) and canonical.startswith("https://"):
        return canonical
    if isinstance(relative, str) and relative.startswith("/"):
        return "https://bama.ir" + relative
    if isinstance(relative, str) and relative.startswith("https://"):
        return relative
    return None


class BamaCollector(BaseCollector):
    """Collector for bama.ir search API (pageIndex / pageSize pagination)."""

    source_key = "bama"

    def page_params(self, page_index: int) -> dict[str, Any]:
        params: dict[str, Any] = {**self.collection_params, "pageIndex": page_index}
        if self.source.pagination and self.source.pagination.page_size:
            params["pageSize"] = self.source.pagination.page_size
        return params

    def iter_ads(self, page_payload: dict) -> list[dict]:
        return iter_bama_ads(page_payload)

    def has_more(self, page_payload: dict) -> bool:
        # Response envelope: {"status": true, "errors": [], "data": {"ads": [...],
        # "metadata": {"total_count", "total_pages", "has_next", ...}}}
        data = page_payload.get("data")
        metadata = data.get("metadata") if isinstance(data, dict) else None
        if isinstance(metadata, dict) and isinstance(metadata.get("has_next"), bool):
            return metadata["has_next"]
        return True

    def listing_id_of(self, ad: dict) -> str | None:
        return listing_id_of(ad)

    def parse_listing(self, ad: dict, payload_ref: RawPayloadRef) -> CanonicalRecord | None:
        detail = ad.get("detail") or {}
        metadata = ad.get("metadata") or {}
        dealer = ad.get("dealer") if isinstance(ad.get("dealer"), dict) else {}
        specs = ad.get("specs") if isinstance(ad.get("specs"), dict) else {}

        errors: dict[str, str] = {}
        record = CanonicalRecord(
            source_platform=self.source_key,
            source_listing_id=str(detail.get("code")),
            scraped_at=utc_now(),
            extraction_version=self.pipeline.extraction_version,
            raw_payload_ref=payload_ref,
        )

        def set_field(field: str, extractor) -> None:
            """Populate one field; extraction failures are recorded, never silent."""
            try:
                setattr(record, field, extractor())
            except Exception as exc:  # noqa: BLE001 — any parser bug is an extraction failure
                errors[field] = f"{type(exc).__name__}: {exc}"

        def first(*values):
            return next((v for v in values if _nonempty(v)), None)

        set_field("title", lambda: detail.get("title"))
        set_field("description", lambda: detail.get("description"))
        set_field("original_url", lambda: bama_public_url(ad))
        set_field(
            "brand_raw",
            lambda: first(detail.get("brand_fa"), detail.get("brand")),
        )
        set_field("brand_en_raw", lambda: detail.get("brand"))
        set_field("model_raw", lambda: detail.get("subtitle"))
        set_field("trim_raw", lambda: detail.get("trim"))
        set_field("year_raw", lambda: detail.get("year"))
        set_field("mileage_raw", lambda: detail.get("mileage"))
        set_field("location_raw", lambda: detail.get("location"))
        set_field("fuel_raw", lambda: detail.get("fuel"))
        set_field("transmission_raw", lambda: detail.get("transmission"))
        set_field(
            "body_type_raw",
            lambda: first(detail.get("body_type_fa"), detail.get("body_type")),
        )
        set_field("body_condition_raw", lambda: detail.get("body_status"))
        set_field("color_raw", lambda: detail.get("color"))

        def extract_price():
            price_obj = ad.get("price") or detail.get("price")
            if isinstance(price_obj, dict):
                record.price_details = price_obj
                record.price_currency_raw = price_obj.get("currency")
                return price_obj.get("price")
            record.price_details = price_obj
            return price_obj

        set_field("price_raw", extract_price)

        set_field("seller_type", lambda: dealer.get("type"))
        set_field("seller_id", lambda: str(dealer["id"]) if dealer.get("id") is not None else None)
        set_field("seller_name_raw", lambda: dealer.get("name"))

        def extract_images():
            urls: list[str] = []
            if _nonempty(detail.get("image")):
                urls.append(detail["image"])
            images = ad.get("images")
            for image in images if isinstance(images, list) else []:
                if isinstance(image, dict):
                    urls.extend(
                        v for v in image.values()
                        if isinstance(v, str) and v.startswith("http")
                    )
                elif isinstance(image, str) and image.startswith("http"):
                    urls.append(image)
            return list(dict.fromkeys(urls))

        set_field("image_urls", extract_images)  # empty list = no images (missing_in_ad)
        set_field("image_count", lambda: detail.get("image_count"))
        set_field("source_published_at", lambda: detail.get("time"))
        set_field("source_modified_at", lambda: detail.get("modified_date"))

        record.source_attributes = {
            "specs": specs if specs else None,
            "detail_extra": detail,
            "metadata_extra": metadata if metadata else None,
        }
        record.extraction_errors = errors
        return record

    @staticmethod
    def dump_payload(ad: dict) -> str:
        return json.dumps(ad, ensure_ascii=False, default=str)
