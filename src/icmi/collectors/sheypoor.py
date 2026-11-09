"""Sheypoor collector.

Endpoint: https://www.sheypoor.com/api/v10.0.0/search/iran/car (experimental).

Pagination (verified live 2026-11-09):
- A fresh chain starts WITHOUT the ``f`` cursor: ``p=0`` returns page one.
- Each response carries ``meta`` with the cursor for the NEXT request in
  ``meta.f``; subsequent requests pass both ``f`` (chained cursor) and ``p``.
- ``meta.p`` echoes the 1-based page number; ``meta.total`` and
  ``meta.items_per_page`` allow a has-more computation.
- The base loop's stale-page stop (zero new IDs on 3 consecutive pages)
  guards against a broken cursor chain repeating the same result set.

Prices: ``attributes.price`` is a list; its first element carries a
comma-formatted ``amount`` string plus ``currency`` (Toman) — amounts are
preserved raw; integer normalization is downstream.
"""

from __future__ import annotations

from typing import Any

from icmi.collectors.base import BaseCollector
from icmi.core.records import CanonicalRecord, RawPayloadRef, utc_now

CAR_CATEGORY_NAMES = {"خودرو", "car", "cars"}


def _walk(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def _attr(ad: dict, *keys: str):
    """First non-empty fullAttributes value matching one of the given keys."""
    for attribute in ad.get("fullAttributes") or []:
        if (
            isinstance(attribute, dict)
            and attribute.get("key") in keys
            and attribute.get("value") is not None
            and attribute.get("value") != ""
        ):
            return attribute.get("value")
    return None


def iter_sheypoor_ads(page_payload: dict) -> list[dict]:
    """Normal (non-native-ad) nodes in the automobile category."""
    ads: list[dict] = []
    for node in _walk(page_payload):
        if node.get("type") != "normal" or not isinstance(node.get("attributes"), dict):
            continue
        if not isinstance(node.get("fullAttributes"), list):
            continue
        categories = node["attributes"].get("categories")
        names = {
            str(c.get("name", "")).strip()
            for c in categories or []
            if isinstance(c, dict)
        }
        if names & CAR_CATEGORY_NAMES:
            ads.append(node)
    return ads


def sheypoor_meta(page_payload: dict) -> dict | None:
    for node in _walk(page_payload):
        if isinstance(node.get("meta"), dict):
            return node["meta"]
    return None


class SheypoorCollector(BaseCollector):
    """Collector for Sheypoor's search API with meta.f cursor chaining."""

    source_key = "sheypoor"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._cursor: str | None = None

    def page_params(self, page_index: int) -> dict[str, Any]:
        params: dict[str, Any] = {**self.collection_params, "p": page_index}
        if self._cursor is not None:
            params["f"] = self._cursor
        return params

    def observe_page(self, page_payload: dict) -> None:
        meta = sheypoor_meta(page_payload)
        if meta and meta.get("f"):
            self._cursor = meta["f"]

    def has_more(self, page_payload: dict) -> bool:
        meta = sheypoor_meta(page_payload)
        if not meta:
            return True
        total = meta.get("total")
        per_page = meta.get("items_per_page")
        current = meta.get("p")  # 1-based echo
        if all(isinstance(v, (int, float)) for v in (total, per_page, current)):
            return bool(current * per_page < total)
        return True

    def iter_ads(self, page_payload: dict) -> list[dict]:
        return iter_sheypoor_ads(page_payload)

    def listing_id_of(self, ad: dict) -> str | None:
        value = ad.get("id")
        return str(value) if value is not None and str(value).strip() else None

    def parse_listing(self, ad: dict, payload_ref: RawPayloadRef) -> CanonicalRecord | None:
        attributes = ad.get("attributes") or {}
        errors: dict[str, str] = {}

        record = CanonicalRecord(
            source_platform=self.source_key,
            source_listing_id=str(ad.get("id")),
            scraped_at=utc_now(),
            extraction_version=self.pipeline.extraction_version,
            raw_payload_ref=payload_ref,
        )

        def set_field(field: str, extractor) -> None:
            try:
                setattr(record, field, extractor())
            except Exception as exc:  # noqa: BLE001
                errors[field] = f"{type(exc).__name__}: {exc}"

        # manufacturer/brand: last category name (the branch under خودرو)
        def manufacturer():
            categories = attributes.get("categories") or []
            for category in reversed(categories):
                if isinstance(category, dict) and category.get("name"):
                    return str(category["name"])
            return None

        def extract_price():
            price_list = attributes.get("price")
            price_obj = price_list[0] if isinstance(price_list, list) and price_list else None
            record.price_details = price_list
            if isinstance(price_obj, dict):
                record.price_currency_raw = price_obj.get("currency")
                amount = price_obj.get("amount")
                if amount not in (None, ""):
                    return amount
                return price_obj.get("label") or None
            return None

        def extract_images():
            urls: list[str] = []
            tree = attributes.get("images")
            if isinstance(tree, dict):
                thumbnails = tree.get("thumbnails")
                if isinstance(thumbnails, dict):
                    urls.extend(
                        v for v in thumbnails.values()
                        if isinstance(v, str) and v.startswith("http")
                    )
            elif isinstance(tree, list):
                urls.extend(
                    v for v in tree if isinstance(v, str) and v.startswith("http")
                )
            return list(dict.fromkeys(urls))

        set_field("title", lambda: attributes.get("title"))
        set_field("description", lambda: ad.get("description"))
        set_field(
            "original_url",
            lambda: attributes.get("url")
            if isinstance(attributes.get("url"), str)
            and attributes["url"].startswith("https://")
            else None,
        )
        set_field("manufacturer_raw", manufacturer)
        set_field("brand_raw", manufacturer)
        set_field("model_raw", lambda: _attr(ad, "مدل خودرو", "مدل"))
        set_field("trim_raw", lambda: _attr(ad, "تیپ خودرو", "تیپ"))
        set_field("year_raw", lambda: _attr(ad, "سال تولید (چهار رقمی)"))
        set_field("mileage_raw", lambda: _attr(ad, "کیلومتر"))
        set_field("price_raw", extract_price)
        set_field("location_raw", lambda: attributes.get("location"))
        set_field("fuel_raw", lambda: _attr(ad, "نوع سوخت"))
        set_field("transmission_raw", lambda: _attr(ad, "گیربکس"))
        set_field("body_type_raw", lambda: _attr(ad, "نوع شاسی"))
        set_field("body_condition_raw", lambda: _attr(ad, "وضعیت بدنه"))
        set_field("color_raw", lambda: _attr(ad, "رنگ"))
        set_field("seller_type", lambda: attributes.get("sellerType"))
        set_field(
            "seller_id",
            lambda: str(attributes["shopId"]) if attributes.get("shopId") is not None else None,
        )
        set_field("image_urls", extract_images)
        set_field("image_count", lambda: attributes.get("imageCount"))
        # Relative label (e.g. "۲ ساعت پیش") — same treatment as Bama's labels.
        set_field("source_published_at", lambda: attributes.get("timePassedLabel"))

        record.source_attributes = {
            "price_tag_raw": attributes.get("priceTag"),
            "categories": [
                {k: c.get(k) for k in ("name", "id") if k in c}
                for c in attributes.get("categories") or []
                if isinstance(c, dict)
            ],
            "video_count": attributes.get("videoCount"),
            "paid_tags": attributes.get("paidTags"),
            "category_id": attributes.get("categoryId"),
            "bump_raw": attributes.get("bump"),
            "is_secure_purchase": attributes.get("isSecurePurchase"),
            "telephone_raw": attributes.get("telephone"),
            "hide_phone_number": attributes.get("hidePhoneNumber"),
            "shop_logo_url": attributes.get("shopLogo"),
            "sort_raw": ad.get("sort"),
            "full_attributes": ad.get("fullAttributes"),
        }
        record.extraction_errors = errors
        return record
