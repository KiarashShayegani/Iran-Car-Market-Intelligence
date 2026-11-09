"""Helpers to build synthetic Bama API payloads for tests (no network)."""

from __future__ import annotations


def bama_ad(
    code: str = "yjbceoez",
    *,
    year: int = 1398,
    price: dict | None = None,
    **detail_overrides,
) -> dict:
    detail = {
        "type": "car",
        "code": code,
        "url": f"/car/پژو-206/{code}",
        "title": "پژو 206، مدل 1398، تیپ 2",
        "subtitle": "پژو 206",
        "brand": "پژو",
        "brand_fa": "پژو",
        "trim": "تیپ 2",
        "year": year,
        "mileage": "45,000 km",
        "description": "بدون رنگ، بیمه کامل، لاستیک‌ها نو.",
        "location": "تهران",
        "fuel": "بنزینی",
        "transmission": "دنده ای",
        "body_type": "هاچبک",
        "body_type_fa": "هاچبک",
        "body_status": "سالم و بی خط و خش",
        "color": "سفید",
        "image": f"https://cdn.bama.ir/{code}/main.jpg",
        "image_count": 8,
        "time": "لحظات پیش",
        "modified_date": None,
    }
    detail.update(detail_overrides)
    return {
        "type": "ad",
        "detail": detail,
        "metadata": {"canonical": f"https://bama.ir/car/پژو-206/{code}"},
        "price": price if price is not None else {
            "price": 750_000_000, "currency": "تومان", "type": "price",
        },
        "dealer": {"type": "personal", "id": 12345, "name": "فروشگاه نمونه"},
        "images": [
            {"thumb": f"https://cdn.bama.ir/{code}/t1.jpg",
             "main": f"https://cdn.bama.ir/{code}/1.jpg"},
        ],
        "specs": {"gearbox": "5-speed"},
    }


def bama_page(*ads: dict, extra_nodes: list[dict] | None = None) -> dict:
    """A search response containing the given ad nodes plus page furniture."""
    return {
        "ads": [*ads, *(extra_nodes or [])],
        " pageInfo": {"pageIndex": 0, "pageSize": len(ads)},
    }
