"""Synthetic Sheypoor and Hamrah Mechanic payloads for tests (no network)."""

from __future__ import annotations


def sheypoor_ad(
    ad_id: int = 469078490,
    *,
    description: str | None = "بی رنگ و خط، بیمه کامل.",
    **attr_overrides,
) -> dict:
    attributes = {
        "title": "پژو 206 تیپ 5 مدل 97",
        "url": f"https://www.sheypoor.com/v/پژو-206-تیپ-5-{ad_id}.html",
        "location": "تهران",
        "price": [{"label": "", "amount": "2,050,000,000", "currency": "تومان"}],
        "imageCount": 6,
        "images": {"thumbnails": {
            "small": f"https://s.sheypoor.com/img/{ad_id}/t.jpg",
            "medium": f"https://s.sheypoor.com/img/{ad_id}/m.jpg",
        }},
        "categories": [
            {"name": "وسایل نقلیه", "id": 3},
            {"name": "خودرو", "id": 43970},
            {"name": "پژو", "id": 44010},
        ],
        "sellerType": "personal",
        "shopId": 99887,
        "timePassedLabel": "۲ ساعت پیش",
        "priceTag": {"label": "قیمت عادی"},
    }
    attributes = {k: v for k, v in attributes.items() if v is not None}
    attributes.update(attr_overrides)
    return {
        "type": "normal",
        "id": ad_id,
        "attributes": attributes,
        "fullAttributes": [
            {"key": "مدل خودرو", "value": "پژو 206"},
            {"key": "تیپ خودرو", "value": "تیپ 5"},
            {"key": "سال تولید (چهار رقمی)", "value": "1397"},
            {"key": "کیلومتر", "value": "85,000"},
            {"key": "نوع سوخت", "value": "بنزینی"},
            {"key": "گیربکس", "value": "دنده"},
            {"key": "نوع شاسی", "value": "هاچبک"},
            {"key": "وضعیت بدنه", "value": "سالم و بی‌خط‌وخش"},
            {"key": "رنگ", "value": "نقره‌ای"},
        ],
        "description": description,
        "sort": 12,
    }


DEFAULT_META_F = "[1.0, 1794265261245]_469078347n_1794253002.951553"


def sheypoor_page(*ads: dict, meta_f: str = DEFAULT_META_F) -> dict:
    return {
        "data": {
            "ads": list(ads),
            "meta": {
                "total": 44621,
                "p": 1,
                "items_per_page": 24,
                "api_version": "v10.0.0",
                "f": meta_f,
            },
        }
    }


def hamrah_ad(
    order_id: int = 3595004,
    *,
    is_sold: bool | None = None,
    **overrides,
) -> dict:
    record = {
        "orderId": order_id,
        "carNamePersian": "هیوندای آزرا ۶ سیلندر",
        "brandEnglishName": "Hyundai",
        "modelEnglishName": "Azera",
        "carTypeName": "سدان",
        "carYear": 2012,
        "km": 250000,
        "price": 7100000000,
        "carLocation": "شیروان",
        "gearBoxPersian": "اتوماتیک",
        "carColorName": "مشکی",
        "imageUrl": f"https://www.hamrah-mechanic.com/img/{order_id}.jpg",
        "exhibitionDetailUrl": f"/cars-for-sale/{order_id}",
        "isSold": is_sold,
        "carId": 321,
        "carTypeId": 65,
        "leasingOption": False,
        "fastSell": True,
        "badgeType": "special",
    }
    record = {k: v for k, v in record.items() if v is not None or k == "isSold"}
    record.update(overrides)
    return record


def hamrah_page(*ads: dict) -> dict:
    return {"data": {"items": list(ads), "totalPages": 40}}
