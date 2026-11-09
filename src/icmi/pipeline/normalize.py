"""Explicit normalization for downstream (analytical) fields.

Raw values on canonical records are NEVER modified — normalization produces
separate derived values with an explicit ``*_VERSION`` marker so that a
formatting change at a source can never be misread as a value change in the
price history (see pipeline/store.py).
"""

from __future__ import annotations

# Persian (۰-۹) and Arabic (٠-٩) digit variants mapped to Western digits.
_DIGIT_TRANSLATION = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

# Bumped whenever normalization semantics change; stored on every observation.
PRICE_NORMALIZATION_VERSION = "price-norm-1"

_THOUSANDS_SEPARATORS = (",", "،", "'", " ")


def normalize_price_amount(raw: int | float | str | None) -> int | None:
    """Normalize a price to an integer amount in the source's currency.

    Returns None when the raw value carries no parseable amount (negotiable
    labels, free text, missing). The caller decides what None means; this
    function never zeroes or fabricates.
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return int(raw) if raw == raw else None  # NaN guard for floats
    text = str(raw).strip().translate(_DIGIT_TRANSLATION)
    for separator in _THOUSANDS_SEPARATORS:
        text = text.replace(separator, "")
    if not text.isdigit():
        return None
    return int(text)


def normalize_year(raw: int | str | None) -> int | None:
    """Normalize a year to an integer without asserting Shamsi vs Gregorian.

    Both calendar systems exist in Iranian listings (imports often Gregorian);
    disambiguation is an analytical decision and happens later. Here we only
    parse digits so downstream layers can branch safely.
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw) if raw == raw else None
    text = str(raw).strip().translate(_DIGIT_TRANSLATION)
    return int(text) if text.isdigit() else None


MILEAGE_NORMALIZATION_VERSION = "mileage-norm-1"

_MILEAGE_WORDS_ZERO = ("صفر کیلومتر", "صفرکیلومتر", "صفر")
_MILEAGE_KM_SUFFIXES = ("km", "کیلومتر", "كيلومتر", "kilometer")


def normalize_mileage_km(raw: int | float | str | None) -> int | None:
    """Parse mileage to integer km. Persian digits and separators handled;
    "صفر کیلومتر" → 0; text without digits (e.g. "کارکرده") → None (unknown,
    never zero-filled)."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return int(raw) if raw == raw else None
    text = str(raw).strip().translate(_DIGIT_TRANSLATION)
    for suffix in _MILEAGE_KM_SUFFIXES:
        text = text.replace(suffix, "")
    text = text.replace(",", "").replace("،", "").strip()
    for word in _MILEAGE_WORDS_ZERO:
        if text == word:
            return 0
    if text.isdigit():
        return int(text)
    return None


def classify_year_calendar(year: int | None) -> str | None:
    """Classify a year as 'shamsi' or 'gregorian' from its numeric range.

    Heuristic by design (documented): Iranian listings mix calendars —
    imports are usually Gregorian, domestic Shamsi. Ambiguous values
    (e.g. 1500-1899) are 'unknown'. Downstream comparisons must group by
    calendar, never mix them.
    """
    if year is None:
        return None
    if 1340 <= year <= 1450:
        return "shamsi"
    if 1900 <= year <= 2100:
        return "gregorian"
    return "unknown"
