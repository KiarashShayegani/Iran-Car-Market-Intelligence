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
