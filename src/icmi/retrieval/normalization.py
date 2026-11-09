"""Persian text normalization for retrieval and analytics.

Two hard rules:
- Raw values on canonical records are NEVER modified; normalization produces
  separate derived fields.
- Normalized forms are for MATCHING only (search keys, embeddings, grouping).
  Display uses the original text.

What matching normalization does:
- Arabic character variants → Persian (ي→ی, ك→ک, ة→ه, أ/إ→ا, ى→ی, ؤ→و, ٱ→ا)
- Zero-width non-joiner (ZWNJ) → removed (می‌شود → میشود) so both spellings match
- Arabic tatweel (ـ) removed; هٔ → ه
- Persian (۰-۹) and Arabic (٠-٩) digits → Western digits
- English lowercased; whitespace collapsed; common separators unified
"""

from __future__ import annotations

import re

ZWNJ = "\u200c"

# Arabic → Persian character variants (matching-oriented, not typographic)
CHAR_MAP = str.maketrans(
    {
        "ي": "ی",
        "ك": "ک",
        "ى": "ی",
        "ة": "ه",
        "أ": "ا",
        "إ": "ا",
        "ٱ": "ا",
        "ؤ": "و",
        "ئ": "ی",
        "\u0654": "",  # combining hamza above
    }
)

DIGIT_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

_TATWEEL = "\u0640"
_HEH_HAMZA_ABOVE = "\u0647\u0654"

_ZWNJ_RE = re.compile(f"[{ZWNJ}\u200c\u200d\u200e\u200f]")
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[،؛٬]")


def persian_to_western_digits(text: str) -> str:
    """Persian/Arabic digits → Western digits (raw text otherwise untouched)."""
    return text.translate(DIGIT_MAP)


def normalize_for_matching(text: str | None) -> str:
    """Normalized form of a text for search keys, embeddings, and grouping.

    Returns "" for None/empty. Original text must be preserved by the caller.
    """
    if text is None:
        return ""
    s = str(text)
    s = s.replace(_HEH_HAMZA_ABOVE, "ه")
    s = s.translate(CHAR_MAP)
    s = _ZWNJ_RE.sub("", s)
    s = s.replace(_TATWEEL, "")
    s = s.translate(DIGIT_MAP)
    s = s.lower()
    s = _PUNCT_RE.sub("،", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def normalize_for_key(text: str | None) -> str:
    """Ultra-normalized form for dictionary keys (brand/model grouping).

    Beyond matching normalization: strips ALL whitespace, punctuation
    (،؛,.()-—–− and dashes), and quotes, so «پژو ۲۰۶» and «پژو206» map to
    one key.
    """
    s = normalize_for_matching(text)
    return re.sub(r"[\s،؛,.()\-—–−\"'«»]+", "", s)
