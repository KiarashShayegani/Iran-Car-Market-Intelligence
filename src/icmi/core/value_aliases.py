"""Canonical value groups for user-facing filter facets.

Sources publish the same concept under several spellings («اتومات» vs
«اتوماتیک», «بنزین» vs «بنزینی», ZWNJ vs space in «دنده‌ای») which fragments
the filter dropdowns. Raw values on canonical records stay verbatim — this
module only groups them at the facet/filter boundary:

- facets: distinct raw values are rolled up into canonical groups (counts
  merged) so dropdowns show one honest option per concept.
- filters: a canonical selection expands back to the raw variants that
  constitute it, so filtering keeps matching the underlying data.
- future data: unknown raw values pass through as their own group — the map
  only merges what it knows; nothing is dropped.

Extend NEW_GROUP_AFTER_OBSERVING after observing new variants — every merge
is reviewable here in Git.
"""

from __future__ import annotations

from functools import lru_cache

from icmi.retrieval.normalization import normalize_for_matching

# canonical label -> raw variants that belong to it (matching-normalized keys)
_GROUPS: dict[str, dict[str, tuple[str, ...]]] = {
    "transmission": {
        "اتوماتیک": ("اتوماتیک", "اتومات", "اتوماتیکy"),
        "دنده‌ای": ("دنده‌ای", "دنده ای", "دستی", "manual"),
    },
    "fuel": {
        "بنزینی": ("بنزینی", "بنزین"),
        "دیزلی": ("دیزلی", "گازوئیل(دیزل)", "گازوئیل", "دیزل"),
        "دوگانه‌سوز": ("دوگانه سوز", "دوگانه‌سوز"),
        "هیبریدی": ("هیبریدی",),  # mild-hybrid and بردافزا stay distinct technologies
        "هیبرید ملایم": ("هیبرید ملایم",),
        "بردافزا": ("بردافزا",),
        "پلاگین هیبرید": ("پلاگین هیبرید",),
        "برقی": ("برقی",),
    },
    "body_type": {
        "سدان": ("سدان", "سدان (سواری)"),
        "هاچبک": ("هاچبک", "هاچ بک"),
        "شاسی بلند": ("شاسی بلند",),  # ZWNJ variants normalize to the same key
        "کراس اور": ("کراس اور",),
        "کوپه": ("کوپه",),
        "ون": ("ون",),
        "وانت": ("وانت",),
        "پیکاپ": ("پیکاپ",),
    },
}


@lru_cache(maxsize=1)
def _variant_index() -> dict[str, dict[str, str]]:
    """matching-normalized variant -> (field, canonical label)."""
    index: dict[str, dict[str, str]] = {}
    for field, groups in _GROUPS.items():
        for canonical, variants in groups.items():
            for variant in variants:
                index[normalize_for_matching(variant)] = {
                    "field": field,
                    "canonical": canonical,
                }
    return index


def canonical_value(field: str, raw: str | None) -> str | None:
    """Canonical label for a raw facet value; unknown values pass through."""
    if not raw:
        return None
    hit = _variant_index().get(normalize_for_matching(raw))
    if hit and hit["field"] == field:
        return hit["canonical"]
    return str(raw).strip()


def variants_of(field: str, canonical: str) -> tuple[str, ...]:
    """Raw variants constituting a canonical selection (filter expansion).

    Includes the canonical label itself so a value stored verbatim still
    matches. Unknown labels expand to themselves only.
    """
    wanted = normalize_for_matching(canonical)
    for label, variants in _GROUPS.get(field, {}).items():
        if normalize_for_matching(label) == wanted:
            return variants
    return (canonical,)


def canonical_facet_values(field: str, raw_counts: list[tuple[str, int]]) -> list[dict]:
    """Roll raw (value, count) pairs into canonical facet options.

    Groups whose members all map to one canonical label are merged; unknown
    values surface verbatim. Returned sorted by merged count descending.
    """
    merged: dict[str, int] = {}
    for raw, count in raw_counts:
        label = canonical_value(field, raw) or str(raw).strip()
        merged[label] = merged.get(label, 0) + int(count)
    return [
        {"label": label, "count": count}
        for label, count in sorted(merged.items(), key=lambda kv: -kv[1])
    ]
