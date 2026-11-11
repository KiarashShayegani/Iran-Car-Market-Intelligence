"""Manufacturer alias map — deterministic cross-platform brand normalization.

config/brand_aliases.yaml maps normalized variants (Latin spellings from
Hamrah Mechanic, etc.) to a canonical Persian label. brand_raw on every
listing stays verbatim; only the derived brand_key/brand_canonical columns
are normalized. Extend the YAML after observing new variants — every alias
is reviewable in Git.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from icmi.retrieval.normalization import normalize_for_key

_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "brand_aliases.yaml"


@lru_cache(maxsize=1)
def load_aliases(path: str | None = None) -> dict[str, str]:
    """normalized-variant -> canonical Persian label."""
    p = Path(path) if path else _DEFAULT_PATH
    if not p.is_file():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    aliases = raw.get("aliases", {}) or {}
    return {
        normalize_for_key(variant): canonical
        for variant, canonical in aliases.items()
        if canonical
    }


def canonical_brand(brand_raw: str | None) -> tuple[str, str]:
    """Returns (brand_key, brand_canonical) for a raw brand string.

    brand_key = normalized key of the canonical label (or of the raw value
    when no alias applies). brand_canonical = the alias target (or the raw
    value verbatim when no alias applies).
    """
    from icmi.retrieval.normalization import normalize_for_key

    if not brand_raw or not str(brand_raw).strip():
        return "", ""
    key = normalize_for_key(str(brand_raw))
    canonical = load_aliases().get(key, str(brand_raw).strip())
    return normalize_for_key(canonical), canonical


def alias_report(store) -> dict:
    """Brand-alias coverage report (Phase 0): raw variants, canonical groups,
    mapped/unmapped counts, and conflicts (two canonical labels normalizing
    to the same brand_key). Deterministic — no fuzzy merging."""
    rows = store.query_df(
        "SELECT brand_raw, COUNT(*) AS n FROM listings GROUP BY brand_raw ORDER BY n DESC"
    )
    aliases = load_aliases()
    mapped, unmapped = [], []
    canonical_groups: dict[str, dict] = {}
    key_conflicts: dict[str, set] = {}

    for _, row in rows.iterrows():
        raw = str(row["brand_raw"])
        n = int(row["n"])
        key, canonical = canonical_brand(raw)
        entry = {"raw": raw, "count": n, "canonical": canonical, "key": key}
        is_mapped = key in aliases or normalize_for_key(raw) in aliases
        (mapped if is_mapped else unmapped).append(entry)
        group = canonical_groups.setdefault(key, {"canonical": canonical, "count": 0, "raws": []})
        group["count"] += n
        group["raws"].append({"raw": raw, "count": n})
        key_conflicts.setdefault(key, set()).add(canonical)

    conflicts = [
        {"key": k, "labels": sorted(labels - {""})}
        for k, labels in key_conflicts.items()
        if len(labels - {""}) > 1
    ]
    return {
        "raw_variants": len(rows),
        "canonical_brands": len(canonical_groups),
        "mapped_variants": len(mapped),
        "unmapped_variants": len(unmapped),
        "conflicts": conflicts,
        "mapped": sorted(mapped, key=lambda e: -e["count"]),
        "unmapped": sorted(unmapped, key=lambda e: -e["count"]),
        "groups": sorted(
            ({"key": k, **g} for k, g in canonical_groups.items()),
            key=lambda g: -g["count"],
        ),
    }
