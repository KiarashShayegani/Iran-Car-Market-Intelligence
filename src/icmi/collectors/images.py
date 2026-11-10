"""Image URL variant selection.

Platform image lists contain multiple quality variants of the SAME photo
(e.g. Bama: one photo × resize,w_120/240/450/600; Sheypoor: path-sized
190x190 / 180x131). Storing every variant makes galleries download 4–5
copies of each picture. Policy (owner decision, G1 review): keep ONE URL
per distinct photo — the second-largest variant when a photo has ≥3
variants (good quality without the heaviest), otherwise the largest.

Raw payloads always keep every variant verbatim; this selection happens at
extraction time so all execution modes (local, Actions, scheduled) inherit
it, and bumping `extraction_version` marks the change.
"""

from __future__ import annotations

import re

# Width tokens: Bama uses ?...resize,w_450; Sheypoor embeds 190x190 in path.
_WIDTH_PATTERNS = (
    re.compile(r"resize,w_(\d+)"),
    re.compile(r"/(\d+)x\d+_"),
)


def _variant_width(url: str) -> int:
    for pattern in _WIDTH_PATTERNS:
        match = pattern.search(url)
        if match:
            return int(match.group(1))
    return 0  # unknown width sorts last


def _photo_identity(url: str) -> str:
    """Strip size tokens/query so variants of one photo share an identity."""
    import hashlib
    from urllib.parse import urlsplit

    split = urlsplit(url)
    path = re.sub(r"/\d+x\d+_", "/", split.path)
    return hashlib.sha1(f"{split.netloc}{path}".encode()).hexdigest()


def select_primary_images(urls: list[str]) -> list[str]:
    """One URL per distinct photo, preferred quality per photo."""
    groups: dict[str, list[str]] = {}
    for url in urls:
        groups.setdefault(_photo_identity(url), []).append(url)

    selected: list[str] = []
    for variants in groups.values():
        ranked = sorted(variants, key=_variant_width, reverse=True)
        if len(ranked) >= 3:
            # second-largest = "quality before the best"
            selected.append(ranked[1])
        else:
            selected.append(ranked[0])
    return selected
