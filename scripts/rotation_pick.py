#!/usr/bin/env python3
"""Pick the next Bama presets by coverage — for manual dispatch inputs.

Scheduled runs rotate presets deterministically (collection policy order).
For MANUAL dispatches this tool improves coverage by surfacing the brands
with the fewest collected listings first (ties broken by policy order),
so a human dispatching the pipeline can fill coverage gaps instead of
re-collecting well-stocked brands.

Mapping preset -> brand: presets named after manufacturers use the alias
table (brand-peugeot -> پژو). Model-name presets (brand-pride) are not
brands — their ads live under their manufacturer (saipa) — so the count
falls back to a title/model search for the slug; it is an approximation
for advisory purposes only.

Output: a comma-separated preset list ready for the workflow's `presets`
input, e.g. `brand-lamari,brand-fownix`.

Examples:
    python scripts/rotation_pick.py               # 2 presets (policy default)
    python scripts/rotation_pick.py --count 4     # 4 presets
    python scripts/rotation_pick.py --list        # full report incl. counts
"""

from __future__ import annotations

import argparse
import sqlite3

from icmi.core.brand_aliases import load_aliases
from icmi.core.paths import canonical_dir
from icmi.pipeline.policy import load_policy


def _preset_count(conn: sqlite3.Connection, slug: str) -> tuple[int, str]:
    """Listings count for a preset slug + how it was resolved."""
    aliases = load_aliases()
    if slug in aliases:
        brand = aliases[slug]
        n = conn.execute(
            "SELECT COUNT(*) FROM listings WHERE brand_canonical = ?",
            (brand,),
        ).fetchone()[0]
        return n, f"brand={brand}"
    like = f"%{slug}%"
    n = conn.execute(
        "SELECT COUNT(*) FROM listings WHERE title LIKE ? COLLATE NOCASE "
        "OR model_raw LIKE ? COLLATE NOCASE",
        (like, like),
    ).fetchone()[0]
    return n, "title/model search"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=None,
                        help="presets to pick (default: policy presets_per_run)")
    parser.add_argument("--list", action="store_true",
                        help="print per-preset counts instead of picking")
    args = parser.parse_args(argv)

    policy = load_policy().sources["bama"]
    presets = policy.presets
    count = args.count or policy.presets_per_run or 2

    db = canonical_dir() / "icmi.db"
    if not db.is_file():
        raise SystemExit("no authoritative DB — run an ingest first")
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        scored = []
        for preset in presets:
            n, how = _preset_count(conn, preset.removeprefix("brand-"))
            scored.append((n, preset, how))
    finally:
        conn.close()

    if args.list:
        print(f"{'preset':<20} {'listings':>9}  resolution")
        for n, preset, how in sorted(scored, key=lambda x: x[0]):
            print(f"{preset:<20} {n:>9}  {how}")
        return 0

    picked = [preset for _, preset, _ in sorted(scored, key=lambda x: x[0])[:count]]
    print(",".join(picked))
    return 0


if __name__ == "__main__":
    main()
