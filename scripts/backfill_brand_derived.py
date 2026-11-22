#!/usr/bin/env python3
"""Backfill derived brand columns (brand_key / brand_canonical) on listings.

Pure derived-field refresh — brand_raw stays verbatim. Run after extending
config/brand_aliases.yaml so existing rows pick up new canonical mappings
(future ingests derive them automatically at upsert). Idempotent: only rows
whose derived value would actually change are updated.

Usage:
    python scripts/backfill_brand_derived.py [--dry-run]
"""

from __future__ import annotations

import argparse

from icmi.core.brand_aliases import canonical_brand
from icmi.core.paths import repo_root
from icmi.pipeline.store import CanonicalStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="report planned updates without writing")
    args = parser.parse_args()

    store = CanonicalStore()
    rows = store.query_df(
        "SELECT listing_key, brand_raw, brand_key, brand_canonical FROM listings "
        "WHERE brand_raw IS NOT NULL"
    )
    updates: list[tuple[str, str, str]] = []
    for _, row in rows.iterrows():
        key, canonical = canonical_brand(row["brand_raw"])
        if not key:
            continue
        if key != row["brand_key"] or canonical != row["brand_canonical"]:
            updates.append((row["listing_key"], key, canonical))

    print(f"listings scanned: {len(rows)}")
    print(f"rows needing update: {len(updates)}")
    if not updates:
        print("already consistent — nothing to do")
        store.close()
        return 0

    if args.dry_run:
        from collections import Counter

        by_canonical = Counter((u[2]) for u in updates)
        for label, count in by_canonical.most_common(20):
            print(f"  would map -> {label}: {count}")
        store.close()
        return 0

    conn = store._conn
    conn.executemany(
        "UPDATE listings SET brand_key = ?, brand_canonical = ? WHERE listing_key = ?",
        [(k, c, lk) for lk, k, c in updates],
    )
    conn.commit()
    print(f"updated {len(updates)} rows (derived columns only; brand_raw untouched)")
    store.close()

    db_path = repo_root() / "data" / "canonical" / "icmi.db"
    print(f"database: {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
