#!/usr/bin/env python3
"""Backfill Divar list-level fields that the original parser missed.

Three corrections (audit 2026-11-28 — divar P0 was 33.3%):
1. original_url  — constructible from the token (divar.ir/v/<token>); was 0%
                   for all divar rows, breaking the «منبع» link + P0.
2. mileage_raw / mileage_km — the list row's top chip carries MILEAGE text
                   («۳۰۰,۰۰۰ کیلومتر» / «صفر کیلومتر»); the old parser filed
                   it under source_published_at. Parsed only when the text
                   actually is a mileage pattern — never fabricated.
3. source_published_at — the mislabeled mileage/condition text is NULLed
                   (list rows carry no posting date; the text survives in
                   source_attributes.top_description_text).

Idempotent; brand-new rows ingest correctly via the fixed collector.

Usage: python scripts/backfill_divar_list_fields.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re

from icmi.pipeline.normalize import normalize_mileage_km
from icmi.pipeline.store import CanonicalStore

_MILEAGE_RE = re.compile(
    r"(صفر|[\u06F0-\u06F9]+(?:[٬,][\u06F0-\u06F9]{3})*|[0-9]+(?:,[0-9]{3})*)\s*کیلومتر"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    store = CanonicalStore()
    rows = store.query_df(
        """
        SELECT listing_key, original_url, mileage_raw, mileage_km,
               source_published_at, source_attributes_json
        FROM listings WHERE source_platform = 'divar'
        """
    )
    total = len(rows)
    need_url = int(rows["original_url"].isna().sum())
    need_mileage = 0
    parsed_mileage = 0
    need_pub_null = 0

    mileage_updates: list[tuple[str, str, int]] = []
    import pandas as pd

    for _, row in rows.iterrows():
        if pd.isna(row["mileage_raw"]) and pd.isna(row["mileage_km"]):
            need_mileage += 1
            attrs = row["source_attributes_json"]
            if attrs:
                try:
                    a = json.loads(attrs)
                except (ValueError, TypeError):
                    a = {}
                text = str(a.get("top_description_text") or "").strip()
                if text and _MILEAGE_RE.search(text):
                    km = normalize_mileage_km(text)
                    if km is not None:
                        parsed_mileage += 1
                        mileage_updates.append((row["listing_key"], text, km))
        pub = row["source_published_at"]
        if pub is not None and "کیلومتر" not in str(pub):
            need_pub_null += 1  # genuine condition texts also get NULLed below

    print(f"divar listings scanned: {total}")
    print(f"missing original_url:        {need_url}")
    print(f"mileage parseable from chip: {parsed_mileage} of {need_mileage} unparsed")
    print(f"source_published_at to NULL: {total - need_pub_null} mileage texts "
          f"(+{need_pub_null} other non-date texts)")

    if args.dry_run:
        print("dry-run — no changes written")
        store.close()
        return 0

    conn = store._conn
    # 1. original_url from token
    conn.execute(
        """
        UPDATE listings
        SET original_url = 'https://divar.ir/v/' || replace(listing_key, 'divar:', '')
        WHERE source_platform = 'divar' AND original_url IS NULL
        """
    )
    # 2. mileage from the top chip (parsed at scan time)
    conn.executemany(
        "UPDATE listings SET mileage_raw = ?, mileage_km = ? WHERE listing_key = ?",
        [(text, km, key) for key, text, km in mileage_updates],
    )
    # 3. NULL the mislabeled source_published_at (text kept in source_attributes)
    conn.execute(
        "UPDATE listings SET source_published_at = NULL WHERE source_platform = 'divar'"
    )
    conn.commit()

    check = store.query_df(
        """
        SELECT COUNT(*) AS n, COUNT(original_url) AS u, COUNT(mileage_km) AS m,
               COUNT(source_published_at) AS p
        FROM listings WHERE source_platform = 'divar'
        """
    ).iloc[0]
    print(f"after: listings={check['n']} original_url={check['u']} "
          f"mileage_km={check['m']} source_published_at={check['p']}")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
