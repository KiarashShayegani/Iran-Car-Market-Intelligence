#!/usr/bin/env python3
"""Export analytical Parquet snapshots from the canonical store.

Examples:
    python scripts/export_analytics.py
    python scripts/export_analytics.py --db path/to/icmi.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from icmi.pipeline.analytical import export_analytics
from icmi.pipeline.store import CanonicalStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=None, help="alternative canonical DB path")
    parser.add_argument("--out", default=None, help="alternative output directory")
    args = parser.parse_args(argv)

    store = CanonicalStore(Path(args.db) if args.db else None)
    try:
        result = export_analytics(store, Path(args.out) if args.out else None)
    finally:
        store.close()
    print(f"listings: {result['listings_rows']} rows -> {result['listings']}")
    print(f"observations: {result['observation_rows']} rows -> {result['price_observations']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
