#!/usr/bin/env python3
"""Search CLI — try Persian queries over the index (semantic or hybrid).

Examples:
    python scripts/search_cli.py --query "ماشین اتوماتیک کم کارکرد"
    python scripts/search_cli.py --query "پژو ارزان" \
        --filters '{"brands": ["پژو"], "price_max": 3000000000}'
    python scripts/search_cli.py --query "دنده ای قدیمی" --top-k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from icmi.retrieval.encoder import DEFAULT_MODEL_ID, get_encoder
from icmi.retrieval.index import INDEX_FORMAT_VERSION, load_index
from icmi.retrieval.search import SearchFilters, SearchService


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--filters", default=None, help="JSON filters, e.g. '{\"price_max\": 3000000000}'"
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    args = parser.parse_args(argv)

    index_dir_path = Path(args.index_dir) if args.index_dir else None
    vectors, metadata, info = load_index(index_dir_path)
    if info.get("format_version") != INDEX_FORMAT_VERSION:
        print(
            f"Index format mismatch ({info.get('format_version')}); "
            "rebuild with build_index.py",
            file=sys.stderr,
        )
        return 2

    filters = SearchFilters(**json.loads(args.filters)) if args.filters else None
    service = SearchService(vectors, metadata, get_encoder(args.model))
    response = service.search(args.query, filters, top_k=args.top_k)

    for line in response.summary_lines():
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
