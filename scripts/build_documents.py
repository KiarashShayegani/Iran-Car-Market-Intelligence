#!/usr/bin/env python3
"""Build searchable documents from the canonical store.

Writes data/index/documents.jsonl — one SearchDocument per line, with full
traceability (listing_key, original_url, raw_payload_ref) and a matching-
normalized copy of the text. Idempotent: regenerates the file from canonical
truth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from icmi.core.paths import index_dir
from icmi.pipeline.store import CanonicalStore
from icmi.retrieval.documents import write_documents


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=None, help="alternative canonical DB path")
    parser.add_argument("--out", default=None, help="alternative output path")
    args = parser.parse_args(argv)

    store = CanonicalStore(Path(args.db) if args.db else None)
    try:
        out = Path(args.out) if args.out else index_dir() / "documents.jsonl"
        count = write_documents(store, out)
    finally:
        store.close()
    print(f"documents: {count} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
