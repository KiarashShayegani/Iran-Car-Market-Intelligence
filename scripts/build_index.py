#!/usr/bin/env python3
"""Build/update the vector index from canonical data.

Uses incremental embedding by default (H4): chunks whose content hash is
unchanged keep their existing vectors — only new/changed chunks are encoded.
`--full` forces a complete re-encode. Model: e5-small (see
docs/internal/EMBEDDING_BENCHMARK.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from icmi.core.paths import index_dir
from icmi.pipeline.store import CanonicalStore
from icmi.retrieval.encoder import DEFAULT_MODEL_ID
from icmi.retrieval.incremental import incremental_index_build


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--full", action="store_true", help="ignore existing index (full re-encode)"
    )
    args = parser.parse_args(argv)

    store = CanonicalStore(Path(args.db) if args.db else None)
    try:
        result = incremental_index_build(
            store, model_id=args.model,
            out_dir=index_dir(), full_rebuild=args.full,
        )
    finally:
        store.close()
    print(
        f"index updated: {result['total_chunks']} chunks "
        f"({result['carried_over']} carried over, {result['newly_encoded']} re-encoded)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
