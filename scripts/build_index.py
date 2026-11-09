#!/usr/bin/env python3
"""Build the vector index from canonical data (documents + metadata + vectors).

Uses the adopted embedding model (e5-small — see docs/internal/
EMBEDDING_BENCHMARK.md). Idempotent: regenerates index artifacts from
canonical truth. Vectors+metadata land in data/index/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from icmi.core.paths import index_dir
from icmi.pipeline.store import CanonicalStore
from icmi.retrieval.encoder import DEFAULT_MODEL_ID, get_encoder
from icmi.retrieval.index import build_index_metadata, save_index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    args = parser.parse_args(argv)

    store = CanonicalStore(Path(args.db) if args.db else None)
    try:
        chunks = build_index_metadata(store)
    finally:
        store.close()
    if not chunks:
        print(
            "No documents to index — run scripts/build_documents.py after collecting",
            file=sys.stderr,
        )
        return 2

    print(f"encoding {len(chunks)} chunks with {args.model}")
    encoder = get_encoder(args.model)
    vectors = np.asarray(encoder.encode_passages([c.normalized_text for c in chunks]))
    paths = save_index(vectors, chunks, args.model, index_dir())

    info = {
        "chunks": len(chunks),
        "dim": vectors.shape[1],
        "vectors": str(paths["vectors"]),
        "metadata": str(paths["metadata"]),
    }
    print(f"index built: {info}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
