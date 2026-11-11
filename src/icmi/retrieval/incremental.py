"""Incremental index build: re-embed only new/changed chunks.

Content-hash strategy (H4): every chunk carries a sha256 of its normalized
text + document version. The incremental build matches new chunks against
the existing index by (listing_key, chunk_index); vectors for unchanged
chunks are carried over, only new/changed chunks are encoded. Deleting the
index file rebuilds everything from scratch — the full-rebuild path is the
correctness baseline, incrementality is an optimization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from icmi.retrieval.encoder import DEFAULT_MODEL_ID, get_encoder
from icmi.retrieval.index import build_index_metadata, save_index


def incremental_index_build(
    store,
    model_id: str = DEFAULT_MODEL_ID,
    out_dir: Path | None = None,
    full_rebuild: bool = False,
) -> dict:
    """Update the index in place; returns counts for the report."""
    from icmi.core.paths import index_dir as idx_dir

    out = out_dir or idx_dir()
    chunks = build_index_metadata(store)
    meta_path = out / "index_metadata.parquet"
    vectors_path = out / "vectors.npy"

    existing_vectors: np.ndarray | None = None
    existing_meta: pd.DataFrame | None = None
    if meta_path.is_file() and vectors_path.is_file():
        try:
            existing_meta = pd.read_parquet(meta_path)
            existing_vectors = np.load(vectors_path)
            if len(existing_meta) != len(existing_vectors):
                existing_meta, existing_vectors = None, None  # corrupt — full rebuild
        except (ValueError, OSError):
            existing_meta, existing_vectors = None, None

    if full_rebuild or (
        existing_meta is not None and "content_hash" not in existing_meta.columns
    ):
        existing_meta, existing_vectors = None, None  # forced or pre-H4 index

    # match: (listing_key, chunk_index) -> vector + content_hash
    old_by_key: dict[tuple, np.ndarray] = {}
    old_hash_by_key: dict[tuple, str] = {}
    if existing_meta is not None and existing_vectors is not None:
        for i, row in existing_meta.iterrows():
            key = (row["listing_key"], int(row["chunk_index"]))
            old_by_key[key] = existing_vectors[i]
            old_hash_by_key[key] = row["content_hash"]

    encoder = get_encoder(model_id)
    carry: list[np.ndarray] = []
    encode_positions: list[int] = []
    for pos, chunk in enumerate(chunks):
        key = (chunk.listing_key, chunk.chunk_index)
        old_vec = old_by_key.get(key)
        if old_vec is not None and old_hash_by_key.get(key) == chunk.content_hash:
            carry.append((pos, old_vec))
        else:
            encode_positions.append(pos)

    new_vectors_list: list[np.ndarray] = [None] * len(chunks)  # type: ignore[list-item]
    for pos, vec in carry:
        new_vectors_list[pos] = vec
    if encode_positions:
        texts = [chunks[p].normalized_text for p in encode_positions]
        encoded = np.asarray(encoder.encode_passages(texts))
        for i, p in enumerate(encode_positions):
            new_vectors_list[p] = encoded[i]
    vectors = np.asarray(new_vectors_list)

    save_index(vectors, chunks, model_id, out)
    return {
        "total_chunks": len(chunks),
        "carried_over": len(carry),
        "newly_encoded": len(encode_positions),
    }
