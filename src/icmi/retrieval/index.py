"""Vector index over prepared documents: numpy vectors + Parquet metadata.

Design guarantees:
- **Filters are pre-applied**: ranking runs only on the subset that passes
  the filters, computed over the ENTIRE corpus metadata — irrelevant listings
  can never sneak into results via post-filtering.
- **Traceability survives retrieval**: every index row carries listing_key,
  original_url, and raw_payload_ref_json.
- Zero-results under restrictive filters are honest (empty list), never a
  silently-loosened filter.

Scale note: brute-force cosine is the honest baseline for this corpus size
(~150 MB of vectors at 100k docs × 384 dims — fits RAM trivially). ANN or
sqlite-vec becomes justified only when profiling says so.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from icmi.core.paths import index_dir
from icmi.retrieval.documents import DOCUMENT_VERSION

INDEX_FORMAT_VERSION = "index-v1"


@dataclass
class IndexedChunk:
    """One indexed row: document chunk + listing-level filter metadata."""

    content_hash: str  # sha256 of normalized_text — incremental embedding key
    listing_key: str
    source_platform: str
    chunk_index: int
    chunk_count: int
    text: str
    normalized_text: str
    original_url: str | None
    raw_payload_ref_json: str
    extraction_version: str
    document_version: str
    brand_raw: str | None
    model_raw: str | None
    trim_raw: str | None
    fuel_raw: str | None
    transmission_raw: str | None
    body_type_raw: str | None
    body_condition_raw: str | None
    brand_key: str  # normalize_for_key(brand_raw) — filter matching key
    model_key: str
    year_num: int | None
    year_calendar: str | None
    price_amount: int | None
    mileage_km: int | None
    location_raw: str | None
    image_urls_json: str | None
    first_seen_at: str


def _metadata_frame(store) -> pd.DataFrame:
    """Listing-level filterable metadata — derived fields come straight from
    the canonical store (single derivation point, populated at ingest)."""
    df = store.query_df(
        """
        SELECT listing_key, original_url, raw_payload_ref_json, extraction_version,
               brand_raw, model_raw, year_raw, mileage_raw, price_amount, location_raw,
               brand_key, model_key, year_num, year_calendar, mileage_km,
               image_urls_json, trim_raw, fuel_raw, transmission_raw,
               body_type_raw, body_condition_raw, first_seen_at
        FROM listings
        """
    )
    df["source_platform"] = df["listing_key"].str.split(":").str[0]
    return df


def build_index_metadata(store) -> list[IndexedChunk]:
    """Join documents (chunk-level) with listing metadata → IndexedChunk list."""
    listings = _metadata_frame(store).set_index("listing_key")
    from icmi.retrieval.documents import iter_documents

    chunks: list[IndexedChunk] = []
    for doc in iter_documents(store):
        listing = listings.loc[doc.listing_key]
        chunks.append(
            IndexedChunk(
                content_hash=hashlib.sha256(
                    (doc.document_version + "|" + doc.normalized_text).encode()
                ).hexdigest()[:16],
                listing_key=doc.listing_key,
                source_platform=listing["source_platform"],
                chunk_index=doc.chunk_index,
                chunk_count=doc.chunk_count,
                text=doc.text,
                normalized_text=doc.normalized_text,
                original_url=doc.original_url,
                raw_payload_ref_json=doc.raw_payload_ref_json,
                extraction_version=doc.extraction_version,
                document_version=doc.document_version,
                brand_raw=listing["brand_raw"],
                model_raw=listing["model_raw"],
                trim_raw=listing["trim_raw"],
                fuel_raw=listing["fuel_raw"],
                transmission_raw=listing["transmission_raw"],
                body_type_raw=listing["body_type_raw"],
                body_condition_raw=listing["body_condition_raw"],
                brand_key=listing["brand_key"],
                model_key=listing["model_key"],
                year_num=listing["year_num"],
                year_calendar=listing["year_calendar"],
                price_amount=listing["price_amount"],
                mileage_km=listing["mileage_km"],
                location_raw=listing["location_raw"],
                image_urls_json=listing["image_urls_json"],
                first_seen_at=listing["first_seen_at"],
            )
        )
    return chunks


def save_index(
    vectors: np.ndarray,
    chunks: list[IndexedChunk],
    model_id: str,
    out_dir: Path | None = None,
) -> dict[str, Path]:
    """Persist vectors + metadata + build info atomically into the index dir."""
    out = out_dir or index_dir()
    out.mkdir(parents=True, exist_ok=True)
    vectors_path = out / "vectors.npy"
    metadata_path = out / "index_metadata.parquet"
    info_path = out / "index_meta.json"

    # Atomic writes: temp files then rename — readers never see partial index.
    np.save(vectors_path.with_suffix(".tmp.npy"), vectors)
    pd.DataFrame([vars(c) for c in chunks]).to_parquet(
        metadata_path.with_suffix(".tmp.parquet"), index=False
    )
    vectors_path.with_suffix(".tmp.npy").replace(vectors_path)
    metadata_path.with_suffix(".tmp.parquet").replace(metadata_path)
    info_path.write_text(
        json.dumps(
            {
                "format_version": INDEX_FORMAT_VERSION,
                "model_id": model_id,
                "document_version": DOCUMENT_VERSION,
                "built_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
                "vector_count": int(vectors.shape[0]),
                "dim": int(vectors.shape[1]),
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    return {"vectors": vectors_path, "metadata": metadata_path, "info": info_path}


def load_index(out_dir: Path | None = None) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Load a persisted index (vectors, metadata frame, info dict)."""
    out = out_dir or index_dir()
    vectors = np.load(out / "vectors.npy")
    metadata = pd.read_parquet(out / "index_metadata.parquet")
    info = json.loads((out / "index_meta.json").read_text(encoding="utf-8"))
    return vectors, metadata, info
