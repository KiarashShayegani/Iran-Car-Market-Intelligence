"""Document preparation: canonical listings → searchable RAG documents.

Composition (Persian-first): structured vehicle facts + the verbatim
description. Every document and chunk retains full traceability:
listing_key, original_url, raw_payload_ref, extraction_version.

Text fields: ``text`` (original composed) and ``normalized_text`` (matching
form from retrieval/normalization.py). Raw source text inside the document is
verbatim — the LLM layer must be able to distinguish source content from
composed context.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass

from icmi.core.records import has_value
from icmi.pipeline.store import CanonicalStore
from icmi.retrieval.normalization import normalize_for_matching

# Composed docs longer than this get their description split into chunks;
# each chunk repeats the structured header so embeddings keep vehicle context.
MAX_DOC_CHARS = 1200

# Document schema version — bumped when composition changes materially.
DOCUMENT_VERSION = "doc-v1"

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!؟\n])\s*")


@dataclass
class SearchDocument:
    listing_key: str
    chunk_index: int  # 0 = whole doc / first chunk
    chunk_count: int
    text: str
    normalized_text: str
    original_url: str | None
    raw_payload_ref_json: str
    extraction_version: str
    document_version: str = DOCUMENT_VERSION


def compose_text(row) -> str:
    """Compose the searchable text for one canonical listing row."""
    facts: list[str] = []

    def add(value, *, prefix: str | None = None) -> None:
        if has_value(value):
            facts.append(f"{prefix} {value}" if prefix else str(value))

    add(row["brand_raw"])
    add(row["model_raw"])
    add(row["trim_raw"])
    add(row["year_raw"], prefix="سال")
    add(row["mileage_raw"], prefix="کارکرد")
    add(row["price_raw"], prefix="قیمت")
    add(row["location_raw"])
    add(row["fuel_raw"])
    add(row["transmission_raw"])
    add(row["body_type_raw"])
    add(row["body_condition_raw"])
    add(row["color_raw"])

    header = "، ".join(facts)
    description = row["description"] if has_value(row["description"]) else ""
    if header and description:
        return f"{header}\n{description}"
    return header or description or (row["title"] or "")


def iter_documents(store: CanonicalStore) -> Iterator[SearchDocument]:
    """Yield search documents (with chunking) for every canonical listing."""
    df = store.query_df(
        """
        SELECT listing_key, original_url, raw_payload_ref_json, extraction_version,
               title, brand_raw, model_raw, trim_raw, year_raw, mileage_raw,
               price_raw, location_raw, fuel_raw, transmission_raw,
               body_type_raw, body_condition_raw, color_raw, description
        FROM listings ORDER BY listing_key
        """
    )
    for _, row in df.iterrows():
        text = compose_text(row)
        if not text.strip():
            continue
        chunks = _chunk_text(text, MAX_DOC_CHARS)
        for index, chunk in enumerate(chunks):
            yield SearchDocument(
                listing_key=row["listing_key"],
                chunk_index=index,
                chunk_count=len(chunks),
                text=chunk,
                normalized_text=normalize_for_matching(chunk),
                original_url=row["original_url"],
                raw_payload_ref_json=row["raw_payload_ref_json"],
                extraction_version=row["extraction_version"],
            )


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Split on sentence boundaries (. / ؟ / ! / newline) when over the cap."""
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for piece in _SENTENCE_SPLIT_RE.split(text):
        if not piece:
            continue
        if length + len(piece) > max_chars and current:
            chunks.append("".join(current))
            current, length = [], 0
        current.append(piece)
        length += len(piece)
    if current:
        chunks.append("".join(current))
    return [chunk for chunk in chunks if chunk.strip()]


def write_documents(store: CanonicalStore, out_path) -> int:
    """Write documents.jsonl (one SearchDocument per line). Returns count."""
    count = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for doc in iter_documents(store):
            fh.write(json.dumps(asdict(doc), ensure_ascii=False, default=str) + "\n")
            count += 1
    return count
