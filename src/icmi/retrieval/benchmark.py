"""Embedding benchmark harness (F3b runner; testable offline with fakes).

Principles (owner directive, 2026-11-10):
- Benchmark BEFORE committing to any large model. Candidates are evaluated
  smallest-first, one download at a time.
- Measured per candidate: retrieval quality on real Persian ads, CPU latency,
  peak RSS (stdlib `resource`), and model footprint — recorded in
  docs/internal/ before an index pipeline is built around any model.

Quality proxy (honest about limits): self-retrieval MRR — for sampled ads, a
"identity query" (title + model, WITHOUT the description) must rank the right
ad highly against the full pool of distractors; plus neighbor-agreement
(overlap@k) between candidate pairs. This is a proxy, not a human-labeled IR
evaluation; results are interpreted with that caveat.
"""

from __future__ import annotations

import resource
import time
from dataclasses import dataclass, field
from typing import Protocol

from icmi.pipeline.store import CanonicalStore
from icmi.retrieval.documents import SearchDocument, iter_documents


class Encoder(Protocol):
    """Any encoder: texts → fixed-size vectors (list[float] or ndarray-like)."""

    model_id: str

    def encode_passages(self, texts: list[str]) -> list[list[float]]: ...

    def encode_query(self, text: str) -> list[float]: ...


@dataclass
class BenchmarkResult:
    model_id: str
    mrr_at_10: float
    top1_accuracy: float
    mean_query_latency_ms: float
    mean_passage_latency_ms: float  # per passage, batched
    peak_rss_mb: float
    params_hint: str
    notes: list[str] = field(default_factory=list)


def peak_rss_mb() -> float:
    """Peak resident set size in MB (Linux/macOS; process-wide)."""
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak_kb / 1024.0, 1)


def identity_queries(docs: list[SearchDocument], sample_size: int) -> list[tuple[int, str]]:
    """Build (doc_index, query) pairs from structured facts only — the query
    deliberately EXCLUDES the description so retrieval must generalize."""
    import random

    rng = random.Random(42)  # deterministic sampling for reproducible benchmarks
    pool = [d for d in docs if d.chunk_index == 0]
    sample = rng.sample(pool, min(sample_size, len(pool)))
    queries: list[tuple[int, str]] = []
    for doc in sample:
        # normalized structured part = text up to the first newline (header)
        header = doc.normalized_text.split("\n")[0]
        queries.append((docs.index(doc), header))
    return queries


def run_benchmark(
    encoder: Encoder,
    docs: list[SearchDocument],
    sample_size: int = 50,
    top_k: int = 10,
) -> BenchmarkResult:
    """Run the benchmark protocol for one encoder over prepared documents."""
    if not docs:
        raise ValueError("No documents to benchmark — build documents first")

    t0 = time.perf_counter()
    passage_vectors = encoder.encode_passages([d.normalized_text for d in docs])
    encode_seconds = time.perf_counter() - t0
    passage_ms = round(1000 * encode_seconds / max(len(docs), 1), 2)

    queries = identity_queries(docs, sample_size)
    reciprocal_ranks: list[float] = []
    top1_hits = 0
    query_ms_total = 0.0

    for doc_index, query_text in queries:
        t0 = time.perf_counter()
        query_vector = encoder.encode_query(query_text)
        query_ms_total += time.perf_counter() - t0

        ranked = rank_by_cosine(query_vector, passage_vectors)
        rank = (
            ranked.index(doc_index) + 1
            if doc_index in ranked[:top_k]
            else None
        )
        if rank == 1:
            top1_hits += 1
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

    return BenchmarkResult(
        model_id=encoder.model_id,
        mrr_at_10=round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4),
        top1_accuracy=round(top1_hits / len(queries), 4),
        mean_query_latency_ms=round(1000 * query_ms_total / max(len(queries), 1), 2),
        mean_passage_latency_ms=passage_ms,
        peak_rss_mb=peak_rss_mb(),
        params_hint="",
    )


def rank_by_cosine(
    query_vector: list[float], passage_vectors: list[list[float]], top_k: int = 10
) -> list[int]:
    """Brute-force cosine ranking — the honest baseline for this corpus size."""
    q_norm = sum(v * v for v in query_vector) ** 0.5 or 1.0
    scored: list[tuple[float, int]] = []
    for index, vector in enumerate(passage_vectors):
        dot = sum(a * b for a, b in zip(query_vector, vector, strict=False))
        v_norm = sum(v * v for v in vector) ** 0.5 or 1.0
        scored.append((dot / (q_norm * v_norm), index))
    scored.sort(reverse=True)
    return [index for _, index in scored[:top_k]]


def load_documents(store: CanonicalStore) -> list[SearchDocument]:
    return list(iter_documents(store))
