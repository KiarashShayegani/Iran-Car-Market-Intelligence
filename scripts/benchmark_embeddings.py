#!/usr/bin/env python3
"""Run the embedding benchmark for one or more candidate models.

Smallest-first policy (owner directive): candidates are evaluated and
downloaded ONE AT A TIME; results accumulate in docs/internal/benchmark_runs/
so the final model choice is documented with numbers before any index
pipeline is built around it.

Examples:
    python scripts/benchmark_embeddings.py --model intfloat/multilingual-e5-small
    python scripts/benchmark_embeddings.py --model intfloat/multilingual-e5-base --sample 80
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from icmi.core.paths import models_dir, repo_root
from icmi.pipeline.store import CanonicalStore
from icmi.retrieval.benchmark import BenchmarkResult, load_documents, run_benchmark

# Model cache stays inside the project (owner rule: nothing outside /data).
os.environ.setdefault("HF_HOME", str(models_dir() / "hf-cache"))

# Candidate registry: smallest-first. params_hint is the expected order of
# magnitude so the report can flag surprises (e.g. unexpected RAM).
CANDIDATES: dict[str, dict] = {
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
        "params_hint": "~118M, ~470MB fp32", "prefix": None,
    },
    "intfloat/multilingual-e5-small": {
        "params_hint": "~118M, ~470MB fp32", "prefix": "passage",
    },
    "intfloat/multilingual-e5-base": {
        "params_hint": "~278M, ~1.1GB fp32", "prefix": "passage",
    },
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": {
        "params_hint": "~278M, ~1.1GB fp32", "prefix": None,
    },
}

E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "


def _load_encoder(model_id: str):
    """Load a sentence-transformers encoder with per-family prefix handling."""
    from sentence_transformers import SentenceTransformer

    spec = CANDIDATES[model_id]
    model = SentenceTransformer(model_id)

    class STAdapter:
        def __init__(self, model, model_id, prefix):
            self.model = model
            self.model_id = model_id
            self._passage_prefix = prefix

        def encode_passages(self, texts):
            prefixed = (
                [f"{self._passage_prefix} {t}" for t in texts]
                if self._passage_prefix
                else texts
            )
            return self.model.encode(prefixed, batch_size=16, show_progress_bar=False).tolist()

        def encode_query(self, text):
            if model_id.startswith("intfloat/multilingual-e5"):
                text = f"{E5_QUERY_PREFIX} {text}"
            return self.model.encode([text], show_progress_bar=False)[0].tolist()

    return STAdapter(model, model_id, spec["prefix"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="HF id; must be in CANDIDATES")
    parser.add_argument("--sample", type=int, default=50, help="identity queries to build")
    parser.add_argument("--db", default=None)
    args = parser.parse_args(argv)

    if args.model not in CANDIDATES:
        print(f"Unknown candidate '{args.model}'. Registry: {list(CANDIDATES)}", file=sys.stderr)
        return 2

    store = CanonicalStore(Path(args.db) if args.db else None)
    try:
        docs = load_documents(store)
    finally:
        store.close()
    if len(docs) < 10:
        print(f"Not enough documents ({len(docs)}) for a meaningful benchmark", file=sys.stderr)
        return 2

    print(
        f"encoding {len(docs)} documents with {args.model} "
        f"(cached under {models_dir()}/hf-cache)"
    )
    encoder = _load_encoder(args.model)
    result: BenchmarkResult = run_benchmark(encoder, docs, sample_size=args.sample)
    result.params_hint = CANDIDATES[args.model]["params_hint"]

    out_dir = repo_root() / "docs" / "internal" / "benchmark_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"{stamp}-{args.model.split('/')[-1]}.json"
    out_path.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=1), encoding="utf-8"
    )

    print(
        f"MRR@10={result.mrr_at_10}  top1={result.top1_accuracy}  "
        f"query={result.mean_query_latency_ms}ms  "
        f"passage={result.mean_passage_latency_ms}ms  "
        f"peakRSS={result.peak_rss_mb}MB"
    )
    print(f"result: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
