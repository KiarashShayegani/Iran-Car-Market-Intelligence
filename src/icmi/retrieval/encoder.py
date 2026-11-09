"""Encoder registry and lazy loading (production search path).

Model cache stays inside the repository (HF_HOME → data/models/hf-cache);
models are loaded only when actually requested, so tests and CI never
download anything. The adopted model and benchmark rationale live in
docs/internal/EMBEDDING_BENCHMARK.md.
"""

from __future__ import annotations

import os
from typing import Any

from icmi.core.paths import models_dir

# Adopted via benchmark (2026-11-10); smallest-first policy — see
# docs/internal/EMBEDDING_BENCHMARK.md for the numbers and upgrade path.
DEFAULT_MODEL_ID = "intfloat/multilingual-e5-small"

E5_QUERY_PREFIX = "query: "

# Candidate registry: family → (passage prefix, params hint). Shared with the
# benchmark so the registry has a single source of truth.
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
        "passage_prefix": None,
        "query_prefix": None,
        "params_hint": "~118M, ~470MB fp32",
    },
    "intfloat/multilingual-e5-small": {
        "passage_prefix": "passage: ",
        "query_prefix": E5_QUERY_PREFIX,
        "params_hint": "~118M, ~470MB fp32",
    },
    "intfloat/multilingual-e5-base": {
        "passage_prefix": "passage: ",
        "query_prefix": E5_QUERY_PREFIX,
        "params_hint": "~278M, ~1.1GB fp32 (NOT downloaded; upgrade path)",
    },
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": {
        "passage_prefix": None,
        "query_prefix": None,
        "params_hint": "~278M, ~1.1GB fp32",
    },
}

_encoders: dict[str, Any] = {}


def _ensure_hf_home() -> None:
    os.environ.setdefault("HF_HOME", str(models_dir() / "hf-cache"))


def model_spec(model_id: str) -> dict[str, Any]:
    if model_id not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{model_id}'. Registry: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[model_id]


def get_encoder(model_id: str = DEFAULT_MODEL_ID):
    """Load (and cache) a sentence-transformers encoder with family-aware
    query/passage prefixes. Lazily imports heavy deps."""
    if model_id in _encoders:
        return _encoders[model_id]
    spec = model_spec(model_id)

    _ensure_hf_home()
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id)

    class STAdapter:
        def __init__(self) -> None:
            self.model_id = model_id

        def encode_passages(self, texts: list[str]) -> list[list[float]]:
            prefixed = (
                [f"{spec['passage_prefix']}{t}" for t in texts]
                if spec["passage_prefix"]
                else texts
            )
            return model.encode(prefixed, batch_size=16, show_progress_bar=False).tolist()

        def encode_query(self, text: str) -> list[float]:
            if spec["query_prefix"]:
                text = f"{spec['query_prefix']}{text}"
            return model.encode([text], show_progress_bar=False)[0].tolist()

    adapter = STAdapter()
    _encoders[model_id] = adapter
    return adapter
