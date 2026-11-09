"""Shared test fakes (deterministic, offline)."""

from __future__ import annotations


class HashingEncoder:
    """Deterministic char-trigram hashing encoder — no model, no network.

    Adjacent texts share n-grams so similarity is meaningful; dimensions are
    hashed into a small fixed vector and L2-normalized.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.model_id = "fake-hashing"

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        cleaned = text.replace(" ", "")
        for i in range(len(cleaned) - 2):
            trigram = cleaned[i : i + 3]
            vec[hash(trigram) % self.dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def encode_query(self, text: str) -> list[float]:
        return self._vector(text)
