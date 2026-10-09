"""Deterministic MockEmbeddingProvider for offline unit tests (Phase 2)."""

from __future__ import annotations

import hashlib
import math

from indiclearn.embeddings.base import EmbeddingProvider, EmbeddingResult


class MockEmbeddingProvider(EmbeddingProvider):
    """Generates deterministic unit-normalized pseudo-embeddings for testing without neural models."""

    def __init__(self, model_name: str = "mock-embedding-v1", dimension: int = 64) -> None:
        self._model_name = model_name
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _generate_vector(self, text: str) -> list[float]:
        if not text:
            return [0.0] * self._dimension

        # Generate deterministic pseudo-random float vector from text hash
        raw_vals: list[float] = []
        for i in range(self._dimension):
            h = hashlib.sha256(f"{text}:{i}".encode()).digest()
            val = (int.from_bytes(h[:4], "big") / (2**32 - 1)) * 2.0 - 1.0
            raw_vals.append(val)

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in raw_vals))
        if norm > 0:
            return [v / norm for v in raw_vals]
        return [0.0] * self._dimension

    def embed_documents(self, texts: list[str]) -> list[EmbeddingResult]:
        results: list[EmbeddingResult] = []
        for text in texts:
            vec = self._generate_vector(text)
            results.append(
                EmbeddingResult(
                    vector=vec,
                    model_name=self._model_name,
                    dimension=self._dimension,
                    normalized=True,
                )
            )
        return results

    def embed_query(self, text: str) -> EmbeddingResult:
        vec = self._generate_vector(text)
        return EmbeddingResult(
            vector=vec,
            model_name=self._model_name,
            dimension=self._dimension,
            normalized=True,
        )
