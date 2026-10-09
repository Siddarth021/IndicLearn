"""Embedding caching to prevent redundant neural inference (Phase 2).

Cache key incorporates text, model_name, model_version, dimension, and normalization status
to prevent accidental reuse across different models or configurations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from indiclearn.embeddings.base import EmbeddingProvider, EmbeddingResult


def compute_cache_key(
    text: str,
    model_name: str,
    dimension: int,
    normalized: bool = True,
    model_version: str = "1.0",
) -> str:
    """Generate a collision-resistant SHA-256 cache key for an embedding request."""
    header = f"{model_name}|{dimension}|{normalized}|{model_version}:"
    full_payload = header.encode("utf-8") + text.encode("utf-8")
    return hashlib.sha256(full_payload).hexdigest()


class EmbeddingCache:
    """Key-value cache for EmbeddingResult objects with optional disk persistence."""

    def __init__(self, persistence_path: Path | str | None = None) -> None:
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self._entries: dict[str, dict[str, Any]] = {}
        if self.persistence_path and self.persistence_path.exists():
            self.load()

    def get(
        self,
        text: str,
        model_name: str,
        dimension: int,
        normalized: bool = True,
        model_version: str = "1.0",
    ) -> EmbeddingResult | None:
        """Retrieve cached embedding if present."""
        key = compute_cache_key(text, model_name, dimension, normalized, model_version)
        entry = self._entries.get(key)
        if entry is None:
            return None
        return EmbeddingResult(**entry)

    def put(
        self,
        text: str,
        result: EmbeddingResult,
        model_version: str = "1.0",
    ) -> None:
        """Store an embedding in the cache."""
        key = compute_cache_key(
            text, result.model_name, result.dimension, result.normalized, model_version
        )
        self._entries[key] = result.model_dump()

    def contains(
        self,
        text: str,
        model_name: str,
        dimension: int,
        normalized: bool = True,
        model_version: str = "1.0",
    ) -> bool:
        """Check if embedding exists in cache."""
        key = compute_cache_key(text, model_name, dimension, normalized, model_version)
        return key in self._entries

    def size(self) -> int:
        """Return number of cached embeddings."""
        return len(self._entries)

    def clear(self) -> None:
        """Clear all entries."""
        self._entries.clear()

    def save(self) -> None:
        """Persist cache to disk if persistence_path is configured."""
        if not self.persistence_path:
            return
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        self.persistence_path.write_text(
            json.dumps(self._entries, ensure_ascii=False), encoding="utf-8"
        )

    def load(self) -> None:
        """Load cache from disk."""
        if not self.persistence_path or not self.persistence_path.exists():
            return
        content = self.persistence_path.read_text(encoding="utf-8")
        if content.strip():
            self._entries = json.loads(content)


class CachedEmbeddingProvider(EmbeddingProvider):
    """Wrapper that caches embeddings for any underlying EmbeddingProvider."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        cache: EmbeddingCache | None = None,
        model_version: str = "1.0",
    ) -> None:
        self._provider = provider
        self.cache = cache if cache is not None else EmbeddingCache()
        self.model_version = model_version

    @property
    def model_name(self) -> str:
        return self._provider.model_name

    @property
    def dimension(self) -> int:
        return self._provider.dimension

    def embed_documents(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed a list of texts, retrieving from cache where available."""
        results: list[EmbeddingResult | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for idx, text in enumerate(texts):
            cached = self.cache.get(
                text=text,
                model_name=self.model_name,
                dimension=self.dimension,
                normalized=True,
                model_version=self.model_version,
            )
            if cached is not None:
                results[idx] = cached
            else:
                missing_indices.append(idx)
                missing_texts.append(text)

        if missing_texts:
            fresh_embeddings = self._provider.embed_documents(missing_texts)
            for idx, fresh in zip(missing_indices, fresh_embeddings, strict=False):
                self.cache.put(texts[idx], fresh, model_version=self.model_version)
                results[idx] = fresh

        # Save to disk if configured
        self.cache.save()

        # All results are populated
        return [r for r in results if r is not None]

    def embed_query(self, text: str) -> EmbeddingResult:
        """Embed query, retrieving from cache if available."""
        cached = self.cache.get(
            text=text,
            model_name=self.model_name,
            dimension=self.dimension,
            normalized=True,
            model_version=self.model_version,
        )
        if cached is not None:
            return cached

        fresh = self._provider.embed_query(text)
        self.cache.put(text, fresh, model_version=self.model_version)
        self.cache.save()
        return fresh
