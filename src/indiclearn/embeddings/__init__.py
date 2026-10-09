"""Embeddings package for IndicLearn (Phase 2)."""

from indiclearn.embeddings.base import EmbeddingBatchResult, EmbeddingProvider, EmbeddingResult
from indiclearn.embeddings.cache import CachedEmbeddingProvider, EmbeddingCache
from indiclearn.embeddings.mock import MockEmbeddingProvider
from indiclearn.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider

__all__ = [
    "CachedEmbeddingProvider",
    "EmbeddingBatchResult",
    "EmbeddingCache",
    "EmbeddingProvider",
    "EmbeddingResult",
    "MockEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
]
