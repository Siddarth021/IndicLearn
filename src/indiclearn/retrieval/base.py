"""Base protocols and domain types for retrieval (Phase 2).

Defines typed RetrievalResult, VectorStore protocol, Retriever protocol,
and Reranker protocol. Keep provider-specific details out of the rest of the application.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from indiclearn.domain.documents import DocumentChunk
from indiclearn.embeddings.base import EmbeddingResult


class RetrievalResult(BaseModel):
    """Result of retrieving a single chunk for a query."""

    chunk_id: str
    document_id: str
    score: float
    rank: int = Field(ge=1)
    text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector indexing and approximate or exact nearest neighbor search."""

    def add(self, chunks: list[DocumentChunk], embeddings: list[EmbeddingResult]) -> None:
        """Index a batch of chunks along with their corresponding embeddings."""
        ...

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Search the index for chunks closest to query_vector, applying metadata filters."""
        ...

    def delete(self, document_id: str) -> int:
        """Delete all chunks belonging to a document_id. Returns number of removed chunks."""
        ...

    def count(self) -> int:
        """Return the total number of unique chunks currently stored in the index."""
        ...

    def save(self, path: Path | str) -> None:
        """Persist index and metadata to disk."""
        ...

    def load(self, path: Path | str) -> None:
        """Load index and metadata from disk."""
        ...


@runtime_checkable
class Retriever(Protocol):
    """Protocol for any retrieval algorithm (lexical, dense, hybrid)."""

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve top_k matching chunks for query, applying optional metadata filters."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Optional protocol for reranking candidate chunks."""

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Rerank candidate chunks according to secondary scoring model."""
        ...
