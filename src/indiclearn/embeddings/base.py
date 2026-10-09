"""Base protocols and domain types for embedding providers (Phase 2).

Any embedding provider (SentenceTransformers, HuggingFace, OpenAI, Mock, etc.)
must satisfy the EmbeddingProvider protocol and return structured EmbeddingResult types.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class EmbeddingResult(BaseModel):
    """Result of an embedding operation for a single piece of text."""

    vector: list[float]
    model_name: str
    dimension: int
    normalized: bool = True

    def __len__(self) -> int:
        return len(self.vector)


class EmbeddingBatchResult(BaseModel):
    """Result of batch embedding operation."""

    embeddings: list[EmbeddingResult]
    model_name: str
    dimension: int
    total_tokens: int | None = None
    batch_size: int = Field(ge=0)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding generation.

    Decouples retrieval and vector indexing from any specific model backend.
    """

    @property
    def model_name(self) -> str:
        """Name or identifier of the embedding model."""
        ...

    @property
    def dimension(self) -> int:
        """Dimensionality of the generated embedding vectors."""
        ...

    def embed_documents(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate embeddings for a list of document chunk texts.

        Empty strings or edge cases must be handled gracefully without crashing.
        """
        ...

    def embed_query(self, text: str) -> EmbeddingResult:
        """Generate an embedding for a search query.

        Some models use asymmetric prompts or prefixes for queries vs documents.
        """
        ...
