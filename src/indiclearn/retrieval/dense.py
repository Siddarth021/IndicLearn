"""Dense vector retriever (Phase 2).

Connects an EmbeddingProvider and a VectorStore to satisfy the Retriever protocol.
"""

from __future__ import annotations

from typing import Any

from indiclearn.embeddings.base import EmbeddingProvider
from indiclearn.retrieval.base import RetrievalResult, Retriever, VectorStore


class DenseRetriever(Retriever):
    """Retrieves document chunks using dense semantic embeddings and a vector store."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Embed the query and retrieve closest chunks from the vector store."""
        if not query.strip() or self.vector_store.count() == 0:
            return []

        query_emb = self.embedding_provider.embed_query(query)
        return self.vector_store.search(
            query_vector=query_emb.vector,
            top_k=top_k,
            filters=filters,
        )
