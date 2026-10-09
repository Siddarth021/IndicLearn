"""In-memory VectorStore implementation using NumPy (Phase 2).

Provides an exact cosine-similarity vector store with zero third-party C/C++ dependencies.
Ideal for unit testing, development, and fallback operation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from indiclearn.domain.documents import DocumentChunk
from indiclearn.embeddings.base import EmbeddingResult
from indiclearn.retrieval.base import RetrievalResult, VectorStore
from indiclearn.retrieval.exceptions import IndexCorruptError, IndexVersionMismatchError


class InMemoryVectorStore(VectorStore):
    """In-memory vector store backed by numpy arrays and metadata sidecars."""

    def __init__(
        self,
        embedding_model: str = "default",
        embedding_dimension: int = 384,
        index_version: int = 1,
    ) -> None:
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.index_version = index_version
        self._vectors: list[list[float]] = []
        self._chunks: list[dict[str, Any]] = []
        self._seen_chunk_ids: set[str] = set()

    def add(self, chunks: list[DocumentChunk], embeddings: list[EmbeddingResult]) -> None:
        """Add chunks and embeddings, ignoring duplicates by chunk_id."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch between chunks ({len(chunks)}) and embeddings ({len(embeddings)})"
            )

        for chunk, emb in zip(chunks, embeddings, strict=False):
            if chunk.chunk_id in self._seen_chunk_ids:
                continue

            if emb.dimension != self.embedding_dimension:
                raise ValueError(
                    f"Vector dimension {emb.dimension} does not match store dimension {self.embedding_dimension}"
                )

            self._seen_chunk_ids.add(chunk.chunk_id)
            self._vectors.append(emb.vector)
            self._chunks.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "subject_id": chunk.subject_id,
                    "page_number": chunk.page_number,
                    "section": chunk.section,
                    "text": chunk.text,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "language": chunk.language,
                    "concept_ids": chunk.concept_ids,
                }
            )

    def _matches_filters(self, record: dict[str, Any], filters: dict[str, Any] | None) -> bool:
        if not filters:
            return True
        for key, expected in filters.items():
            if key == "document_id" and record["document_id"] != expected:
                return False
            if key == "subject_id" and record["subject_id"] != expected:
                return False
            if key == "language" and record.get("language") != expected:
                return False
            if key == "concept_id":
                concept_ids = record.get("concept_ids", [])
                if isinstance(expected, list):
                    if not any(cid in concept_ids for cid in expected):
                        return False
                elif expected not in concept_ids:
                    return False
        return True

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Search the store for nearest vectors, with optional metadata filtering."""
        if not self._vectors:
            return []

        matrix = np.array(self._vectors, dtype=np.float32)
        q = np.array(query_vector, dtype=np.float32)

        # Norms
        matrix_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix_norm[matrix_norm == 0] = 1.0
        normalized_matrix = matrix / matrix_norm

        q_norm = np.linalg.norm(q)
        q_norm = 1.0 if q_norm == 0 else q_norm
        normalized_q = q / q_norm

        # Cosine similarity
        sims = np.dot(normalized_matrix, normalized_q)

        # Collect and filter
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for idx, (sim, rec) in enumerate(zip(sims, self._chunks, strict=False)):
            if not self._matches_filters(rec, filters):
                continue
            scored.append((float(sim), idx, rec))

        # Sort descending by similarity, ascending by chunk_id for deterministic ties
        scored.sort(key=lambda x: (-x[0], x[2]["chunk_id"]))

        results: list[RetrievalResult] = []
        for rank, (score, _, rec) in enumerate(scored[:top_k], start=1):
            results.append(
                RetrievalResult(
                    chunk_id=rec["chunk_id"],
                    document_id=rec["document_id"],
                    score=score,
                    rank=rank,
                    text=rec["text"],
                    metadata={
                        "subject_id": rec["subject_id"],
                        "page_number": rec["page_number"],
                        "section": rec.get("section"),
                        "char_start": rec["char_start"],
                        "char_end": rec["char_end"],
                        "language": rec.get("language"),
                        "concept_ids": rec.get("concept_ids", []),
                    },
                )
            )
        return results

    def delete(self, document_id: str) -> int:
        """Delete all chunks for a document_id."""
        keep_vectors: list[list[float]] = []
        keep_chunks: list[dict[str, Any]] = []
        deleted_count = 0

        for vec, rec in zip(self._vectors, self._chunks, strict=False):
            if rec["document_id"] == document_id:
                self._seen_chunk_ids.discard(rec["chunk_id"])
                deleted_count += 1
            else:
                keep_vectors.append(vec)
                keep_chunks.append(rec)

        self._vectors = keep_vectors
        self._chunks = keep_chunks
        return deleted_count

    def count(self) -> int:
        return len(self._chunks)

    def save(self, path: Path | str) -> None:
        """Save vector store to disk as a JSON bundle."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "index_version": self.index_version,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "count": len(self._chunks),
            "vectors": self._vectors,
            "chunks": self._chunks,
        }
        save_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def load(self, path: Path | str) -> None:
        """Load vector store from disk with version validation."""
        load_path = Path(path)
        if not load_path.exists():
            raise IndexCorruptError(f"Index file not found: {load_path}")

        try:
            content = json.loads(load_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise IndexCorruptError(f"Failed to parse index JSON at {load_path}: {e}") from e

        persisted_model = content.get("embedding_model", "")
        persisted_dim = content.get("embedding_dimension", 0)

        # Validate index version compatibility
        if persisted_model != self.embedding_model or persisted_dim != self.embedding_dimension:
            raise IndexVersionMismatchError(
                index_model=persisted_model,
                index_dim=persisted_dim,
                expected_model=self.embedding_model,
                expected_dim=self.embedding_dimension,
            )

        self.index_version = content.get("index_version", 1)
        self._vectors = content.get("vectors", [])
        self._chunks = content.get("chunks", [])
        self._seen_chunk_ids = {c["chunk_id"] for c in self._chunks}
