"""FAISS VectorStore implementation (Phase 2).

Integrates Facebook AI Similarity Search (FAISS) behind the VectorStore protocol.
Uses IndexFlatIP on L2-normalized embeddings for exact inner-product (cosine) search.
Maintains metadata sidecar, supports filtering, duplicate prevention, and version verification.
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


class FaissVectorStore(VectorStore):
    """FAISS-based vector store with version-aware metadata sidecar and filtering."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension: int = 384,
        index_version: int = 1,
    ) -> None:
        import faiss

        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.index_version = index_version

        # Inner-product on normalized vectors equals cosine similarity
        self.index = faiss.IndexFlatIP(self.embedding_dimension)
        self._chunks: list[dict[str, Any]] = []
        self._seen_chunk_ids: set[str] = set()
        self._raw_vectors: list[list[float]] = []

    def add(self, chunks: list[DocumentChunk], embeddings: list[EmbeddingResult]) -> None:
        """Add chunks and embeddings to FAISS index, discarding duplicate chunk_ids."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch between chunks ({len(chunks)}) and embeddings ({len(embeddings)})"
            )

        new_vectors: list[list[float]] = []
        for chunk, emb in zip(chunks, embeddings, strict=False):
            if chunk.chunk_id in self._seen_chunk_ids:
                continue

            if emb.dimension != self.embedding_dimension:
                raise ValueError(
                    f"Vector dimension {emb.dimension} does not match store dimension {self.embedding_dimension}"
                )

            self._seen_chunk_ids.add(chunk.chunk_id)
            new_vectors.append(emb.vector)
            self._raw_vectors.append(emb.vector)
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

        if new_vectors:
            vec_arr = np.array(new_vectors, dtype=np.float32)
            # Ensure vectors are normalized
            norms = np.linalg.norm(vec_arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vec_arr = vec_arr / norms
            self.index.add(vec_arr)

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
        """Search FAISS index for nearest vectors, applying metadata post-filtering."""
        if self.index.ntotal == 0:
            return []

        q_arr = np.array([query_vector], dtype=np.float32)
        q_norm = np.linalg.norm(q_arr)
        if q_norm > 0:
            q_arr = q_arr / q_norm

        # If filters are provided, query candidate pool up to min(ntotal, max(top_k * 4, 50))
        fetch_k = min(self.index.ntotal, max(top_k * 4, 50)) if filters else min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_arr, fetch_k)

        scored_candidates: list[tuple[float, int, dict[str, Any]]] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self._chunks):
                continue
            rec = self._chunks[idx]
            if not self._matches_filters(rec, filters):
                continue
            scored_candidates.append((float(score), idx, rec))

        # Sort descending by score, ascending by chunk_id for deterministic ties
        scored_candidates.sort(key=lambda x: (-x[0], x[2]["chunk_id"]))

        results: list[RetrievalResult] = []
        for rank, (score, _, rec) in enumerate(scored_candidates[:top_k], start=1):
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
        """Delete all chunks for a document_id and rebuild the FAISS index."""
        import faiss

        keep_chunks: list[dict[str, Any]] = []
        keep_vectors: list[list[float]] = []
        deleted_count = 0

        for vec, rec in zip(self._raw_vectors, self._chunks, strict=False):
            if rec["document_id"] == document_id:
                self._seen_chunk_ids.discard(rec["chunk_id"])
                deleted_count += 1
            else:
                keep_chunks.append(rec)
                keep_vectors.append(vec)

        # Rebuild index
        self._chunks = keep_chunks
        self._raw_vectors = keep_vectors
        self.index = faiss.IndexFlatIP(self.embedding_dimension)

        if keep_vectors:
            vec_arr = np.array(keep_vectors, dtype=np.float32)
            norms = np.linalg.norm(vec_arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vec_arr = vec_arr / norms
            self.index.add(vec_arr)

        return deleted_count

    def count(self) -> int:
        return self.index.ntotal

    def save(self, path: Path | str) -> None:
        """Save FAISS binary index and metadata sidecar."""
        import faiss

        base_path = Path(path)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        index_file = base_path.with_suffix(".faiss") if base_path.is_file() else base_path / "index.faiss"
        meta_file = base_path.with_suffix(".meta.json") if base_path.is_file() else base_path / "metadata.json"

        index_file.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_file))

        metadata = {
            "index_version": self.index_version,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "count": self.index.ntotal,
            "chunks": self._chunks,
            "raw_vectors": self._raw_vectors,
        }
        meta_file.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

    def load(self, path: Path | str) -> None:
        """Load FAISS index and metadata sidecar with version verification."""
        import faiss

        base_path = Path(path)
        index_file = base_path.with_suffix(".faiss") if base_path.is_file() else base_path / "index.faiss"
        meta_file = base_path.with_suffix(".meta.json") if base_path.is_file() else base_path / "metadata.json"

        if not index_file.exists():
            raise IndexCorruptError(f"FAISS index file not found: {index_file}")
        if not meta_file.exists():
            raise IndexCorruptError(f"FAISS metadata sidecar not found: {meta_file}")

        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception as e:
            raise IndexCorruptError(f"Failed to parse metadata sidecar: {e}") from e

        persisted_model = meta.get("embedding_model", "")
        persisted_dim = meta.get("embedding_dimension", 0)

        # Validate version and dimension
        if persisted_model != self.embedding_model or persisted_dim != self.embedding_dimension:
            raise IndexVersionMismatchError(
                index_model=persisted_model,
                index_dim=persisted_dim,
                expected_model=self.embedding_model,
                expected_dim=self.embedding_dimension,
            )

        self.index_version = meta.get("index_version", 1)
        self.index = faiss.read_index(str(index_file))
        self._chunks = meta.get("chunks", [])
        self._raw_vectors = meta.get("raw_vectors", [])
        self._seen_chunk_ids = {c["chunk_id"] for c in self._chunks}
