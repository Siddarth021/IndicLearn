"""TF-IDF lexical retrieval baseline (Phase 2).

Provides an alternative lexical baseline using scikit-learn's TfidfVectorizer.
Satisfies the Retriever protocol.
"""

from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from indiclearn.domain.documents import DocumentChunk
from indiclearn.retrieval.base import RetrievalResult, Retriever


class TFIDFRetriever(Retriever):
    """TF-IDF retriever for DocumentChunk collections."""

    def __init__(self, sublinear_tf: bool = True) -> None:
        self.sublinear_tf = sublinear_tf
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix: Any = None
        self.chunks: list[DocumentChunk] = []

    def index(self, chunks: list[DocumentChunk]) -> None:
        """Index a list of DocumentChunks into a TF-IDF matrix."""
        self.chunks = list(chunks)
        if not chunks:
            self.vectorizer = None
            self.tfidf_matrix = None
            return

        corpus = [chunk.text for chunk in chunks]
        self.vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b",
            sublinear_tf=self.sublinear_tf,
            lowercase=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def _matches_filters(self, chunk: DocumentChunk, filters: dict[str, Any] | None) -> bool:
        """Check if chunk satisfies metadata filters."""
        if not filters:
            return True
        for key, expected in filters.items():
            if key == "document_id" and chunk.document_id != expected:
                return False
            if key == "subject_id" and chunk.subject_id != expected:
                return False
            if key == "language" and chunk.language != expected:
                return False
            if key == "concept_id":
                if isinstance(expected, list):
                    if not any(cid in chunk.concept_ids for cid in expected):
                        return False
                elif expected not in chunk.concept_ids:
                    return False
        return True

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve top_k chunks matching query via TF-IDF cosine similarity."""
        if not self.chunks or self.vectorizer is None or not query.strip():
            return []

        try:
            q_vec = self.vectorizer.transform([query])
        except (ValueError, AttributeError):
            return []

        if q_vec.nnz == 0:
            # Query has zero recognized terms in vocabulary
            return []

        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        scores: list[tuple[float, int, DocumentChunk]] = []
        for idx, (sim, chunk) in enumerate(zip(sims, self.chunks, strict=False)):
            if not self._matches_filters(chunk, filters):
                continue
            if sim > 0:
                scores.append((float(sim), idx, chunk))

        # Sort descending by score, ascending by chunk_id for deterministic ties
        scores.sort(key=lambda item: (-item[0], item[2].chunk_id))

        results: list[RetrievalResult] = []
        for rank, (score, _, chunk) in enumerate(scores[:top_k], start=1):
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    score=float(score),
                    rank=rank,
                    text=chunk.text,
                    metadata={
                        "subject_id": chunk.subject_id,
                        "page_number": chunk.page_number,
                        "section": chunk.section,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                        "language": chunk.language,
                        "concept_ids": chunk.concept_ids,
                    },
                )
            )
        return results
