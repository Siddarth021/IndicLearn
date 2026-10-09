"""Hybrid retriever combining lexical and dense search (Phase 2).

Supports two fusion strategies:
1. Reciprocal Rank Fusion (RRF): rank-based fusion invariant to score distributions.
2. Linear Combination with Min-Max Score Normalization: normalizes BM25 and dense
   cosine similarity scores to [0.0, 1.0] before weighted summation.
"""

from __future__ import annotations

from typing import Any, Literal

from indiclearn.retrieval.base import RetrievalResult, Retriever


def min_max_normalize(scores: list[float]) -> list[float]:
    """Normalize a list of scores to the range [0.0, 1.0]."""
    if not scores:
        return []
    min_val = min(scores)
    max_val = max(scores)
    diff = max_val - min_val
    if diff == 0.0:
        return [1.0] * len(scores)
    return [(s - min_val) / diff for s in scores]


class HybridRetriever(Retriever):
    """Combines lexical (e.g., BM25) and dense semantic retrieval results."""

    def __init__(
        self,
        lexical_retriever: Retriever,
        dense_retriever: Retriever,
        strategy: Literal["rrf", "linear"] = "rrf",
        weight_lexical: float = 0.5,
        weight_dense: float = 0.5,
        rrf_k: int = 60,
    ) -> None:
        self.lexical_retriever = lexical_retriever
        self.dense_retriever = dense_retriever
        self.strategy = strategy
        self.weight_lexical = weight_lexical
        self.weight_dense = weight_dense
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve candidates from both retrievers and merge them using the selected fusion method."""
        if not query.strip():
            return []

        # Retrieve a broader candidate pool from both branches
        fetch_k = max(top_k * 3, 20)
        lexical_results = self.lexical_retriever.retrieve(query, top_k=fetch_k, filters=filters)
        dense_results = self.dense_retriever.retrieve(query, top_k=fetch_k, filters=filters)

        if not lexical_results and not dense_results:
            return []
        if not lexical_results:
            return dense_results[:top_k]
        if not dense_results:
            return lexical_results[:top_k]

        if self.strategy == "rrf":
            return self._fuse_rrf(lexical_results, dense_results, top_k)
        return self._fuse_linear(lexical_results, dense_results, top_k)

    def _fuse_rrf(
        self,
        lexical: list[RetrievalResult],
        dense: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Fuse results using Reciprocal Rank Fusion."""
        all_chunks: dict[str, RetrievalResult] = {}
        rrf_scores: dict[str, float] = {}

        for item in lexical:
            all_chunks[item.chunk_id] = item
            score_contrib = self.weight_lexical * (1.0 / (self.rrf_k + item.rank))
            rrf_scores[item.chunk_id] = rrf_scores.get(item.chunk_id, 0.0) + score_contrib

        for item in dense:
            all_chunks[item.chunk_id] = item
            score_contrib = self.weight_dense * (1.0 / (self.rrf_k + item.rank))
            rrf_scores[item.chunk_id] = rrf_scores.get(item.chunk_id, 0.0) + score_contrib

        # Rank sorted by RRF score descending, chunk_id ascending
        sorted_chunk_ids = sorted(
            rrf_scores.keys(), key=lambda cid: (-rrf_scores[cid], cid)
        )

        fused: list[RetrievalResult] = []
        for rank, cid in enumerate(sorted_chunk_ids[:top_k], start=1):
            src = all_chunks[cid]
            fused.append(
                RetrievalResult(
                    chunk_id=cid,
                    document_id=src.document_id,
                    score=float(rrf_scores[cid]),
                    rank=rank,
                    text=src.text,
                    metadata=dict(src.metadata),
                )
            )
        return fused

    def _fuse_linear(
        self,
        lexical: list[RetrievalResult],
        dense: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Fuse results using Min-Max score normalization followed by weighted linear combination."""
        lex_norm = min_max_normalize([r.score for r in lexical])
        dense_norm = min_max_normalize([r.score for r in dense])

        all_chunks: dict[str, RetrievalResult] = {}
        combined_scores: dict[str, float] = {}

        for norm_score, item in zip(lex_norm, lexical, strict=False):
            all_chunks[item.chunk_id] = item
            combined_scores[item.chunk_id] = self.weight_lexical * norm_score

        for norm_score, item in zip(dense_norm, dense, strict=False):
            all_chunks[item.chunk_id] = item
            prior = combined_scores.get(item.chunk_id, 0.0)
            combined_scores[item.chunk_id] = prior + (self.weight_dense * norm_score)

        sorted_chunk_ids = sorted(
            combined_scores.keys(), key=lambda cid: (-combined_scores[cid], cid)
        )

        fused: list[RetrievalResult] = []
        for rank, cid in enumerate(sorted_chunk_ids[:top_k], start=1):
            src = all_chunks[cid]
            fused.append(
                RetrievalResult(
                    chunk_id=cid,
                    document_id=src.document_id,
                    score=float(combined_scores[cid]),
                    rank=rank,
                    text=src.text,
                    metadata=dict(src.metadata),
                )
            )
        return fused
