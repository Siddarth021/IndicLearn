"""End-to-end retrieval pipeline (Phase 2).

Orchestrates query preprocessing, retriever selection, candidate retrieval,
optional reranking, and logging.
"""

from __future__ import annotations

import unicodedata
from typing import Any

from indiclearn.retrieval.base import Reranker, RetrievalResult, Retriever
from indiclearn.utils.logging import get_logger, log_event

logger = get_logger("retrieval.pipeline")


def preprocess_query(query: str, form: str = "NFC") -> str:
    """Preprocess query string: normalize Unicode NFC and strip surrounding whitespace."""
    if not query:
        return ""
    normalized = unicodedata.normalize(form, query)
    return normalized.strip()


class RetrievalPipeline:
    """Configurable retrieval execution pipeline."""

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker | None = None,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker

    def run(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Execute retrieval workflow: preprocess -> retrieve -> optional rerank."""
        clean_q = preprocess_query(query)
        if not clean_q:
            return []

        log_event(
            logger,
            "retrieval_started",
            query_len=len(clean_q),
            top_k=top_k,
            has_filters=bool(filters),
        )

        candidates = self.retriever.retrieve(clean_q, top_k=top_k, filters=filters)

        if self.reranker is not None and candidates:
            final_results = self.reranker.rerank(clean_q, candidates, top_k=top_k)
        else:
            final_results = candidates

        log_event(
            logger,
            "retrieval_completed",
            retrieved_count=len(final_results),
            top_score=final_results[0].score if final_results else None,
        )

        return final_results
