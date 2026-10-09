"""Evidence selection and deduplication stage for RAG (Phase 5).

Filters, ranks, and assigns stable sequential source IDs (S1, S2, ...) while
preserving complete provenance and detecting evidence insufficiency.
"""

from __future__ import annotations

import logging
from typing import Any

from indiclearn.rag.models import EvidenceItem, EvidenceSet
from indiclearn.retrieval.base import RetrievalResult

logger = logging.getLogger(__name__)


class EvidenceSelector:
    """Selects, deduplicates, and validates academic evidence from retrieval results."""

    def __init__(
        self,
        min_score_threshold: float = 0.005,
        max_context_chunks: int = 5,
        max_total_chars: int = 4000,
    ) -> None:
        self.min_score_threshold = min_score_threshold
        self.max_context_chunks = max_context_chunks
        self.max_total_chars = max_total_chars

    def select_evidence(
        self,
        query: str,
        subject_id: str,
        retrieval_results: list[RetrievalResult],
        top_k: int | None = None,
        min_threshold: float | None = None,
    ) -> EvidenceSet:
        """Select best candidate chunks, eliminate duplicates, and assign source IDs.

        Args:
            query: Student question text.
            subject_id: Academic subject namespace.
            retrieval_results: Raw scored chunks from retrieval layer.
            top_k: Optional override for max chunks to select.
            min_threshold: Optional override for minimum similarity score.

        Returns:
            EvidenceSet populated with structured EvidenceItems.
        """
        k = top_k or self.max_context_chunks
        threshold = min_threshold if min_threshold is not None else self.min_score_threshold

        if not retrieval_results:
            return EvidenceSet(
                items=[],
                query=query,
                subject_id=subject_id,
                total_candidates=0,
                insufficient_evidence=True,
                insufficiency_reason="Retrieval returned zero candidate chunks.",
            )

        # 1. Deduplicate by chunk_id, preserving the entry with the highest score
        deduped: dict[str, RetrievalResult] = {}
        for res in retrieval_results:
            cid = res.chunk_id
            if cid not in deduped or res.score > deduped[cid].score:
                deduped[cid] = res

        # Sort descending by score
        candidates = sorted(deduped.values(), key=lambda r: r.score, reverse=True)

        # Adapt threshold if scores are from rank fusion (RRF) where max possible is ~0.02
        max_cand_score = max((c.score for c in candidates), default=0.0)
        effective_threshold = threshold
        if 0.0 < max_cand_score < 0.05:
            # Map standard [0, 1] threshold (e.g. 0.15) to RRF scale: 0.15 * 0.05 = 0.0075
            effective_threshold = threshold * 0.05

        # 2. Filter by threshold
        valid_candidates = [c for c in candidates if c.score >= effective_threshold]

        if not valid_candidates:
            return EvidenceSet(
                items=[],
                query=query,
                subject_id=subject_id,
                total_candidates=len(candidates),
                insufficient_evidence=True,
                insufficiency_reason=(
                    f"Top candidate score ({candidates[0].score:.4f}) is below the "
                    f"required threshold ({effective_threshold:.4f})."
                ),
            )

        # 3. Select top-k within character budget
        selected_items: list[EvidenceItem] = []
        char_count = 0

        for rank_idx, cand in enumerate(valid_candidates[:k], start=1):
            text = cand.text or ""
            if char_count + len(text) > self.max_total_chars and selected_items:
                # Exceeded total character budget; stop adding chunks
                break

            source_id = f"S{rank_idx}"
            meta: dict[str, Any] = cand.metadata or {}

            item = EvidenceItem(
                source_id=source_id,
                document_id=cand.document_id,
                chunk_id=cand.chunk_id,
                text=text,
                page_number=meta.get("page_number"),
                section=meta.get("section"),
                score=round(cand.score, 4),
                retrieval_method=meta.get("retrieval_method", "hybrid"),
                concept_ids=meta.get("concept_ids", []),
                rank=rank_idx,
                metadata=meta,
            )
            selected_items.append(item)
            char_count += len(text)

        return EvidenceSet(
            items=selected_items,
            query=query,
            subject_id=subject_id,
            total_candidates=len(candidates),
            insufficient_evidence=False,
            insufficiency_reason=None,
        )
