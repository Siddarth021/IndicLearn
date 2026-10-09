"""Information retrieval evaluation metrics (Phase 2).

Implements reusable metric calculations:
- Recall@k
- Precision@k
- MRR (Mean Reciprocal Rank)
- nDCG@k (Normalized Discounted Cumulative Gain)
Includes dataset-level evaluation aggregator supporting multi-dimensional slicing.
"""

from __future__ import annotations

import math
from collections.abc import Collection


def recall_at_k(retrieved_ids: list[str], relevant_ids: Collection[str], k: int) -> float:
    """Calculate Recall@k for a single query.

    Recall@k = |retrieved[:k] ∩ relevant| / |relevant|
    """
    if not relevant_ids or k <= 0:
        return 0.0
    cutoff = retrieved_ids[:k]
    hits = len(set(cutoff) & set(relevant_ids))
    return hits / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: Collection[str], k: int) -> float:
    """Calculate Precision@k for a single query.

    Precision@k = |retrieved[:k] ∩ relevant| / k
    """
    if k <= 0:
        return 0.0
    cutoff = retrieved_ids[:k]
    hits = len(set(cutoff) & set(relevant_ids))
    return hits / k


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: Collection[str]) -> float:
    """Calculate Reciprocal Rank (RR) for a single query.

    RR = 1 / rank of first relevant item, or 0.0 if not found.
    """
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def dcg_at_k(
    retrieved_ids: list[str],
    relevance_scores: dict[str, float],
    k: int,
) -> float:
    """Calculate Discounted Cumulative Gain at k (DCG@k).

    DCG@k = sum_{i=1}^k (2^{rel_i} - 1) / log2(i + 1)
    """
    score = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        rel = relevance_scores.get(doc_id, 0.0)
        if rel > 0:
            score += (2.0**rel - 1.0) / math.log2(i + 1)
    return score


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant: Collection[str] | dict[str, float],
    k: int,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k (nDCG@k).

    Supports binary relevance (Collection[str]) or graded relevance (dict[str, float]).
    """
    if not relevant or k <= 0:
        return 0.0

    if not isinstance(relevant, dict):
        rel_map: dict[str, float] = {doc_id: 1.0 for doc_id in relevant}
    else:
        rel_map = relevant

    actual_dcg = dcg_at_k(retrieved_ids, rel_map, k)

    # Ideal ranking: sort all relevant items by score descending
    sorted_ideal = sorted(rel_map.keys(), key=lambda d: rel_map[d], reverse=True)
    ideal_dcg = dcg_at_k(sorted_ideal, rel_map, k)

    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


class QueryEvaluation:
    """Evaluation result for an individual query."""

    def __init__(
        self,
        query_id: str,
        retrieved_ids: list[str],
        relevant_ids: Collection[str],
        relevance_grades: dict[str, float] | None = None,
        language: str = "en",
        query_type: str = "general",
    ) -> None:
        self.query_id = query_id
        self.retrieved_ids = retrieved_ids
        self.relevant_ids = set(relevant_ids)
        self.relevance_grades = relevance_grades or {qid: 1.0 for qid in relevant_ids}
        self.language = language
        self.query_type = query_type

        self.r1 = recall_at_k(retrieved_ids, self.relevant_ids, 1)
        self.r3 = recall_at_k(retrieved_ids, self.relevant_ids, 3)
        self.r5 = recall_at_k(retrieved_ids, self.relevant_ids, 5)
        self.r10 = recall_at_k(retrieved_ids, self.relevant_ids, 10)
        self.mrr = reciprocal_rank(retrieved_ids, self.relevant_ids)
        self.ndcg10 = ndcg_at_k(retrieved_ids, self.relevance_grades, 10)


def aggregate_metrics(evaluations: list[QueryEvaluation]) -> dict[str, float]:
    """Compute mean metrics across a list of QueryEvaluations."""
    if not evaluations:
        return {
            "count": 0.0,
            "Recall@1": 0.0,
            "Recall@3": 0.0,
            "Recall@5": 0.0,
            "Recall@10": 0.0,
            "MRR": 0.0,
            "nDCG@10": 0.0,
        }

    n = len(evaluations)
    return {
        "count": float(n),
        "Recall@1": sum(e.r1 for e in evaluations) / n,
        "Recall@3": sum(e.r3 for e in evaluations) / n,
        "Recall@5": sum(e.r5 for e in evaluations) / n,
        "Recall@10": sum(e.r10 for e in evaluations) / n,
        "MRR": sum(e.mrr for e in evaluations) / n,
        "nDCG@10": sum(e.ndcg10 for e in evaluations) / n,
    }
