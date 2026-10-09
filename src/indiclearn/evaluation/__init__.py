"""Evaluation package for IndicLearn (Phase 2)."""

from indiclearn.evaluation.retrieval_metrics import (
    QueryEvaluation,
    aggregate_metrics,
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "QueryEvaluation",
    "aggregate_metrics",
    "dcg_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
