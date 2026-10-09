"""Unit tests for IR retrieval metrics (Phase 2)."""

import pytest

from indiclearn.evaluation.retrieval_metrics import (
    QueryEvaluation,
    aggregate_metrics,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_perfect():
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    relevant = {"doc1", "doc2"}
    assert recall_at_k(retrieved, relevant, k=2) == 1.0
    assert recall_at_k(retrieved, relevant, k=1) == 0.5


def test_recall_at_k_partial_and_zero():
    retrieved = ["docA", "docB", "docC"]
    relevant = {"doc1", "doc2"}
    assert recall_at_k(retrieved, relevant, k=3) == 0.0
    assert recall_at_k(retrieved, set(), k=3) == 0.0
    assert recall_at_k(retrieved, relevant, k=0) == 0.0


def test_precision_at_k():
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    relevant = {"doc1", "doc3"}
    assert precision_at_k(retrieved, relevant, k=2) == 0.5
    assert precision_at_k(retrieved, relevant, k=4) == 0.5
    assert precision_at_k(retrieved, relevant, k=0) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["docA", "doc1", "docB"], {"doc1"}) == 0.5
    assert reciprocal_rank(["doc1", "docA"], {"doc1"}) == 1.0
    assert reciprocal_rank(["docA", "docB", "docC"], {"doc1"}) == 0.0
    assert reciprocal_rank([], {"doc1"}) == 0.0
    assert reciprocal_rank(["doc1"], set()) == 0.0


def test_dcg_and_ndcg_at_k():
    # Ideal ranking: doc1 (rel=2), doc2 (rel=1)
    retrieved = ["doc1", "doc2", "doc3"]
    relevance = {"doc1": 2.0, "doc2": 1.0}
    # Retrieved in ideal order: nDCG should be 1.0
    assert pytest.approx(ndcg_at_k(retrieved, relevance, k=3), 0.001) == 1.0

    # Reversed order: doc2 (rel=1), doc1 (rel=2)
    reversed_retrieved = ["doc2", "doc1", "doc3"]
    ndcg_rev = ndcg_at_k(reversed_retrieved, relevance, k=2)
    assert 0.0 < ndcg_rev < 1.0

    # Binary relevance via set
    binary_retrieved = ["docA", "docB", "docC"]
    assert ndcg_at_k(binary_retrieved, {"docA"}, k=1) == 1.0
    assert ndcg_at_k(binary_retrieved, set(), k=3) == 0.0


def test_aggregate_metrics():
    eval1 = QueryEvaluation("Q1", ["c1", "c2"], {"c1"})
    eval2 = QueryEvaluation("Q2", ["c3", "c4"], {"c4"})

    summary = aggregate_metrics([eval1, eval2])
    assert summary["count"] == 2.0
    assert summary["Recall@1"] == 0.5  # eval1 has it at rank 1, eval2 at rank 2
    assert summary["Recall@3"] == 1.0
    assert summary["MRR"] == 0.75  # (1.0 + 0.5) / 2

    # Empty list
    empty_summary = aggregate_metrics([])
    assert empty_summary["count"] == 0.0
    assert empty_summary["MRR"] == 0.0
