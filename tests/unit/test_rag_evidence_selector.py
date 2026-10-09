"""Unit tests for EvidenceSelector in RAG pipeline (Phase 5)."""

from __future__ import annotations

from indiclearn.rag.evidence_selector import EvidenceSelector
from indiclearn.retrieval.base import RetrievalResult


def test_evidence_selector_deduplication():
    selector = EvidenceSelector(min_score_threshold=0.01)
    results = [
        RetrievalResult(chunk_id="chunk_1", document_id="doc_A", score=0.8, rank=1, text="Text 1"),
        RetrievalResult(chunk_id="chunk_2", document_id="doc_A", score=0.6, rank=2, text="Text 2"),
        RetrievalResult(chunk_id="chunk_1", document_id="doc_A", score=0.9, rank=3, text="Text 1 updated"),
    ]
    evidence_set = selector.select_evidence(query="test query", subject_id="signals", retrieval_results=results)

    assert len(evidence_set.items) == 2
    assert evidence_set.items[0].chunk_id == "chunk_1"
    assert evidence_set.items[0].score == 0.9
    assert evidence_set.items[0].source_id == "S1"
    assert evidence_set.items[1].chunk_id == "chunk_2"
    assert evidence_set.items[1].source_id == "S2"
    assert not evidence_set.insufficient_evidence


def test_evidence_selector_threshold():
    selector = EvidenceSelector(min_score_threshold=0.5)
    results = [
        RetrievalResult(chunk_id="c1", document_id="d1", score=0.7, rank=1, text="Relevant text"),
        RetrievalResult(chunk_id="c2", document_id="d1", score=0.3, rank=2, text="Weak text"),
    ]
    evidence_set = selector.select_evidence(query="test", subject_id="signals", retrieval_results=results)

    assert len(evidence_set.items) == 1
    assert evidence_set.items[0].chunk_id == "c1"
    assert not evidence_set.insufficient_evidence


def test_evidence_selector_insufficient_when_below_threshold():
    selector = EvidenceSelector(min_score_threshold=0.8)
    results = [
        RetrievalResult(chunk_id="c1", document_id="d1", score=0.5, rank=1, text="Weak match"),
    ]
    evidence_set = selector.select_evidence(query="test", subject_id="signals", retrieval_results=results)

    assert len(evidence_set.items) == 0
    assert evidence_set.insufficient_evidence is True
    assert "below the required threshold" in (evidence_set.insufficiency_reason or "")


def test_evidence_selector_empty_results():
    selector = EvidenceSelector()
    evidence_set = selector.select_evidence(query="empty", subject_id="signals", retrieval_results=[])

    assert len(evidence_set.items) == 0
    assert evidence_set.insufficient_evidence is True
    assert "zero candidate chunks" in (evidence_set.insufficiency_reason or "")


def test_evidence_selector_character_budget():
    selector = EvidenceSelector(min_score_threshold=0.01, max_total_chars=50)
    results = [
        RetrievalResult(chunk_id="c1", document_id="d1", score=0.9, rank=1, text="A" * 30),
        RetrievalResult(chunk_id="c2", document_id="d1", score=0.8, rank=2, text="B" * 30),
    ]
    evidence_set = selector.select_evidence(query="budget", subject_id="signals", retrieval_results=results)

    # First item takes 30 chars; adding second would exceed 50, so only 1 selected
    assert len(evidence_set.items) == 1
    assert evidence_set.items[0].chunk_id == "c1"


def test_evidence_selector_rrf_scale_adaptation():
    selector = EvidenceSelector(min_score_threshold=0.15)
    # RRF scores are typically around 0.016
    results = [
        RetrievalResult(chunk_id="c1", document_id="d1", score=0.016, rank=1, text="RRF result"),
    ]
    evidence_set = selector.select_evidence(query="rrf", subject_id="signals", retrieval_results=results)

    assert len(evidence_set.items) == 1
    assert evidence_set.items[0].chunk_id == "c1"
    assert not evidence_set.insufficient_evidence
