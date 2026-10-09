"""Unit tests for ContextBuilder in RAG pipeline (Phase 5)."""

from __future__ import annotations

from indiclearn.domain.concepts import Concept
from indiclearn.rag.context_builder import ContextBuilder
from indiclearn.rag.models import EvidenceItem, EvidenceSet


def test_context_builder_formats_evidence():
    builder = ContextBuilder()
    item = EvidenceItem(
        source_id="S1",
        document_id="doc_signals.pdf",
        chunk_id="chunk_042",
        text="Convolution is an integral operation.",
        page_number=12,
        section="Convolution Properties",
        score=0.95,
        retrieval_method="hybrid",
        concept_ids=["convolution"],
        rank=1,
    )
    evidence_set = EvidenceSet(
        items=[item],
        query="Explain convolution",
        subject_id="signals_and_systems",
        total_candidates=1,
    )

    context_str = builder.build_context(evidence_set)

    assert "=== RETRIEVED ACADEMIC EVIDENCE ===" in context_str
    assert "[SOURCE 1] (ID: S1)" in context_str
    assert "Document: doc_signals.pdf | Page 12 | Section: Convolution Properties | Chunk: chunk_042" in context_str
    assert "Related Concepts: convolution" in context_str
    assert "Convolution is an integral operation." in context_str


def test_context_builder_includes_graph_context():
    builder = ContextBuilder()
    evidence_set = EvidenceSet(
        items=[
            EvidenceItem(
                source_id="S1",
                document_id="doc.pdf",
                chunk_id="c1",
                text="System text",
                score=0.9,
                rank=1,
            )
        ],
        query="What is LTI?",
        subject_id="signals_and_systems",
    )
    graph_concepts = [
        Concept(
            concept_id="cpt_lti",
            canonical_name="LTI System",
            subject_id="signals_and_systems",
            concept_type="concept",
            aliases=["linear time-invariant system"],
        )
    ]
    prereq_map = {"cpt_lti": ["Linearity", "Time-Invariance"]}

    context_str = builder.build_context(
        evidence_set,
        graph_concepts=graph_concepts,
        prerequisite_map=prereq_map,
    )

    assert "=== ACADEMIC CURRICULUM CONTEXT ===" in context_str
    assert "Concept: LTI System (concept) [Aliases: linear time-invariant system]" in context_str
    assert "Prerequisites: Linearity, Time-Invariance" in context_str
    assert "=== RETRIEVED ACADEMIC EVIDENCE ===" in context_str


def test_context_builder_empty_evidence():
    builder = ContextBuilder()
    evidence_set = EvidenceSet(items=[], query="empty", subject_id="signals")
    context_str = builder.build_context(evidence_set)
    assert context_str == "No academic sources retrieved."
