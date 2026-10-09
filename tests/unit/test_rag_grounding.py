"""Unit tests for GroundingVerifier in RAG pipeline (Phase 5)."""

from __future__ import annotations

from indiclearn.rag.citation import CitationValidator
from indiclearn.rag.grounding import GroundingVerifier
from indiclearn.rag.models import EvidenceItem, EvidenceSet


def _make_evidence_set() -> EvidenceSet:
    return EvidenceSet(
        items=[
            EvidenceItem(
                source_id="S1",
                document_id="doc1.pdf",
                chunk_id="chunk_001",
                text="Convolution is commutative and associative.",
                page_number=5,
                score=0.9,
                rank=1,
            ),
            EvidenceItem(
                source_id="S2",
                document_id="doc1.pdf",
                chunk_id="chunk_002",
                text="Impulse response completely specifies continuous LTI systems.",
                page_number=6,
                score=0.85,
                rank=2,
            ),
        ],
        query="Explain convolution",
        subject_id="signals",
    )


def test_grounding_verifier_fully_grounded():
    evidence_set = _make_evidence_set()
    verifier = GroundingVerifier(validator=CitationValidator())

    answer = "Convolution is commutative and associative [S1]. The impulse response completely specifies LTI systems [S2]."
    report = verifier.verify_grounding(answer, evidence_set)

    assert report.status == "GROUNDED"
    assert report.check.is_grounded is True
    assert report.check.citation_validity_rate == 1.0
    assert report.check.citation_coverage == 1.0
    assert report.check.unsupported_citation_rate == 0.0
    assert len(report.invalid_citations) == 0


def test_grounding_verifier_unsupported_citation():
    evidence_set = _make_evidence_set()
    verifier = GroundingVerifier(validator=CitationValidator())

    # Cites S999 which is not in evidence_set
    answer = "Convolution is commutative [S1]. Another property is stated elsewhere [S999]."
    report = verifier.verify_grounding(answer, evidence_set)

    assert report.status == "UNSUPPORTED"
    assert report.check.is_grounded is False
    assert report.check.citation_validity_rate == 0.5
    assert report.check.unsupported_citation_rate == 0.5
    assert "S999" in report.invalid_citations


def test_grounding_verifier_insufficient_evidence():
    evidence_set = EvidenceSet(
        items=[],
        query="Unknown query",
        subject_id="signals",
        insufficient_evidence=True,
    )
    verifier = GroundingVerifier(validator=CitationValidator())

    fallback = "I could not find sufficient information in the uploaded academic materials to answer this question reliably."
    report = verifier.verify_grounding(fallback, evidence_set, is_insufficient_evidence_response=True)

    assert report.status == "INSUFFICIENT_EVIDENCE"
    assert report.check.is_grounded is True
    assert report.check.evidence_sufficiency is False
    assert len(report.valid_citations) == 0


def test_grounding_verifier_partially_grounded_missing_citations():
    evidence_set = _make_evidence_set()
    verifier = GroundingVerifier(validator=CitationValidator(), min_coverage_threshold=0.8)

    # One sentence cited, another uncited
    answer = "Convolution is commutative [S1]. Furthermore this mathematical theorem holds across all continuous-time and discrete-time domains."
    report = verifier.verify_grounding(answer, evidence_set)

    assert report.status == "PARTIALLY_GROUNDED"
    assert report.check.citation_validity_rate == 1.0
    assert report.check.citation_coverage == 0.5
