"""Unit tests for CitationParser and CitationValidator (Phase 5)."""

from __future__ import annotations

from indiclearn.rag.citation import CitationParser, CitationValidator
from indiclearn.rag.models import EvidenceItem, EvidenceSet


def _make_evidence_set() -> EvidenceSet:
    return EvidenceSet(
        items=[
            EvidenceItem(
                source_id="S1",
                document_id="doc1.pdf",
                chunk_id="chunk_001",
                text="Convolution satisfies commutative and associative properties.",
                page_number=5,
                section="Convolution",
                score=0.9,
                rank=1,
            ),
            EvidenceItem(
                source_id="S2",
                document_id="doc1.pdf",
                chunk_id="chunk_002",
                text="The impulse response characterizes an LTI system completely.",
                page_number=6,
                section="Impulse Response",
                score=0.85,
                rank=2,
            ),
        ],
        query="Explain convolution and impulse response",
        subject_id="signals",
    )


def test_citation_parser_extracts_single_and_multi():
    parser = CitationParser()
    text = "Convolution is commutative [S1]. The system is determined by impulse response [S1, S2]."

    spans = parser.parse_citation_spans(text)
    assert len(spans) == 2
    assert spans[0].cited_source_ids == ["S1"]
    assert spans[1].cited_source_ids == ["S1", "S2"]

    ids = parser.extract_unique_source_ids(text)
    assert ids == ["S1", "S2"]


def test_citation_validator_valid_citations():
    evidence_set = _make_evidence_set()
    validator = CitationValidator()

    answer = "Convolution is commutative [S1]. Impulse response determines output [S2]."
    all_citations, valid_citations, invalid_ids = validator.validate_citations(answer, evidence_set)

    assert len(all_citations) == 2
    assert len(valid_citations) == 2
    assert len(invalid_ids) == 0
    assert all_citations[0].source_id == "S1"
    assert all_citations[0].chunk_id == "chunk_001"
    assert all_citations[0].document_id == "doc1.pdf"
    assert all_citations[0].page_number == 5
    assert all_citations[1].source_id == "S2"
    assert all_citations[1].chunk_id == "chunk_002"


def test_citation_validator_detects_invalid_citations():
    evidence_set = _make_evidence_set()
    validator = CitationValidator()

    # Answer cites an invented source S999
    answer = "Convolution is fundamental [S1]. This is proven in advanced theory [S999]."
    all_citations, valid_citations, invalid_ids = validator.validate_citations(answer, evidence_set)

    assert len(all_citations) == 2
    assert len(valid_citations) == 1
    assert valid_citations[0].source_id == "S1"

    assert len(invalid_ids) == 1
    assert invalid_ids[0] == "S999"

    invalid_cit = next(c for c in all_citations if not c.is_valid)
    assert invalid_cit.source_id == "S999"
    assert "was not supplied in prompt evidence" in (invalid_cit.validation_error or "")
