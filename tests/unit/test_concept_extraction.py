"""Unit tests for baseline heuristic concept and relation extraction (Phase 3)."""

from __future__ import annotations

import pytest

from indiclearn.concepts.extractor import HeuristicConceptExtractor
from indiclearn.domain.concepts import RelationType
from indiclearn.domain.documents import DocumentChunk


def make_chunk(
    chunk_id: str = "chunk_001",
    text: str = "",
    section: str | None = None,
    page_number: int = 1,
    subject_id: str = "signals_and_systems",
) -> DocumentChunk:
    """Helper to construct valid DocumentChunk instances for extraction tests."""
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc_test",
        subject_id=subject_id,
        page_number=page_number,
        section=section,
        text=text,
        char_start=0,
        char_end=len(text),
    )


@pytest.fixture()
def extractor() -> HeuristicConceptExtractor:
    return HeuristicConceptExtractor(subject_id="signals_and_systems")


class TestHeuristicConceptExtractor:
    """Tests for deterministic rule-based extractor."""

    def test_extract_from_empty_chunk(self, extractor: HeuristicConceptExtractor) -> None:
        chunk = make_chunk(chunk_id="chunk_empty", text="")
        concepts, relations = extractor.extract(chunk)
        assert concepts == []
        assert relations == []

    def test_extract_from_whitespace_chunk(self, extractor: HeuristicConceptExtractor) -> None:
        chunk = make_chunk(chunk_id="chunk_ws", text="   \n\t   ")
        concepts, relations = extractor.extract(chunk)
        assert concepts == []
        assert relations == []

    def test_extract_from_heading(self, extractor: HeuristicConceptExtractor) -> None:
        text = "## Continuous-Time Convolution\n\nThe mathematical operation is fundamental."
        chunk = make_chunk(
            chunk_id="chunk_heading",
            text=text,
            section="Chapter 2: Convolution",
        )
        concepts, _ = extractor.extract(chunk)
        names = {c.canonical_name for c in concepts}
        assert "Convolution" in names or "Continuous-Time Convolution" in names
        assert any(c.confidence >= 0.85 for c in concepts)

    def test_extract_definitional_grammar(self, extractor: HeuristicConceptExtractor) -> None:
        text = (
            "An LTI system is characterized by linearity and time-invariance. "
            "Convolution is defined as the integral combining the input signal with the impulse response."
        )
        chunk = make_chunk(chunk_id="chunk_def", text=text)
        concepts, _relations = extractor.extract(chunk)
        names = {c.canonical_name for c in concepts}

        assert "LTI System" in names
        assert "Convolution" in names
        assert "Linearity" in names
        assert "Time Invariance" in names

        # Provenance verification
        for c in concepts:
            assert "chunk_def" in c.source_chunk_ids
            assert "doc_test" in c.source_document_ids

    def test_extract_technical_terms(self, extractor: HeuristicConceptExtractor) -> None:
        text = (
            "The Continuous-Time Fourier Transform represents continuous signals in frequency domain. "
            "The Region of Convergence of the Laplace Transform determines system stability."
        )
        chunk = make_chunk(chunk_id="chunk_tech", text=text)
        concepts, _ = extractor.extract(chunk)
        names = {c.canonical_name for c in concepts}
        assert "Continuous-Time Fourier Transform" in names
        assert "Laplace Transform" in names
        assert "Region of Convergence" in names

    def test_extract_relations_between_cooccurring_concepts(
        self, extractor: HeuristicConceptExtractor
    ) -> None:
        text = (
            "Linear Time-Invariant System satisfies the properties of Linearity and Time Invariance. "
            "Convolution is used in LTI systems to compute the output."
        )
        chunk = make_chunk(chunk_id="chunk_rel", text=text)
        _, relations = extractor.extract(chunk)
        assert len(relations) >= 1

        rel_types = {r.relation_type for r in relations}
        assert (
            RelationType.HAS_PROPERTY in rel_types
            or RelationType.USED_IN in rel_types
            or RelationType.RELATED_TO in rel_types
        )

        # Verify relation provenance
        for r in relations:
            assert r.source_chunk_ids == ["chunk_rel"]
            assert r.confidence > 0.0
            assert r.extraction_method != ""

    def test_extract_with_telugu_text(self, extractor: HeuristicConceptExtractor) -> None:
        text = (
            "Convolution operation అనేది సిగ్నల్స్ అండ్ సిస్టమ్స్‌లో ముఖ్యమైన కాన్సెప్ట్. "
            "కన్వల్యూషన్ ఉపయోగించి LTI System output ను లెక్కిస్తారు."
        )
        chunk = make_chunk(chunk_id="chunk_telugu", text=text)
        concepts, _ = extractor.extract(chunk)
        names = {c.canonical_name for c in concepts}
        assert "Convolution" in names or "LTI System" in names

    def test_extract_with_codemixed_telugu_english(
        self, extractor: HeuristicConceptExtractor
    ) -> None:
        text = (
            "LTI systems lo Linearity mariyu Time Invariance chala important properties. "
            "Continuous-Time Fourier Transform ni use chesi frequency response telusukuntamu."
        )
        chunk = make_chunk(chunk_id="chunk_codemix", text=text)
        concepts, _ = extractor.extract(chunk)
        names = {c.canonical_name for c in concepts}
        assert "LTI System" in names
        assert "Linearity" in names
        assert "Continuous-Time Fourier Transform" in names

    def test_duplicate_terminology_deduplication(
        self, extractor: HeuristicConceptExtractor
    ) -> None:
        text = (
            "Convolution is important. Convolution combines signals. "
            "The convolution operation is associative and commutative. "
            "Convolution represents linear filtering."
        )
        chunk = make_chunk(chunk_id="chunk_dups", text=text)
        concepts, _ = extractor.extract(chunk)
        # Should only have one canonical Convolution concept extracted from this chunk
        conv_concepts = [c for c in concepts if c.canonical_name == "Convolution"]
        assert len(conv_concepts) == 1
