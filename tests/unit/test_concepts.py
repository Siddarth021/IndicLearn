"""Unit tests for Concept domain models and normalization (Phase 3)."""

from __future__ import annotations

import json

from indiclearn.concepts.normalizer import (
    ConceptNormalizer,
    generate_concept_id,
    generate_relation_id,
    slugify,
)
from indiclearn.domain.concepts import (
    Concept,
    ConceptRelation,
    ConceptType,
    PrerequisiteNature,
    RelationType,
)


class TestConceptDomainModels:
    """Tests for Concept and ConceptRelation Pydantic models."""

    def test_concept_creation_valid(self) -> None:
        concept = Concept(
            concept_id="cpt_signals_and_systems_convolution",
            canonical_name="Convolution",
            concept_type=ConceptType.METHOD,
            aliases=["convolution operation", "కన్వల్యూషన్", "convolution ante"],
            description="Integral combining input signal with impulse response.",
            subject_id="signals_and_systems",
            source_document_ids=["doc_001"],
            source_chunk_ids=["chunk_001", "chunk_004"],
            page_numbers=[12, 13],
            sections=["Chapter 2", "Section 2.1"],
            confidence=0.95,
        )
        assert concept.concept_id == "cpt_signals_and_systems_convolution"
        assert concept.canonical_name == "Convolution"
        assert concept.concept_type == ConceptType.METHOD
        assert len(concept.aliases) == 3
        assert "కన్వల్యూషన్" in concept.aliases
        assert concept.confidence == 0.95

    def test_concept_serialization_roundtrip(self) -> None:
        concept = Concept(
            concept_id="cpt_signals_and_systems_lti_system",
            canonical_name="LTI System",
            concept_type=ConceptType.CONCEPT,
            aliases=["Linear Time-Invariant System", "LTI"],
            subject_id="signals_and_systems",
            source_chunk_ids=["chunk_001"],
        )
        data = concept.model_dump(mode="json")
        json_str = json.dumps(data)
        loaded = Concept.model_validate(json.loads(json_str))
        assert loaded.concept_id == concept.concept_id
        assert loaded.canonical_name == concept.canonical_name
        assert loaded.concept_type == ConceptType.CONCEPT
        assert loaded.aliases == concept.aliases

    def test_concept_relation_creation_and_provenance(self) -> None:
        rel = ConceptRelation(
            relation_id="rel_signals_and_systems_lti_convolution",
            source_concept_id="cpt_signals_and_systems_lti_system",
            target_concept_id="cpt_signals_and_systems_convolution",
            relation_type=RelationType.PREREQUISITE_OF,
            subject_id="signals_and_systems",
            confidence=0.90,
            extraction_method="heuristic_heading_flow",
            prerequisite_nature=PrerequisiteNature.EXPLICIT_PREREQUISITE,
            source_chunk_ids=["chunk_001"],
            page_numbers=[5],
        )
        assert rel.relation_type == RelationType.PREREQUISITE_OF
        assert rel.confidence == 0.90
        assert rel.prerequisite_nature == PrerequisiteNature.EXPLICIT_PREREQUISITE
        assert rel.source_chunk_ids == ["chunk_001"]

    def test_student_fields_not_in_academic_concept(self) -> None:
        concept = Concept(
            concept_id="cpt_signals_and_systems_linearity",
            canonical_name="Linearity",
            subject_id="signals_and_systems",
        )
        assert not hasattr(concept, "mastery")
        assert not hasattr(concept, "knowledge_score")
        assert not hasattr(concept, "student_id")
        assert not hasattr(concept, "quiz_performance")


class TestConceptNormalizerAndIdentifiers:
    """Tests for slugify, ID generation, and ConceptNormalizer."""

    def test_slugify_basic(self) -> None:
        assert slugify("LTI System") == "lti_system"
        assert slugify("Continuous-Time Fourier Transform") == "continuous_time_fourier_transform"
        assert slugify("   Region   of   Convergence   ") == "region_of_convergence"
        assert slugify("2*B Nyquist Rate") == "2b_nyquist_rate"

    def test_slugify_multilingual_unicode(self) -> None:
        slug = slugify("కన్వల్యూషన్ ఆపరేషన్")
        assert len(slug) > 0
        assert " " not in slug

    def test_deterministic_concept_id_generation(self) -> None:
        id1 = generate_concept_id("signals_and_systems", "Convolution")
        id2 = generate_concept_id("signals_and_systems", "convolution")
        id3 = generate_concept_id("signals_and_systems", "  Convolution  ")
        assert id1 == id2 == id3 == "cpt_signals_and_systems_convolution"

    def test_subject_isolation_in_concept_id(self) -> None:
        id_ss = generate_concept_id("signals_and_systems", "Signal")
        id_dl = generate_concept_id("digital_logic", "Signal")
        assert id_ss != id_dl
        assert id_ss == "cpt_signals_and_systems_signal"
        assert id_dl == "cpt_digital_logic_signal"

    def test_deterministic_relation_id_generation(self) -> None:
        rid1 = generate_relation_id(
            "signals_and_systems",
            "cpt_ss_lti",
            RelationType.PREREQUISITE_OF,
            "cpt_ss_conv",
        )
        rid2 = generate_relation_id(
            "signals_and_systems",
            "cpt_ss_lti",
            RelationType.PREREQUISITE_OF,
            "cpt_ss_conv",
        )
        assert rid1 == rid2
        assert "prerequisite_of" in rid1

    def test_normalizer_acronym_mapping(self) -> None:
        normalizer = ConceptNormalizer()
        assert normalizer.canonicalize("LTI") == "LTI System"
        assert normalizer.canonicalize("lti systems") == "LTI System"
        assert normalizer.canonicalize("linear time-invariant system") == "LTI System"
        assert normalizer.canonicalize("ROC") == "Region of Convergence"
        assert normalizer.canonicalize("CTFT") == "Continuous-Time Fourier Transform"

    def test_normalizer_preserves_distinct_concepts(self) -> None:
        normalizer = ConceptNormalizer()
        # Fourier Transform and Fourier Series must remain separate
        ft = normalizer.canonicalize("Fourier Transform")
        fs = normalizer.canonicalize("Fourier Series")
        assert ft != fs
        assert "Transform" in ft
        assert "Series" in fs

    def test_normalizer_strips_articles_and_prefixes(self) -> None:
        normalizer = ConceptNormalizer()
        assert normalizer.canonicalize("the continuous-time signal") == "Continuous-Time Signal"
        assert normalizer.canonicalize("a power signal") == "Power Signal"
        assert normalizer.canonicalize("properties of linearity") == "Linearity"

    def test_normalizer_rejects_noise_fragments(self) -> None:
        normalizer = ConceptNormalizer()
        assert not normalizer.canonicalize("a continuous or")
        assert not normalizer.canonicalize("in")
        assert not normalizer.canonicalize("which is defined as")
        assert not normalizer.canonicalize("and")

    def test_normalizer_alias_curation(self) -> None:
        normalizer = ConceptNormalizer()
        aliases = normalizer.get_curated_aliases("LTI System")
        assert "LTI" in aliases
        assert "linear time-invariant system" in aliases
        assert "linear time invariant system" in aliases
