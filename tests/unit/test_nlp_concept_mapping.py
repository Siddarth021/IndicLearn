"""Unit tests for Cross-Lingual Concept Mapping (Baselines A-D) (Phase 4)."""

import pytest

from indiclearn.domain.concepts import Concept
from indiclearn.embeddings.mock import MockEmbeddingProvider
from indiclearn.nlp.concept_mapping.mapper import (
    EmbeddingConceptMapper,
    ExactConceptMapper,
    HybridConceptMapper,
    LexicalConceptMapper,
)


class TestConceptMapping:
    """Test suite for Concept Mappers."""

    @pytest.fixture
    def sample_concepts(self) -> list[Concept]:
        return [
            Concept(
                concept_id="cpt_ss_convolution",
                canonical_name="Convolution",
                subject_id="signals_and_systems",
                aliases=["convolution operation", "కన్వల్యూషన్"],
                description="Mathematical convolution operation for LTI systems.",
            ),
            Concept(
                concept_id="cpt_ss_lti_system",
                canonical_name="LTI System",
                subject_id="signals_and_systems",
                aliases=["LTI", "linear time invariant system"],
                description="Linear Time Invariant system.",
            ),
            Concept(
                concept_id="cpt_dsp_filter",
                canonical_name="Digital Filter",
                subject_id="dsp",
                aliases=["filter"],
                description="Digital signal processing filter.",
            ),
        ]

    def test_baseline_a_exact_match(self, sample_concepts: list[Concept]) -> None:
        mapper = ExactConceptMapper(concepts=sample_concepts)
        # Exact canonical query matches
        res = mapper.map_concept("Convolution", subject_id="signals_and_systems")
        assert res.primary_concept is not None
        assert res.primary_concept.concept_id == "cpt_ss_convolution"

        # Conversational query returns empty in exact baseline
        res2 = mapper.map_concept("What is convolution?", subject_id="signals_and_systems")
        assert res2.primary_concept is None

    def test_baseline_b_lexical_match(self, sample_concepts: list[Concept]) -> None:
        mapper = LexicalConceptMapper(concepts=sample_concepts)
        res = mapper.map_concept("What is convolution?", subject_id="signals_and_systems")
        assert res.primary_concept is not None
        assert res.primary_concept.concept_id == "cpt_ss_convolution"
        assert (
            "terminology" in res.primary_concept.match_method
            or res.primary_concept.match_method == "lexical"
        )

    def test_baseline_c_embedding_mapper(self, sample_concepts: list[Concept]) -> None:
        provider = MockEmbeddingProvider()
        mapper = EmbeddingConceptMapper(
            embedding_provider=provider, concepts=sample_concepts
        )
        res = mapper.map_concept("What is convolution?", subject_id="signals_and_systems")
        assert len(res.candidates) > 0
        assert res.primary_concept is not None
        assert res.primary_concept.match_method in ["embedding", "semantic_embedding"]

    def test_baseline_d_hybrid_mapper(self, sample_concepts: list[Concept]) -> None:
        provider = MockEmbeddingProvider()
        mapper = HybridConceptMapper(
            embedding_provider=provider, concepts=sample_concepts
        )
        res = mapper.map_concept("convolution ante enti?", subject_id="signals_and_systems")
        assert res.primary_concept is not None
        assert res.primary_concept.concept_id == "cpt_ss_convolution"
        assert res.mapping_confidence > 0.50

    def test_unknown_concept(self, sample_concepts: list[Concept]) -> None:
        mapper = HybridConceptMapper(concepts=sample_concepts)
        res = mapper.map_concept("quantum chromodynamics", subject_id="signals_and_systems")
        # No known terminology matched; confidence is low or score indicates no exact/term match
        if res.primary_concept:
            assert res.primary_concept.confidence < 0.60

    def test_subject_isolation(self, sample_concepts: list[Concept]) -> None:
        mapper = HybridConceptMapper(concepts=sample_concepts)
        # Search in signals_and_systems must NOT return dsp concept
        res = mapper.map_concept("Digital Filter", subject_id="signals_and_systems")
        for cand in res.candidates:
            assert cand.concept_id != "cpt_dsp_filter"

        # Search in dsp must return dsp concept
        res_dsp = mapper.map_concept("Digital Filter", subject_id="dsp")
        assert res_dsp.primary_concept is not None
        assert res_dsp.primary_concept.concept_id == "cpt_dsp_filter"
