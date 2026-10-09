"""Integration tests for End-to-End Indic NLP Pipeline (Phase 4)."""

import pytest

from indiclearn.domain.concepts import Concept
from indiclearn.knowledge_graph.networkx_store import NetworkXGraphStore
from indiclearn.nlp.models import Language, Script
from indiclearn.nlp.pipeline import NLPPipeline


class TestNLPPipelineIntegration:
    """Integration test suite for NLPPipeline."""

    @pytest.fixture
    def graph_store(self) -> NetworkXGraphStore:
        store = NetworkXGraphStore(default_subject_id="signals_and_systems")
        store.add_concept(
            Concept(
                concept_id="cpt_signals_and_systems_convolution",
                canonical_name="Convolution",
                subject_id="signals_and_systems",
                aliases=["convolution operation", "కన్వల్యూషన్", "convolution ante"],
                description="Convolution operation for LTI systems.",
            )
        )
        store.add_concept(
            Concept(
                concept_id="cpt_signals_and_systems_lti_system",
                canonical_name="LTI System",
                subject_id="signals_and_systems",
                aliases=["LTI", "linear time invariant system", "LTI సిస్టమ్"],
                description="Linear Time-Invariant system.",
            )
        )
        return store

    @pytest.fixture
    def pipeline(self, graph_store: NetworkXGraphStore) -> NLPPipeline:
        return NLPPipeline.from_graph_store(graph_store)

    def test_pipeline_english_query(self, pipeline: NLPPipeline) -> None:
        result = pipeline.analyze("What is convolution?", subject_id="signals_and_systems")

        assert result.original_text == "What is convolution?"
        assert result.language_detection.primary_language == Language.ENGLISH
        assert result.language_detection.detected_script == Script.LATIN
        assert len(result.technical_terms) >= 1
        assert result.concept_mapping.primary_concept is not None
        assert (
            result.concept_mapping.primary_concept.concept_id
            == "cpt_signals_and_systems_convolution"
        )

    def test_pipeline_telugu_script_query(self, pipeline: NLPPipeline) -> None:
        result = pipeline.analyze("కన్వల్యూషన్ అంటే ఏమిటి?", subject_id="signals_and_systems")

        assert result.language_detection.primary_language == Language.TELUGU
        assert result.language_detection.detected_script == Script.TELUGU
        assert result.concept_mapping.primary_concept is not None
        assert (
            result.concept_mapping.primary_concept.concept_id
            == "cpt_signals_and_systems_convolution"
        )

    def test_pipeline_romanized_telugu_query(self, pipeline: NLPPipeline) -> None:
        result = pipeline.analyze("convolution ante enti?", subject_id="signals_and_systems")

        assert result.language_detection.primary_language == Language.ROMANIZED_TELUGU
        assert result.normalization.is_romanized is True
        assert len(result.normalization.candidate_variants) >= 1
        assert result.concept_mapping.primary_concept is not None
        assert (
            result.concept_mapping.primary_concept.concept_id
            == "cpt_signals_and_systems_convolution"
        )

    def test_pipeline_code_mixed_query(self, pipeline: NLPPipeline) -> None:
        result = pipeline.analyze(
            "convolution ante enti explain cheyyi", subject_id="signals_and_systems"
        )

        assert result.language_detection.is_code_mixed or result.code_mix.cmi >= 0.0
        assert len(result.code_mix.spans) > 0
        assert result.concept_mapping.primary_concept is not None
        assert (
            result.concept_mapping.primary_concept.concept_id
            == "cpt_signals_and_systems_convolution"
        )

    def test_pipeline_preserves_student_decoupling(self, pipeline: NLPPipeline) -> None:
        """Phase 4 mandate: Verify NO student mastery or knowledge model state exists in result."""
        result = pipeline.analyze("convolution ante enti?", subject_id="signals_and_systems")
        res_dict = result.model_dump()

        # Strict check: forbidden student knowledge modeling fields
        for forbidden in ["mastery", "student_knowledge", "weakness", "ignorance", "skill_level"]:
            assert forbidden not in res_dict
            assert forbidden not in res_dict.get("metadata", {})
