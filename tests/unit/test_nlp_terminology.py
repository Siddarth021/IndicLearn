"""Unit tests for Graph-Grounded Technical Term Detection (Phase 4)."""

import pytest

from indiclearn.domain.concepts import Concept
from indiclearn.knowledge_graph.networkx_store import NetworkXGraphStore
from indiclearn.nlp.terminology.detector import GraphGroundedTermDetector


class TestTechnicalTermDetection:
    """Test suite for GraphGroundedTermDetector."""

    @pytest.fixture
    def sample_concepts(self) -> list[Concept]:
        return [
            Concept(
                concept_id="cpt_signals_and_systems_convolution",
                canonical_name="Convolution",
                subject_id="signals_and_systems",
                aliases=["convolution operation", "కన్వల్యూషన్", "convolution ante"],
            ),
            Concept(
                concept_id="cpt_signals_and_systems_lti_system",
                canonical_name="LTI System",
                subject_id="signals_and_systems",
                aliases=["LTI", "linear time invariant system", "LTI సిస్టమ్"],
            ),
            Concept(
                concept_id="cpt_signals_and_systems_region_of_convergence",
                canonical_name="Region of Convergence",
                subject_id="signals_and_systems",
                aliases=["ROC"],
            ),
            Concept(
                concept_id="cpt_signals_and_systems_continuous_time_fourier_transform",
                canonical_name="Continuous-Time Fourier Transform",
                subject_id="signals_and_systems",
                aliases=["CTFT", "ఫోరియర్ ట్రాన్స్‌ఫార్మ్"],
            ),
        ]

    @pytest.fixture
    def term_detector(self, sample_concepts: list[Concept]) -> GraphGroundedTermDetector:
        return GraphGroundedTermDetector(concepts=sample_concepts)

    def test_canonical_concept_match(
        self, term_detector: GraphGroundedTermDetector
    ) -> None:
        terms = term_detector.detect_terms("What is convolution?", subject_id="signals_and_systems")
        assert len(terms) >= 1
        t = terms[0]
        assert t.concept_id == "cpt_signals_and_systems_convolution"
        assert t.canonical_name == "Convolution"
        assert t.term_type == "canonical"

    def test_acronym_match(self, term_detector: GraphGroundedTermDetector) -> None:
        terms = term_detector.detect_terms("Explain LTI properties", subject_id="signals_and_systems")
        lti_term = next((t for t in terms if t.matched_text == "LTI"), None)
        assert lti_term is not None
        assert lti_term.concept_id == "cpt_signals_and_systems_lti_system"
        assert lti_term.term_type == "acronym"

    def test_telugu_alias_match(self, term_detector: GraphGroundedTermDetector) -> None:
        terms = term_detector.detect_terms("కన్వల్యూషన్ అంటే ఏమిటి?", subject_id="signals_and_systems")
        assert len(terms) >= 1
        assert terms[0].concept_id == "cpt_signals_and_systems_convolution"
        assert terms[0].term_type == "telugu_alias"

    def test_multi_word_technical_term(
        self, term_detector: GraphGroundedTermDetector
    ) -> None:
        query = "Explain continuous-time Fourier transform derivation"
        terms = term_detector.detect_terms(query, subject_id="signals_and_systems")
        ctft_term = next((t for t in terms if t.canonical_name == "Continuous-Time Fourier Transform"), None)
        assert ctft_term is not None
        assert ctft_term.start >= 0
        assert ctft_term.end > ctft_term.start

    def test_grounded_in_graph_store(self, sample_concepts: list[Concept]) -> None:
        store = NetworkXGraphStore(default_subject_id="signals_and_systems")
        for c in sample_concepts:
            store.add_concept(c)

        detector = GraphGroundedTermDetector.from_graph_store(store)
        terms = detector.detect_terms("What is ROC?", subject_id="signals_and_systems")
        assert len(terms) == 1
        assert terms[0].concept_id == "cpt_signals_and_systems_region_of_convergence"
