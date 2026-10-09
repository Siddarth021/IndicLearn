"""End-to-End Indic NLP Pipeline for student query analysis and concept mapping.

Orchestrates:
1. Language and Script Identification
2. Code-Mix Token Span Analysis
3. Unicode and Romanized Telugu Normalization
4. Graph-Grounded Technical Term Detection
5. Canonical Concept Mapping (Baselines A-D)

Strictly preserves student input and uncertainty, completely decoupled from student
knowledge/mastery models.
"""

from __future__ import annotations

import logging
from typing import Any

from indiclearn.knowledge_graph.base import GraphStore
from indiclearn.nlp.code_mix.base import CodeMixDetector
from indiclearn.nlp.code_mix.detector import RuleBasedCodeMixDetector
from indiclearn.nlp.concept_mapping.base import ConceptMapper
from indiclearn.nlp.concept_mapping.mapper import HybridConceptMapper
from indiclearn.nlp.language_detection.base import LanguageDetector
from indiclearn.nlp.language_detection.detector import HeuristicLanguageDetector
from indiclearn.nlp.models import NLPAnalysisResult
from indiclearn.nlp.normalization.base import TextNormalizer
from indiclearn.nlp.normalization.normalizer import UnifiedQueryNormalizer
from indiclearn.nlp.terminology.base import TechnicalTermDetector
from indiclearn.nlp.terminology.detector import GraphGroundedTermDetector

logger = logging.getLogger(__name__)


class NLPPipeline:
    """Modular Indic NLP Pipeline orchestrating multi-stage query understanding."""

    def __init__(
        self,
        language_detector: LanguageDetector | None = None,
        code_mix_detector: CodeMixDetector | None = None,
        normalizer: TextNormalizer | None = None,
        term_detector: TechnicalTermDetector | None = None,
        concept_mapper: ConceptMapper | None = None,
        graph_store: GraphStore | None = None,
    ) -> None:
        """Initialize pipeline with modular components.

        If components are not explicitly provided, sensible defaults are instantiated.
        """
        self.language_detector = language_detector or HeuristicLanguageDetector()
        self.code_mix_detector = code_mix_detector or RuleBasedCodeMixDetector()
        self.normalizer = normalizer or UnifiedQueryNormalizer()

        # Initialize terminology and concept mapping grounded in graph store if provided
        if term_detector is not None:
            self.term_detector = term_detector
        elif graph_store is not None:
            self.term_detector = GraphGroundedTermDetector.from_graph_store(graph_store)
        else:
            self.term_detector = GraphGroundedTermDetector()

        if concept_mapper is not None:
            self.concept_mapper = concept_mapper
        elif graph_store is not None:
            self.concept_mapper = HybridConceptMapper(
                concepts=graph_store.get_all_concepts(),
            )
        else:
            self.concept_mapper = HybridConceptMapper()

    @classmethod
    def from_graph_store(
        cls,
        graph_store: GraphStore,
        concept_mapper: ConceptMapper | None = None,
    ) -> NLPPipeline:
        """Factory method to instantiate a fully grounded pipeline from GraphStore."""
        term_detector = GraphGroundedTermDetector.from_graph_store(graph_store)
        if concept_mapper is None:
            concepts = graph_store.get_all_concepts()
            concept_mapper = HybridConceptMapper(concepts=concepts)

        return cls(
            term_detector=term_detector,
            concept_mapper=concept_mapper,
            graph_store=graph_store,
        )

    def analyze(
        self,
        query: str,
        subject_id: str = "signals_and_systems",
        metadata: dict[str, Any] | None = None,
    ) -> NLPAnalysisResult:
        """Execute full end-to-end analysis on a student query.

        Args:
            query: Raw student query text in English, Telugu, Romanized Telugu, or mixed.
            subject_id: Canonical subject namespace.
            metadata: Optional contextual metadata.

        Returns:
            NLPAnalysisResult containing structured outputs from all stages.
        """
        # Stage 1: Language and Script Identification
        lang_res = self.language_detector.detect(query)

        # Stage 2: Code-Mix Span Detection
        code_mix_res = self.code_mix_detector.detect_spans(query)

        # Stage 3: Normalization and Transliteration Candidate Generation
        norm_res = self.normalizer.normalize(query, primary_language=lang_res.primary_language)

        # Stage 4: Technical Term Detection (run on both original and normalized text)
        detected_terms = self.term_detector.detect_terms(query, subject_id=subject_id)
        if norm_res.normalized_text != query:
            norm_terms = self.term_detector.detect_terms(
                norm_res.normalized_text, subject_id=subject_id
            )
            # Merge without duplicate concept_ids/spans
            seen = {(t.concept_id, t.matched_span) for t in detected_terms}
            for nt in norm_terms:
                if (nt.concept_id, nt.matched_span) not in seen:
                    detected_terms.append(nt)
                    seen.add((nt.concept_id, nt.matched_span))

        # Stage 5: Concept Mapping
        mapping_res = self.concept_mapper.map_query(
            query=norm_res.normalized_text,
            subject_id=subject_id,
            detected_terms=detected_terms,
        )

        return NLPAnalysisResult(
            original_text=query,
            language_detection=lang_res,
            code_mix=code_mix_res,
            normalization=norm_res,
            technical_terms=detected_terms,
            concept_mapping=mapping_res,
            metadata=metadata or {},
        )
