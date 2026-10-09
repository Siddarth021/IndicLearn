"""Concept Mappers implementing Baseline A, B, C, and D (Hybrid) (Phase 4).

Connects student queries across English, Telugu script, Romanized Telugu, and code-mixed
formulations to canonical academic concept IDs in the Knowledge Graph.
"""

from __future__ import annotations

from indiclearn.domain.concepts import Concept
from indiclearn.embeddings.base import EmbeddingProvider
from indiclearn.knowledge_graph.base import GraphStore
from indiclearn.nlp.concept_mapping.base import ConceptMapper
from indiclearn.nlp.concept_mapping.candidate_generator import CandidateGenerator
from indiclearn.nlp.models import ConceptCandidate, ConceptMappingResult, TechnicalTerm
from indiclearn.nlp.normalization.normalizer import UnifiedQueryNormalizer
from indiclearn.nlp.terminology.detector import GraphGroundedTermDetector


class BaseConceptMapper:
    """Common foundation for concept mappers managing GraphStore and concept retrieval."""

    def __init__(
        self,
        graph_store: GraphStore | None = None,
        concepts: list[Concept] | None = None,
        default_subject_id: str = "signals_and_systems",
    ) -> None:
        self.graph_store = graph_store
        self.default_subject_id = default_subject_id
        self._preloaded_concepts = concepts or []
        self.normalizer = UnifiedQueryNormalizer()
        self.generator = CandidateGenerator(normalizer=self.normalizer)
        self.term_detector = GraphGroundedTermDetector(
            graph_store=graph_store,
            default_subject_id=default_subject_id,
            concepts=concepts,
        )

    def _get_concepts(self, subject_id: str) -> list[Concept]:
        if self._preloaded_concepts:
            return [c for c in self._preloaded_concepts if c.subject_id == subject_id]
        if self.graph_store is not None:
            return self.graph_store.list_concepts(subject_id=subject_id)
        return []

    def _finalize_result(
        self,
        query: str,
        subject_id: str,
        candidates: list[ConceptCandidate],
        top_k: int = 3,
    ) -> ConceptMappingResult:
        """Assign ranks, detect ambiguity, and build structured ConceptMappingResult."""
        if not candidates:
            return ConceptMappingResult(
                query=query,
                subject_id=subject_id,
                primary_concept=None,
                candidates=[],
                is_ambiguous=False,
                mapping_confidence=0.0,
                unmapped_terms=[],
            )

        # Sort descending by score
        candidates.sort(key=lambda c: c.score, reverse=True)

        # Truncate to top_k and re-index ranks
        top_candidates = candidates[:top_k]
        for idx, cand in enumerate(top_candidates, start=1):
            cand.rank = idx

        primary = top_candidates[0]
        is_ambiguous = False
        score_margin: float | None = None

        # Compute margin between top candidate and runner-up
        if len(top_candidates) >= 2:
            second = top_candidates[1]
            score_margin = round(primary.score - second.score, 4)
            # Small margin between close competing candidates indicates ambiguity
            if score_margin < 0.08 and primary.score >= 0.40:
                is_ambiguous = True
        else:
            score_margin = round(primary.score, 4)

        return ConceptMappingResult(
            query=query,
            subject_id=subject_id,
            primary_concept=primary,
            candidates=top_candidates,
            is_ambiguous=is_ambiguous,
            mapping_confidence=primary.score,
            score_margin=score_margin,
            unmapped_terms=[],
        )

    def map_query(
        self,
        query: str,
        subject_id: str = "signals_and_systems",
        top_k: int = 3,
        detected_terms: list[TechnicalTerm] | None = None,
    ) -> ConceptMappingResult:
        """Alias for map_concept compatible with both interfaces."""
        return self.map_concept(
            query=query,
            subject_id=subject_id,
            top_k=top_k,
            detected_terms=detected_terms,
        )


class ExactConceptMapper(BaseConceptMapper, ConceptMapper):
    """Baseline A: Exact canonical and alias matching only."""

    def map_concept(
        self,
        query: str,
        subject_id: str = "signals_and_systems",
        top_k: int = 3,
        detected_terms: list[TechnicalTerm] | None = None,
    ) -> ConceptMappingResult:
        concepts = self._get_concepts(subject_id or self.default_subject_id)
        candidates = self.generator.generate_exact_candidates(query, concepts)
        return self._finalize_result(query, subject_id, candidates, top_k=top_k)


class LexicalConceptMapper(BaseConceptMapper, ConceptMapper):
    """Baseline B: Normalized lexical matching (exact + terminology + token overlap)."""

    def map_concept(
        self,
        query: str,
        subject_id: str = "signals_and_systems",
        top_k: int = 3,
        detected_terms: list[TechnicalTerm] | None = None,
    ) -> ConceptMappingResult:
        concepts = self._get_concepts(subject_id or self.default_subject_id)
        # 1. Exact matches
        exact_cands = self.generator.generate_exact_candidates(query, concepts)
        if exact_cands and exact_cands[0].score >= 0.95:
            return self._finalize_result(query, subject_id, exact_cands, top_k=top_k)

        # 2. Terminology matches
        terms = (
            detected_terms
            if detected_terms is not None
            else self.term_detector.detect_terms(query, subject_id=subject_id)
        )
        term_cands = self.generator.generate_terminology_candidates(terms, concepts)

        # 3. Lexical / token overlap
        lex_cands = self.generator.generate_lexical_candidates(query, concepts)

        # Merge by concept_id keeping highest score
        merged: dict[str, ConceptCandidate] = {}
        for c in exact_cands + term_cands + lex_cands:
            if c.concept_id not in merged or c.score > merged[c.concept_id].score:
                merged[c.concept_id] = c

        return self._finalize_result(query, subject_id, list(merged.values()), top_k=top_k)


class EmbeddingConceptMapper(BaseConceptMapper, ConceptMapper):
    """Baseline C: Dense semantic embedding ranking."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        graph_store: GraphStore | None = None,
        concepts: list[Concept] | None = None,
        default_subject_id: str = "signals_and_systems",
    ) -> None:
        super().__init__(graph_store=graph_store, concepts=concepts, default_subject_id=default_subject_id)
        if embedding_provider is None:
            from indiclearn.embeddings.mock import MockEmbeddingProvider

            self.embedding_provider = MockEmbeddingProvider()
        else:
            self.embedding_provider = embedding_provider
        self._concept_embeddings_cache: dict[str, list[float]] = {}

    def map_concept(
        self,
        query: str,
        subject_id: str = "signals_and_systems",
        top_k: int = 3,
        detected_terms: list[TechnicalTerm] | None = None,
    ) -> ConceptMappingResult:
        concepts = self._get_concepts(subject_id or self.default_subject_id)
        candidates = self.generator.generate_embedding_candidates(
            query=query,
            concepts=concepts,
            embedding_provider=self.embedding_provider,
            concept_embeddings_cache=self._concept_embeddings_cache,
        )
        return self._finalize_result(query, subject_id, candidates, top_k=top_k)


class HybridConceptMapper(BaseConceptMapper, ConceptMapper):
    """Baseline D (Primary System): Multi-signal hybrid mapper combining symbolic and semantic evidence."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        graph_store: GraphStore | None = None,
        concepts: list[Concept] | None = None,
        default_subject_id: str = "signals_and_systems",
        weight_terminology: float = 0.50,
        weight_lexical: float = 0.20,
        weight_semantic: float = 0.30,
    ) -> None:
        super().__init__(graph_store=graph_store, concepts=concepts, default_subject_id=default_subject_id)
        self.embedding_provider = embedding_provider
        self.weight_terminology = weight_terminology
        self.weight_lexical = weight_lexical
        self.weight_semantic = weight_semantic
        self._concept_embeddings_cache: dict[str, list[float]] = {}

    def map_concept(
        self,
        query: str,
        subject_id: str = "signals_and_systems",
        top_k: int = 3,
        detected_terms: list[TechnicalTerm] | None = None,
    ) -> ConceptMappingResult:
        concepts = self._get_concepts(subject_id or self.default_subject_id)
        if not concepts:
            return self._finalize_result(query, subject_id, [], top_k=top_k)

        # 1. High-confidence shortcut: exact canonical or alias match
        exact_cands = self.generator.generate_exact_candidates(query, concepts)
        if exact_cands and exact_cands[0].score >= 0.98:
            return self._finalize_result(query, subject_id, exact_cands, top_k=top_k)

        # 2. Terminology signal
        terms = (
            detected_terms
            if detected_terms is not None
            else self.term_detector.detect_terms(query, subject_id=subject_id)
        )
        term_cands = self.generator.generate_terminology_candidates(terms, concepts)
        term_scores = {c.concept_id: (c.score, c.evidence_spans) for c in term_cands}

        # 3. Lexical / Normalized token overlap signal
        lex_cands = self.generator.generate_lexical_candidates(query, concepts)
        lex_scores = {c.concept_id: (c.score, c.evidence_spans) for c in lex_cands}

        # 4. Dense semantic embedding signal
        dense_scores: dict[str, float] = {}
        if self.embedding_provider is not None:
            dense_cands = self.generator.generate_embedding_candidates(
                query=query,
                concepts=concepts,
                embedding_provider=self.embedding_provider,
                concept_embeddings_cache=self._concept_embeddings_cache,
            )
            dense_scores = {c.concept_id: c.score for c in dense_cands}

        # 5. Hybrid fusion
        candidate_map: dict[str, ConceptCandidate] = {}
        for cpt in concepts:
            cid = cpt.concept_id
            t_score, t_ev = term_scores.get(cid, (0.0, []))
            l_score, l_ev = lex_scores.get(cid, (0.0, []))
            d_score = dense_scores.get(cid, 0.0)

            # If embedding provider is absent, re-weight between term and lexical
            if self.embedding_provider is None:
                final_score = t_score * 0.70 + l_score * 0.30
            else:
                # If term match is explicit and strong (> 0.85), boost score
                if t_score >= 0.85:
                    final_score = t_score * 0.60 + d_score * 0.25 + l_score * 0.15
                else:
                    final_score = (
                        t_score * self.weight_terminology
                        + l_score * self.weight_lexical
                        + d_score * self.weight_semantic
                    )

            if final_score > 0.10:
                evidence = list(set(t_ev + l_ev))
                match_method = (
                    "hybrid_terminology_boost"
                    if t_score >= 0.85
                    else ("hybrid_dense_dominant" if d_score > max(t_score, l_score) else "hybrid_fusion")
                )
                candidate_map[cid] = ConceptCandidate(
                    concept_id=cid,
                    canonical_name=cpt.canonical_name,
                    score=round(final_score, 4),
                    match_method=match_method,
                    evidence_spans=evidence,
                    metadata={
                        "term_score": round(t_score, 4),
                        "lex_score": round(l_score, 4),
                        "dense_score": round(d_score, 4),
                    },
                )

        return self._finalize_result(query, subject_id, list(candidate_map.values()), top_k=top_k)
