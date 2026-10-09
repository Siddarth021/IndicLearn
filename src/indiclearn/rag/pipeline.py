"""End-to-End Retrieval-Augmented Generation (RAG) Pipeline for IndicLearn (Phase 5).

Orchestrates:
1. Student Query NLP Analysis & Cross-Lingual Concept Mapping (Phase 4)
2. Controlled Academic Curriculum Graph Context (Phase 3)
3. Concept-Aware Hybrid Retrieval (Phase 2)
4. Evidence Selection, Deduplication & Sufficiency Auditing
5. Structured Deterministic Context Assembly
6. LLM Completion via Decoupled Provider Protocol
7. Citation Attribution Parsing & Verification
8. Heuristic Grounding Verification & Hallucination Prevention
"""

from __future__ import annotations

import logging
import time
from typing import Any

from indiclearn.domain.concepts import Concept
from indiclearn.knowledge_graph.base import GraphStore
from indiclearn.llm.base import LLMMessage, LLMProvider
from indiclearn.llm.mock import MockLLMProvider
from indiclearn.nlp.pipeline import NLPPipeline
from indiclearn.rag.citation import CitationValidator
from indiclearn.rag.context_builder import ContextBuilder
from indiclearn.rag.evidence_selector import EvidenceSelector
from indiclearn.rag.grounding import GroundingVerifier
from indiclearn.rag.models import (
    EvidenceSet,
    GenerationMetadata,
    RAGRequest,
    RAGResponse,
)
from indiclearn.rag.policies import (
    AMBIGUOUS_SCOPE_MESSAGES,
    INSUFFICIENT_EVIDENCE_MESSAGES,
    OUT_OF_SCOPE_MESSAGES,
    RAG_SYSTEM_PROMPT,
    build_user_prompt,
)
from indiclearn.rag.scope import HeuristicScopeAssessor, ScopeAssessor
from indiclearn.retrieval.base import Retriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Modular RAG Pipeline orchestrating evidence retrieval, synthesis, and grounding."""

    def __init__(
        self,
        nlp_pipeline: NLPPipeline,
        retriever: Retriever,
        llm_provider: LLMProvider | None = None,
        graph_store: GraphStore | None = None,
        evidence_selector: EvidenceSelector | None = None,
        context_builder: ContextBuilder | None = None,
        citation_validator: CitationValidator | None = None,
        grounding_verifier: GroundingVerifier | None = None,
        scope_assessor: ScopeAssessor | None = None,
    ) -> None:
        self.nlp_pipeline = nlp_pipeline
        self.retriever = retriever
        self.llm_provider = llm_provider or MockLLMProvider()
        self.graph_store = graph_store
        self.evidence_selector = evidence_selector or EvidenceSelector()
        self.context_builder = context_builder or ContextBuilder()
        self.citation_validator = citation_validator or CitationValidator()
        self.grounding_verifier = grounding_verifier or GroundingVerifier(
            validator=self.citation_validator
        )
        self.scope_assessor = scope_assessor or HeuristicScopeAssessor()

    def _get_graph_context(
        self,
        concept_ids: list[str],
        max_hops: int = 1,
        max_concepts: int = 3,
    ) -> tuple[list[Concept], dict[str, list[str]]]:
        """Fetch direct prerequisites and related concepts from GraphStore without student state."""
        if not self.graph_store or not concept_ids or max_hops <= 0 or max_concepts <= 0:
            return [], {}

        graph_concepts: list[Concept] = []
        prerequisite_map: dict[str, list[str]] = {}
        seen_ids: set[str] = set()

        for cid in concept_ids[:max_concepts]:
            cpt = self.graph_store.get_concept(cid)
            if cpt and cpt.concept_id not in seen_ids:
                graph_concepts.append(cpt)
                seen_ids.add(cpt.concept_id)

                # Fetch direct prerequisites
                prereq_items = self.graph_store.get_prerequisites(cid)
                prereq_names = []
                for p in prereq_items:
                    p_cpt = p if isinstance(p, Concept) else self.graph_store.get_concept(p)
                    if p_cpt:
                        prereq_names.append(p_cpt.canonical_name)
                        if len(graph_concepts) < max_concepts and p_cpt.concept_id not in seen_ids:
                            graph_concepts.append(p_cpt)
                            seen_ids.add(p_cpt.concept_id)
                prerequisite_map[cid] = prereq_names

        return graph_concepts, prerequisite_map

    def answer(
        self,
        request: RAGRequest | str,
        **overrides: Any,
    ) -> RAGResponse:
        """Execute end-to-end RAG workflow for a student query.

        Args:
            request: RAGRequest instance or query string.
            **overrides: Optional runtime argument overrides.

        Returns:
            RAGResponse with generated answer, citations, evidence, and grounding report.
        """
        start_time = time.perf_counter()

        if isinstance(request, str):
            req = RAGRequest(query=request, **overrides)
        else:
            req = request

        # Stage 1: Phase 4 NLP Analysis & Cross-Lingual Concept Mapping
        nlp_res = self.nlp_pipeline.analyze(req.query, subject_id=req.subject_id)
        norm_query = nlp_res.normalization.normalized_text
        detected_lang = nlp_res.language_detection.primary_language.value
        requested_lang = req.requested_language or detected_lang

        concept_candidates = nlp_res.concept_mapping.candidates
        mapped_concept_ids = [c.concept_id for c in concept_candidates]

        # Stage 2: Retrieval (Phase 2) - Direct Academic Evidence Chunks
        retrieval_start = time.perf_counter()
        filters = {"subject_id": req.subject_id}

        # Query using normalized text
        raw_candidates = self.retriever.retrieve(
            query=norm_query,
            top_k=req.top_k * 2,
            filters=filters,
        )

        # Also retrieve with canonical concept names if available to maximize recall
        if nlp_res.concept_mapping.primary_concept:
            cpt_name = nlp_res.concept_mapping.primary_concept.canonical_name
            if cpt_name.lower() not in norm_query.lower():
                cpt_candidates = self.retriever.retrieve(
                    query=cpt_name,
                    top_k=req.top_k,
                    filters=filters,
                )
                raw_candidates.extend(cpt_candidates)

        retrieval_time_ms = round((time.perf_counter() - retrieval_start) * 1000, 2)

        # Stage 3: Curriculum Scope & Ambiguity Assessment (Phase 5.2)
        curriculum_concepts = (
            self.graph_store.list_concepts(subject_id=req.subject_id)
            if self.graph_store
            else None
        )
        scope_decision = self.scope_assessor.assess_scope(
            query=req.query,
            nlp_result=nlp_res,
            retrieval_candidates=raw_candidates,
            curriculum_concepts=curriculum_concepts,
        )

        # Handle explicit OUT_OF_SCOPE rejection
        if scope_decision.scope == "OUT_OF_SCOPE":
            fallback_msg = OUT_OF_SCOPE_MESSAGES.get(
                requested_lang, OUT_OF_SCOPE_MESSAGES["en"]
            )
            evidence_set = EvidenceSet(
                items=[],
                query=req.query,
                subject_id=req.subject_id,
                total_candidates=len(raw_candidates),
                insufficient_evidence=True,
                insufficiency_reason=scope_decision.reason,
            )
            grounding_report = self.grounding_verifier.verify_grounding(
                answer_text=fallback_msg,
                evidence_set=evidence_set,
                is_insufficient_evidence_response=True,
                fallback_status="OUT_OF_SCOPE",
            )
            total_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return RAGResponse(
                query=req.query,
                subject_id=req.subject_id,
                answer=fallback_msg,
                is_answerable=False,
                status="OUT_OF_SCOPE",
                citations=[],
                evidence=evidence_set,
                concept_ids=mapped_concept_ids,
                grounding_report=grounding_report,
                metadata=GenerationMetadata(
                    model_name=self.llm_provider.model_name,
                    requested_language=requested_lang,
                    generated_language=requested_lang,
                    prompt_tokens=0,
                    completion_tokens=len(fallback_msg.split()),
                    latency_ms=total_time_ms,
                    graph_concepts_used=[],
                    retrieval_time_ms=retrieval_time_ms,
                    metadata={"scope_decision": scope_decision.__dict__},
                ),
            )

        # Handle explicit AMBIGUOUS query phrase / margin rejection
        if scope_decision.scope == "AMBIGUOUS":
            fallback_msg = AMBIGUOUS_SCOPE_MESSAGES.get(
                requested_lang, AMBIGUOUS_SCOPE_MESSAGES["en"]
            )
            evidence_set = EvidenceSet(
                items=[],
                query=req.query,
                subject_id=req.subject_id,
                total_candidates=len(raw_candidates),
                insufficient_evidence=True,
                insufficiency_reason=scope_decision.reason,
            )
            grounding_report = self.grounding_verifier.verify_grounding(
                answer_text=fallback_msg,
                evidence_set=evidence_set,
                is_insufficient_evidence_response=True,
                fallback_status="AMBIGUOUS_SCOPE",
            )
            total_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return RAGResponse(
                query=req.query,
                subject_id=req.subject_id,
                answer=fallback_msg,
                is_answerable=False,
                status="AMBIGUOUS_SCOPE",
                citations=[],
                evidence=evidence_set,
                concept_ids=mapped_concept_ids,
                grounding_report=grounding_report,
                metadata=GenerationMetadata(
                    model_name=self.llm_provider.model_name,
                    requested_language=requested_lang,
                    generated_language=requested_lang,
                    prompt_tokens=0,
                    completion_tokens=len(fallback_msg.split()),
                    latency_ms=total_time_ms,
                    graph_concepts_used=[],
                    retrieval_time_ms=retrieval_time_ms,
                    metadata={"scope_decision": scope_decision.__dict__},
                ),
            )

        # Stage 4: Evidence Selection (Phase 5)
        evidence_set = self.evidence_selector.select_evidence(
            query=req.query,
            subject_id=req.subject_id,
            retrieval_results=raw_candidates,
            top_k=req.top_k,
            min_threshold=req.min_score_threshold,
        )

        # Handle Insufficient Evidence
        if (
            evidence_set.insufficient_evidence
            or not evidence_set.items
            or scope_decision.scope == "INSUFFICIENT_EVIDENCE"
        ):
            fallback_msg = INSUFFICIENT_EVIDENCE_MESSAGES.get(
                requested_lang, INSUFFICIENT_EVIDENCE_MESSAGES["en"]
            )
            grounding_report = self.grounding_verifier.verify_grounding(
                answer_text=fallback_msg,
                evidence_set=evidence_set,
                is_insufficient_evidence_response=True,
                fallback_status="INSUFFICIENT_EVIDENCE",
            )
            total_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return RAGResponse(
                query=req.query,
                subject_id=req.subject_id,
                answer=fallback_msg,
                is_answerable=False,
                status="INSUFFICIENT_EVIDENCE",
                citations=[],
                evidence=evidence_set,
                concept_ids=mapped_concept_ids,
                grounding_report=grounding_report,
                metadata=GenerationMetadata(
                    model_name=self.llm_provider.model_name,
                    requested_language=requested_lang,
                    generated_language=requested_lang,
                    prompt_tokens=0,
                    completion_tokens=len(fallback_msg.split()),
                    latency_ms=total_time_ms,
                    graph_concepts_used=[],
                    retrieval_time_ms=retrieval_time_ms,
                    metadata={"scope_decision": scope_decision.__dict__},
                ),
            )

        # Stage 5: Graph Context Enrichment (Phase 3 & Phase 5.2)
        # Note: Graph context enriches the answer explanation without competing
        # with direct evidence chunks in the retrieval candidate pool.
        graph_concepts, prereq_map = self._get_graph_context(
            concept_ids=mapped_concept_ids,
            max_hops=req.max_graph_hops,
            max_concepts=req.max_graph_concepts,
        )

        # Stage 6: Context Assembly
        context_str = self.context_builder.build_context(
            evidence_set=evidence_set,
            graph_concepts=graph_concepts,
            prerequisite_map=prereq_map,
        )

        # Stage 7: Prompt Synthesis & LLM Generation
        user_prompt = build_user_prompt(
            query=req.query,
            context_str=context_str,
            requested_language=requested_lang,
        )

        messages = [
            LLMMessage(role="system", content=RAG_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]

        llm_response = self.llm_provider.generate_messages(messages)
        answer_text = llm_response.text.strip()

        # Check if the LLM's answer indicates insufficient evidence
        is_insufficient = any(
            phrase in answer_text.lower()
            for phrase in [
                "insufficient information",
                "could not find",
                "not enough information",
                "సమాచారం దొరకలేదు",
                "information ledu",
            ]
        )

        # Stage 8: Citation Verification & Grounding Audit
        grounding_report = self.grounding_verifier.verify_grounding(
            answer_text=answer_text,
            evidence_set=evidence_set,
            is_insufficient_evidence_response=is_insufficient,
        )

        total_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return RAGResponse(
            query=req.query,
            subject_id=req.subject_id,
            answer=answer_text,
            is_answerable=not is_insufficient,
            status=grounding_report.status,
            citations=grounding_report.valid_citations,
            evidence=evidence_set,
            concept_ids=mapped_concept_ids,
            grounding_report=grounding_report,
            metadata=GenerationMetadata(
                model_name=llm_response.model_name,
                requested_language=requested_lang,
                generated_language=requested_lang,
                prompt_tokens=llm_response.prompt_tokens,
                completion_tokens=llm_response.completion_tokens,
                latency_ms=total_time_ms,
                graph_concepts_used=[c.canonical_name for c in graph_concepts],
                retrieval_time_ms=retrieval_time_ms,
                metadata=llm_response.metadata,
            ),
        )
