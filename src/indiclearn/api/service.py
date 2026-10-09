"""Dashboard Application Service Layer (Phase 8).

Wires together the backend research components:
- Canonical Academic Knowledge Graph (read-only)
- Student Knowledge Model (KnowledgeStateService)
- Adaptive Assessment Core (AdaptiveAssessmentService)
- Personalized Recommendation Planner (RecommendationService)
- Academic Curriculum Corpus (source chunks)

Core Principle:
This application service mediates between the HTTP API and domain services.
It never re-implements scoring or recommendation heuristics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from indiclearn.api.schemas import (
    AssessmentItemView,
    AssessmentResultView,
    AssessmentSessionView,
    ChunkDetailView,
    ConceptDetailView,
    EvidenceItemView,
    GraphEdgeView,
    GraphNodeView,
    KnowledgeMapView,
    RecommendationCardView,
    StudentDashboardOverview,
    SubjectSummary,
)
from indiclearn.assessment.repository import InMemoryAssessmentItemRepository
from indiclearn.assessment.service import AdaptiveAssessmentService
from indiclearn.domain.concepts import Concept, ConceptRelation, RelationType
from indiclearn.knowledge_graph.base import GraphStore
from indiclearn.knowledge_graph.networkx_store import NetworkXGraphStore
from indiclearn.recommendation.service import RecommendationService
from indiclearn.student_model.models import (
    EvidenceType,
    MasteryStatus,
)
from indiclearn.student_model.service import KnowledgeStateService

logger = logging.getLogger(__name__)


class DashboardApplicationService:
    """Primary application service backing student dashboard and knowledge map views."""

    def __init__(
        self,
        graph_store: GraphStore,
        knowledge_service: KnowledgeStateService,
        assessment_service: AdaptiveAssessmentService,
        recommendation_service: RecommendationService,
        chunks_data_path: str | Path | None = "data/evaluation/signals_and_systems_eval.json",
    ) -> None:
        self.graph_store = graph_store
        self.knowledge_service = knowledge_service
        self.assessment_service = assessment_service
        self.recommendation_service = recommendation_service
        self._chunks: dict[str, dict[str, Any]] = {}
        self._subjects: dict[str, dict[str, Any]] = {
            "signals_and_systems": {
                "name": "Signals and Systems",
                "description": "Core undergraduate electrical and electronics engineering curriculum.",
            }
        }

        if chunks_data_path:
            p = Path(chunks_data_path)
            if p.exists():
                try:
                    with open(p, encoding="utf-8") as f:
                        data = json.load(f)
                    for chk in data.get("corpus", []):
                        self._chunks[chk["chunk_id"]] = chk
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("Failed to load chunk corpus from %s: %s", p, exc)

    @classmethod
    def create_default(
        cls,
        graph_path: str | Path = "data/evaluation/concept_graph_gold.json",
        items_path: str | Path = "data/evaluation/signals_systems_items.json",
        chunks_path: str | Path = "data/evaluation/signals_and_systems_eval.json",
    ) -> DashboardApplicationService:
        """Factory initializing application service with gold standard fixtures."""
        # 1. Academic Knowledge Graph
        graph = NetworkXGraphStore()
        gp = Path(graph_path)
        if gp.exists():
            with open(gp, encoding="utf-8") as f:
                gdata = json.load(f)
            for c in gdata.get("concepts", []):
                graph.add_concept(
                    Concept(
                        concept_id=c["concept_id"],
                        subject_id=c.get("subject_id", "signals_and_systems"),
                        canonical_name=c["canonical_name"],
                        concept_type=c.get("concept_type", "concept"),
                        aliases=c.get("aliases", []),
                        description=c.get("description"),
                        source_chunk_ids=c.get("source_chunk_ids", []),
                    )
                )
            for r in gdata.get("relations", []):
                graph.add_relation(
                    ConceptRelation(
                        relation_id=f"rel_{r['source_concept_id']}_{r['relation_type']}_{r['target_concept_id']}",
                        subject_id="signals_and_systems",
                        source_concept_id=r["source_concept_id"],
                        target_concept_id=r["target_concept_id"],
                        relation_type=RelationType(r["relation_type"]),
                    )
                )

        # 2. Student Knowledge Model
        k_service = KnowledgeStateService(graph_store=graph)

        # 3. Adaptive Assessment Core
        ip = Path(items_path)
        if ip.exists():
            item_repo = InMemoryAssessmentItemRepository.from_json_file(ip)
        else:
            item_repo = InMemoryAssessmentItemRepository()

        ass_service = AdaptiveAssessmentService(
            item_repository=item_repo,
            knowledge_service=k_service,
            graph_store=graph,
        )

        # 4. Personalized Recommendation Planner
        rec_service = RecommendationService(
            knowledge_service=k_service,
            graph_store=graph,
            assessment_service=ass_service,
        )

        return cls(
            graph_store=graph,
            knowledge_service=k_service,
            assessment_service=ass_service,
            recommendation_service=rec_service,
            chunks_data_path=chunks_path,
        )

    def get_subjects(self) -> list[SubjectSummary]:
        """List academic subjects."""
        summaries: list[SubjectSummary] = []
        for s_id, s_info in self._subjects.items():
            count_c = self.graph_store.node_count(s_id)
            count_r = self.graph_store.edge_count(s_id)
            summaries.append(
                SubjectSummary(
                    subject_id=s_id,
                    name=s_info["name"],
                    concept_count=count_c,
                    relation_count=count_r,
                    description=s_info["description"],
                )
            )
        return summaries

    def get_knowledge_map(
        self,
        student_id: str,
        subject_id: str = "signals_and_systems",
    ) -> KnowledgeMapView:
        """Render academic graph nodes with student knowledge state overlay."""
        concepts = self.graph_store.list_concepts(subject_id=subject_id)
        relations = self.graph_store.get_relations(subject_id=subject_id)

        node_views: list[GraphNodeView] = []
        for c in concepts:
            st = self.knowledge_service.get_concept_state(student_id, c.concept_id, subject_id)
            node_views.append(
                GraphNodeView(
                    id=c.concept_id,
                    name=c.canonical_name,
                    type=c.concept_type.value,
                    aliases=c.aliases,
                    description=c.description,
                    source_chunk_ids=c.source_chunk_ids,
                    knowledge_score=st.knowledge_score,
                    confidence=st.confidence,
                    mastery_status=st.mastery_status,
                    evidence_count=st.evidence_count,
                    strong_evidence_count=st.strong_evidence_count,
                    weak_evidence_count=st.weak_evidence_count,
                    last_interaction_at=st.last_interaction_at,
                    last_assessed_at=st.last_assessed_at,
                )
            )

        edge_views: list[GraphEdgeView] = []
        for r in relations:
            label = r.relation_type.value.replace("_", " ").lower()
            edge_views.append(
                GraphEdgeView(
                    id=r.relation_id,
                    source=r.source_concept_id,
                    target=r.target_concept_id,
                    relation_type=r.relation_type.value,
                    label=label,
                )
            )

        return KnowledgeMapView(
            subject_id=subject_id,
            student_id=student_id,
            nodes=node_views,
            edges=edge_views,
        )

    def get_concept_details(
        self,
        student_id: str,
        concept_id: str,
        subject_id: str = "signals_and_systems",
    ) -> ConceptDetailView:
        """Retrieve full details for concept drawer."""
        c = self.graph_store.get_concept(concept_id)
        if c is None:
            raise KeyError(f"Concept '{concept_id}' not found in academic graph.")

        st = self.knowledge_service.get_concept_state(student_id, concept_id, subject_id)
        expl = self.knowledge_service.explain_concept_state(student_id, concept_id)

        # Graph relationships
        prereqs = self.graph_store.get_prerequisites(concept_id, transitive=False)
        prereq_list = [{"id": p.concept_id, "name": p.canonical_name} for p in prereqs]

        dependents = self.graph_store.get_dependents(concept_id, transitive=False)
        dep_list = [{"id": d.concept_id, "name": d.canonical_name} for d in dependents]

        # Recommendation context
        cands = self.recommendation_service.get_candidates(student_id, subject_id)
        cand_match = next((cand for cand in cands if cand.concept_id == concept_id), None)

        # Evidence history items
        ev_items: list[EvidenceItemView] = []
        for item in st.evidence_history[-10:]:
            ev_items.append(
                EvidenceItemView(
                    event_id=item.event_id,
                    event_type=item.event_type.value,
                    strength=item.strength.value,
                    score=item.score,
                    timestamp=item.timestamp,
                    summary=item.summary,
                )
            )

        source_avail = len(c.source_chunk_ids) > 0

        return ConceptDetailView(
            concept_id=c.concept_id,
            canonical_name=c.canonical_name,
            concept_type=c.concept_type.value,
            description=c.description,
            subject_id=subject_id,
            aliases=c.aliases,
            source_chunk_ids=c.source_chunk_ids,
            source_available=source_avail,
            prerequisites=prereq_list,
            dependents=dep_list,
            knowledge_score=st.knowledge_score,
            confidence=st.confidence,
            mastery_status=st.mastery_status,
            explanation_reason=expl.reason,
            evidence_count=st.evidence_count,
            strong_evidence_count=st.strong_evidence_count,
            weak_evidence_count=st.weak_evidence_count,
            has_conflicting_evidence=expl.has_conflicting_evidence,
            evidence_history=ev_items,
            last_interaction_at=st.last_interaction_at,
            last_assessed_at=st.last_assessed_at,
            recommended_action=cand_match.action_type if cand_match else None,
            recommendation_reasons=cand_match.reasons if cand_match else [],
            recommendation_priority=cand_match.priority_score if cand_match else None,
        )

    def get_student_dashboard(
        self,
        student_id: str,
        subject_id: str = "signals_and_systems",
    ) -> StudentDashboardOverview:
        """Calculate dashboard summary statistics across all curriculum concepts."""
        concepts = self.graph_store.list_concepts(subject_id=subject_id)
        total_concepts = len(concepts)

        states = [
            self.knowledge_service.get_concept_state(student_id, c.concept_id, subject_id)
            for c in concepts
        ]

        mastered = sum(1 for s in states if s.mastery_status == MasteryStatus.MASTERED)
        developing = sum(1 for s in states if s.mastery_status == MasteryStatus.DEVELOPING)
        emerging = sum(1 for s in states if s.mastery_status == MasteryStatus.EMERGING)
        uncertain = sum(1 for s in states if s.mastery_status == MasteryStatus.UNCERTAIN)
        unknown = sum(1 for s in states if s.mastery_status == MasteryStatus.UNKNOWN)

        evidence_count = sum(1 for s in states if s.evidence_count > 0)
        assessed_count = sum(1 for s in states if s.strong_evidence_count > 0)
        recent_count = sum(len(s.evidence_history) for s in states)

        mean_score = sum(s.knowledge_score for s in states) / total_concepts if total_concepts else 0.50
        mean_conf = sum(s.confidence for s in states) / total_concepts if total_concepts else 0.0

        return StudentDashboardOverview(
            student_id=student_id,
            subject_id=subject_id,
            total_concepts_tracked=total_concepts,
            concepts_with_evidence=evidence_count,
            concepts_assessed=assessed_count,
            mastered_count=mastered,
            developing_count=developing,
            emerging_count=emerging,
            uncertain_count=uncertain,
            unknown_count=unknown,
            mean_knowledge_score=round(mean_score, 4),
            mean_confidence=round(mean_conf, 4),
            recent_activity_count=recent_count,
        )

    def get_recommendations(
        self,
        student_id: str,
        subject_id: str = "signals_and_systems",
        top_k: int = 5,
    ) -> list[RecommendationCardView]:
        """Obtain prioritized personalized recommendations."""
        recs = self.recommendation_service.get_recommendations(
            student_id=student_id,
            subject_id=subject_id,
            top_k=top_k,
        )
        return [
            RecommendationCardView(
                recommendation_id=r.recommendation_id,
                student_id=r.student_id,
                concept_id=r.concept_id,
                canonical_name=r.canonical_name,
                action_type=r.action_type,
                priority_score=r.priority_score,
                reason_codes=r.reason_codes,
                explanation=r.explanation,
                knowledge_score=r.knowledge_score,
                confidence=r.confidence,
                mastery_status=r.mastery_status,
                source_chunk_ids=r.source_chunk_ids,
                source_available=r.source_available,
                suggested_difficulty=r.suggested_difficulty,
            )
            for r in recs
        ]

    def start_assessment(
        self,
        student_id: str,
        subject_id: str = "signals_and_systems",
        concept_id: str | None = None,
        requested_language: str = "en",
    ) -> AssessmentSessionView:
        """Start an assessment session, selecting the next eligible item."""
        session, explanation = self.assessment_service.start_session(
            student_id=student_id,
            subject_id=subject_id,
            requested_language=requested_language,
        )

        item_view = None
        concept_reasons = []
        if explanation and explanation.item:
            it = explanation.item
            item_view = AssessmentItemView(
                item_id=it.item_id,
                concept_id=it.concept_id,
                question=it.question,
                question_type=it.question_type,
                difficulty=it.difficulty,
                language=it.language,
                options=it.options,
                source_chunk_ids=it.source_chunk_ids,
            )
            concept_reasons = explanation.concept_reasons

        return AssessmentSessionView(
            session_id=session.session_id,
            student_id=session.student_id,
            subject_id=session.subject_id,
            is_active=session.is_active,
            current_index=session.current_index,
            total_answered=len(session.responses),
            current_item=item_view,
            concept_reasons=concept_reasons,
        )

    def submit_assessment_response(
        self,
        session_id: str,
        item_id: str,
        answer: str,
        response_time_ms: int | None = None,
        confidence_if_provided: float | None = None,
        requested_language: str = "en",
    ) -> AssessmentResultView:
        """Submit answer, evaluate, update student model, and adaptively select next item."""
        response, updated_state, next_explanation = self.assessment_service.submit_response(
            session_id=session_id,
            item_id=item_id,
            selected_answer=answer,
            response_time_ms=response_time_ms,
            confidence_if_provided=confidence_if_provided,
            requested_language=requested_language,
        )

        next_item_view = None
        session_completed = next_explanation is None
        if next_explanation and next_explanation.item:
            nit = next_explanation.item
            next_item_view = AssessmentItemView(
                item_id=nit.item_id,
                concept_id=nit.concept_id,
                question=nit.question,
                question_type=nit.question_type,
                difficulty=nit.difficulty,
                language=nit.language,
                options=nit.options,
                source_chunk_ids=nit.source_chunk_ids,
            )

        return AssessmentResultView(
            session_id=session_id,
            item_id=item_id,
            selected_answer=answer,
            correct=response.correct,
            feedback=response.feedback,
            updated_knowledge_score=updated_state.knowledge_score,
            updated_confidence=updated_state.confidence,
            updated_mastery_status=updated_state.mastery_status,
            next_item=next_item_view,
            session_completed=session_completed,
        )

    def get_chunk(self, chunk_id: str) -> ChunkDetailView | None:
        """Inspect grounded source chunk text and provenance."""
        chk = self._chunks.get(chunk_id)
        if not chk:
            return None
        return ChunkDetailView(
            chunk_id=chk["chunk_id"],
            document_id=chk.get("document_id", "doc_unknown"),
            subject_id=chk.get("subject_id", "signals_and_systems"),
            page_number=chk.get("page_number"),
            section=chk.get("section"),
            text=chk.get("text", ""),
            language=chk.get("language", "en"),
            concept_ids=chk.get("concept_ids", []),
        )

    def list_concepts(self, subject_id: str = "signals_and_systems") -> list[dict[str, Any]]:
        """List all concepts for a subject."""
        concepts = self.graph_store.list_concepts(subject_id=subject_id)
        return [
            {
                "concept_id": c.concept_id,
                "canonical_name": c.canonical_name,
                "concept_type": c.concept_type.value,
                "description": c.description,
                "subject_id": c.subject_id,
                "aliases": c.aliases,
                "source_chunk_ids": c.source_chunk_ids,
            }
            for c in concepts
        ]

    def get_assessment_session(self, session_id: str) -> AssessmentSessionView | None:
        """Retrieve existing assessment session status."""
        session = self.assessment_service.session_repository.get_session(session_id)
        if not session:
            return None
        item_view = None
        current_item_id = None
        if session.is_active and session.current_index < len(session.item_ids):
            current_item_id = session.item_ids[session.current_index]
        if current_item_id:
            it = self.assessment_service.item_repository.get_item(current_item_id)
            if it:
                item_view = AssessmentItemView(
                    item_id=it.item_id,
                    concept_id=it.concept_id,
                    question=it.question,
                    question_type=it.question_type,
                    difficulty=it.difficulty,
                    language=it.language,
                    options=it.options,
                    source_chunk_ids=it.source_chunk_ids,
                )
        return AssessmentSessionView(
            session_id=session.session_id,
            student_id=session.student_id,
            subject_id=session.subject_id,
            is_active=session.is_active,
            current_index=session.current_index,
            total_answered=len(session.responses),
            current_item=item_view,
            concept_reasons=[],
        )

    def load_demo_data(self, student_id: str) -> dict[str, Any]:
        """Populate clearly isolated demonstration data.

        Explicit Policy: This is DEMO DATA for exploratory visualization;
        it is never mixed with real student evaluation benchmarks.
        """
        # 1. Mastered LTI System
        for _ in range(4):
            self.knowledge_service.record_interaction(
                student_id=student_id,
                concept_id="cpt_signals_and_systems_lti_system",
                event_type=EvidenceType.QUIZ_CORRECT,
                score=1.0,
                source="demo_fixture",
            )

        # 2. Developing Convolution (2 correct, 1 incorrect)
        self.knowledge_service.record_interaction(
            student_id=student_id,
            concept_id="cpt_signals_and_systems_convolution",
            event_type=EvidenceType.QUIZ_CORRECT,
            score=1.0,
            source="demo_fixture",
        )
        self.knowledge_service.record_interaction(
            student_id=student_id,
            concept_id="cpt_signals_and_systems_convolution",
            event_type=EvidenceType.QUIZ_INCORRECT,
            score=0.0,
            source="demo_fixture",
        )
        self.knowledge_service.record_interaction(
            student_id=student_id,
            concept_id="cpt_signals_and_systems_convolution",
            event_type=EvidenceType.QUIZ_CORRECT,
            score=1.0,
            source="demo_fixture",
        )

        # 3. Uncertain Impulse Response (conflicting outcomes)
        for sc in [1.0, 0.0, 1.0, 0.0]:
            self.knowledge_service.record_interaction(
                student_id=student_id,
                concept_id="cpt_signals_and_systems_impulse_response",
                event_type=EvidenceType.QUIZ_CORRECT if sc > 0.5 else EvidenceType.QUIZ_INCORRECT,
                score=sc,
                source="demo_fixture",
            )

        # 4. Weakness in Continuous-Time Fourier Transform
        for _ in range(3):
            self.knowledge_service.record_interaction(
                student_id=student_id,
                concept_id="cpt_signals_and_systems_continuous_time_fourier_transform",
                event_type=EvidenceType.QUIZ_INCORRECT,
                score=0.0,
                source="demo_fixture",
            )

        return {
            "status": "SUCCESS",
            "student_id": student_id,
            "message": "DEMO DATA loaded successfully for visualization testing.",
            "demo_concepts_modified": [
                "cpt_signals_and_systems_lti_system (MASTERED)",
                "cpt_signals_and_systems_convolution (DEVELOPING)",
                "cpt_signals_and_systems_impulse_response (UNCERTAIN)",
                "cpt_signals_and_systems_continuous_time_fourier_transform (EMERGING)",
            ],
        }

    def ingest_document(
        self,
        file_path: str | Path,
        filename: str,
        subject_id: str | None = None,
        subject_name: str | None = None,
    ) -> dict[str, Any]:
        """Ingest an uploaded document (PDF, PPTX, TXT, MD) and index its chunks and concepts."""
        import re

        from indiclearn.chunking.recursive import RecursiveCharChunker
        from indiclearn.concepts.extractor import HeuristicConceptExtractor
        from indiclearn.config.settings import load_settings
        from indiclearn.ingestion.service import IngestionService

        # Derive clean subject_id from subject_name if provided
        if subject_name and subject_name.strip():
            raw_name = subject_name.strip()
            derived_id = re.sub(r"[^a-zA-Z0-9]+", "_", raw_name.lower()).strip("_")
            subject_id = derived_id or "custom_topic"
            display_name = raw_name
        else:
            subject_id = (subject_id or "signals_and_systems").strip()
            display_name = subject_id.replace("_", " ").title()

        if subject_id not in self._subjects:
            self._subjects[subject_id] = {
                "name": display_name,
                "description": f"Curriculum and concepts extracted from {filename}.",
            }

        settings = load_settings()
        chunker = RecursiveCharChunker(
            chunk_size=settings.chunking.chunk_size,
            chunk_overlap=settings.chunking.chunk_overlap,
        )
        ingestion_service = IngestionService(settings=settings, chunker=chunker)
        result = ingestion_service.ingest(file_path, subject_id=subject_id)

        if not result.ok:
            return {
                "status": "FAILED",
                "filename": filename,
                "subject_id": subject_id,
                "subject_name": display_name,
                "message": result.document.error_message or "Document extraction failed.",
                "pages_extracted": 0,
                "chunks_indexed": 0,
                "concepts_discovered": 0,
            }

        # 1. Register chunks in memory
        indexed_chunk_ids: list[str] = []
        for chk in result.chunks:
            self._chunks[chk.chunk_id] = {
                "chunk_id": chk.chunk_id,
                "document_id": filename,
                "page_number": chk.page_number,
                "section": chk.section or f"Page {chk.page_number}",
                "text": chk.text,
                "language": "en",
                "concept_ids": [],
            }
            indexed_chunk_ids.append(chk.chunk_id)

        # 2. Extract concepts and relations
        extractor = HeuristicConceptExtractor(subject_id=subject_id)
        concepts, relations = extractor.extract(result.chunks)

        new_concepts_count = 0
        for c in concepts:
            if not self.graph_store.has_concept(c.concept_id):
                self.graph_store.add_concept(c)
                new_concepts_count += 1
            else:
                existing = self.graph_store.get_concept(c.concept_id)
                if existing:
                    merged = sorted(set(existing.source_chunk_ids + c.source_chunk_ids))
                    existing.source_chunk_ids = merged

        for r in relations:
            self.graph_store.add_relation(r)

        return {
            "status": "SUCCESS",
            "filename": filename,
            "subject_id": subject_id,
            "subject_name": display_name,
            "document_id": result.document.document_id,
            "pages_extracted": len(result.pages),
            "chunks_indexed": len(indexed_chunk_ids),
            "concepts_discovered": new_concepts_count,
            "message": (
                f"Successfully ingested {filename} into '{display_name}': {len(result.pages)} slides/pages, "
                f"{len(indexed_chunk_ids)} chunks, {new_concepts_count} new concepts added."
            ),
        }

