"""Adaptive Assessment Service orchestrator (Phase 7A).

Coordinates the closed-loop adaptive assessment cycle:
1. Student Knowledge State inspection
2. Concept Candidate Generation & Multi-Signal Prioritization
3. Policy-driven Assessment Item Selection (Difficulty, Language, Provenance, Repeat Avoidance)
4. Session Management & Deterministic Answer Evaluation
5. Evidence Integration via KnowledgeStateService (emitting EvidenceEvents)
6. Dynamic Next-Item Selection based on updated knowledge states
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from indiclearn.assessment.candidate import (
    AssessmentCandidateGenerator,
    HeuristicCandidateGenerator,
)
from indiclearn.assessment.items import evaluate_response
from indiclearn.assessment.models import (
    AssessmentCandidate,
    AssessmentResponse,
    AssessmentSelectionExplanation,
    AssessmentSession,
)
from indiclearn.assessment.repository import (
    AssessmentItemRepository,
    AssessmentSessionRepository,
    InMemoryAssessmentSessionRepository,
)
from indiclearn.assessment.selector import AssessmentItemSelector, DeterministicItemSelector
from indiclearn.assessment.session import AssessmentSessionManager
from indiclearn.knowledge_graph.base import GraphStore
from indiclearn.student_model.models import (
    EvidenceEvent,
    EvidenceStrength,
    EvidenceType,
    KnowledgeState,
)
from indiclearn.student_model.service import KnowledgeStateService

logger = logging.getLogger(__name__)


class AdaptiveAssessmentService:
    """Primary service coordinating adaptive concept and item selection."""

    def __init__(
        self,
        item_repository: AssessmentItemRepository,
        knowledge_service: KnowledgeStateService,
        graph_store: GraphStore | None = None,
        candidate_generator: AssessmentCandidateGenerator | None = None,
        item_selector: AssessmentItemSelector | None = None,
        session_repository: AssessmentSessionRepository | None = None,
        default_language: str = "en",
    ) -> None:
        self.item_repository = item_repository
        self.knowledge_service = knowledge_service
        self.graph_store = graph_store
        self.session_repository = session_repository or InMemoryAssessmentSessionRepository()
        self.session_manager = AssessmentSessionManager(self.session_repository)
        self.default_language = default_language

        self.candidate_generator = candidate_generator or HeuristicCandidateGenerator(
            graph_store=graph_store,
            knowledge_service=knowledge_service,
        )
        self.item_selector = item_selector or DeterministicItemSelector(
            default_language=default_language,
        )

    def get_candidates(
        self,
        student_id: str,
        subject_id: str = "signals_and_systems",
        now: datetime | None = None,
    ) -> list[AssessmentCandidate]:
        """Generate and rank all concept assessment candidates for a student."""
        return self.candidate_generator.generate_candidates(
            student_id=student_id,
            subject_id=subject_id,
            now=now,
        )

    def select_next_assessment(
        self,
        student_id: str,
        subject_id: str = "signals_and_systems",
        requested_language: str | None = None,
        session_id: str | None = None,
        seen_item_ids: set[str] | None = None,
        now: datetime | None = None,
    ) -> AssessmentSelectionExplanation | None:
        """Select the highest-priority eligible concept and matching assessment item."""
        candidates = self.get_candidates(student_id=student_id, subject_id=subject_id, now=now)
        if not candidates:
            return None

        # Determine seen items
        effective_seen: set[str] = set()
        if seen_item_ids is not None:
            effective_seen.update(seen_item_ids)
        elif session_id is not None:
            session = self.session_manager.get_session(session_id)
            if session:
                effective_seen.update(session.item_ids)
        else:
            effective_seen.update(
                self.session_manager.get_seen_item_ids(student_id, subject_id)
            )

        # Iterate candidates by priority until an item can be selected
        all_items = self.item_repository.list_items(subject_id=subject_id)
        for cand in candidates:
            item, explanation = self.item_selector.select_item(
                candidate=cand,
                available_items=all_items,
                requested_language=requested_language or self.default_language,
                seen_item_ids=effective_seen,
            )
            if item is not None and explanation is not None:
                return explanation

        return None

    def start_session(
        self,
        student_id: str,
        subject_id: str = "signals_and_systems",
        requested_language: str | None = None,
    ) -> tuple[AssessmentSession, AssessmentSelectionExplanation | None]:
        """Initialize an adaptive assessment session and pick the first assessment item."""
        session = self.session_manager.create_session(student_id=student_id, subject_id=subject_id)

        explanation = self.select_next_assessment(
            student_id=student_id,
            subject_id=subject_id,
            requested_language=requested_language,
            session_id=session.session_id,
        )

        if explanation is not None:
            self.session_manager.add_item_to_session(session, explanation.item.item_id)

        return session, explanation

    def submit_response(
        self,
        session_id: str,
        item_id: str,
        selected_answer: str,
        response_time_ms: int | None = None,
        confidence_if_provided: float | None = None,
        requested_language: str | None = None,
        now: datetime | None = None,
    ) -> tuple[AssessmentResponse, KnowledgeState, AssessmentSelectionExplanation | None]:
        """Evaluate student response, emit EvidenceEvent to update Student Model, and adaptively select next item."""
        session = self.session_manager.get_session(session_id)
        if session is None:
            raise ValueError(f"Session with ID '{session_id}' not found.")
        if not session.is_active:
            raise ValueError(f"Session '{session_id}' is already closed/completed.")

        item = self.item_repository.get_item(item_id)
        if item is None:
            raise ValueError(f"Assessment item '{item_id}' not found.")

        # 1. Evaluate response deterministically
        is_correct, feedback = evaluate_response(item, selected_answer)

        # 2. Record AssessmentResponse
        ans_time = now or datetime.now(timezone.utc)
        response = AssessmentResponse(
            session_id=session.session_id,
            item_id=item.item_id,
            student_id=session.student_id,
            concept_id=item.concept_id,
            selected_answer=selected_answer,
            correct=is_correct,
            feedback=feedback,
            answered_at=ans_time,
            response_time_ms=response_time_ms,
            confidence_if_provided=confidence_if_provided,
            metadata={"difficulty": item.difficulty.value, "language": item.language},
        )
        self.session_manager.record_response(session, response)

        # 3. Evidence Integration via Student Knowledge Model
        # Crucial: The assessment engine emits an EvidenceEvent; the student model updates knowledge state.
        event_type = EvidenceType.QUIZ_CORRECT if is_correct else EvidenceType.QUIZ_INCORRECT
        score = 1.0 if is_correct else 0.0

        event = EvidenceEvent(
            student_id=session.student_id,
            concept_id=item.concept_id,
            subject_id=session.subject_id,
            event_type=event_type,
            strength=EvidenceStrength.STRONG,
            score=score,
            timestamp=ans_time,
            source="adaptive_assessment_service",
            metadata={
                "item_id": item.item_id,
                "session_id": session.session_id,
                "difficulty": item.difficulty.value,
                "question_type": item.question_type.value,
                "source_chunk_ids": item.source_chunk_ids,
            },
        )
        updated_state = self.knowledge_service.record_event(event)

        # 4. Adaptively select next item using updated knowledge state
        next_explanation = self.select_next_assessment(
            student_id=session.student_id,
            subject_id=session.subject_id,
            requested_language=requested_language or item.language,
            session_id=session.session_id,
            now=ans_time,
        )

        if next_explanation is not None:
            self.session_manager.add_item_to_session(session, next_explanation.item.item_id)

        return response, updated_state, next_explanation

    def complete_session(self, session_id: str) -> AssessmentSession:
        """Close an active assessment session."""
        session = self.session_manager.get_session(session_id)
        if session is None:
            raise ValueError(f"Session with ID '{session_id}' not found.")
        return self.session_manager.complete_session(session)
