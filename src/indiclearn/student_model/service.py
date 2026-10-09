"""Knowledge State Service Layer (Phase 6).

Coordinates student knowledge state retrieval, evidence recording, provenance explanation,
and prerequisite-aware readiness inspection.

Maintains strict separation:
- Canonical Academic Knowledge Graph is never mutated.
- Prerequisite querying provides contextual readiness analysis without state propagation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from indiclearn.knowledge_graph.base import GraphStore
from indiclearn.student_model.estimator import EvidenceBasedKnowledgeEstimator, KnowledgeEstimator
from indiclearn.student_model.models import (
    ConceptStateExplanation,
    EvidenceEvent,
    EvidenceType,
    KnowledgeState,
    MasteryStatus,
    PrerequisiteReadiness,
    StudentKnowledgeSummary,
    default_strength_for_type,
)
from indiclearn.student_model.repository import (
    InMemoryKnowledgeStateRepository,
    KnowledgeStateRepository,
)

logger = logging.getLogger(__name__)


class KnowledgeStateService:
    """Service layer managing student concept knowledge states and evidence logs."""

    def __init__(
        self,
        repository: KnowledgeStateRepository | None = None,
        estimator: KnowledgeEstimator | None = None,
        graph_store: GraphStore | None = None,
    ) -> None:
        self.repository = repository or InMemoryKnowledgeStateRepository()
        self.estimator = estimator or EvidenceBasedKnowledgeEstimator()
        self.graph_store = graph_store

    def record_event(self, event: EvidenceEvent) -> KnowledgeState:
        """Record an evidence event and update the student's knowledge state."""
        self.repository.add_event(event)

        current_state = self.repository.get_state(event.student_id, event.concept_id)
        if current_state is None:
            current_state = self.estimator.create_initial_state(
                student_id=event.student_id,
                concept_id=event.concept_id,
                subject_id=event.subject_id,
            )

        updated_state = self.estimator.update(current_state, event)
        self.repository.save_state(updated_state)
        return updated_state

    def record_interaction(
        self,
        student_id: str,
        concept_id: str,
        event_type: EvidenceType,
        score: float | None = None,
        subject_id: str = "signals_and_systems",
        source: str = "unspecified",
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeState:
        """Convenience method to construct and record an evidence event."""
        strength = default_strength_for_type(event_type)

        # Default scores if not provided
        if score is None:
            if event_type == EvidenceType.QUIZ_CORRECT:
                score = 1.0
            elif event_type == EvidenceType.QUIZ_INCORRECT:
                score = 0.0
            elif event_type in {EvidenceType.QUESTION_ASKED, EvidenceType.CONCEPT_VIEWED}:
                score = 0.50
            elif event_type == EvidenceType.SUCCESSFUL_TASK:
                score = 1.0
            elif event_type == EvidenceType.UNSUCCESSFUL_TASK:
                score = 0.0
            else:
                score = 0.50

        event = EvidenceEvent(
            student_id=student_id,
            concept_id=concept_id,
            subject_id=subject_id,
            event_type=event_type,
            strength=strength,
            score=max(0.0, min(1.0, score)),
            timestamp=datetime.now(timezone.utc),
            source=source,
            metadata=metadata or {},
        )
        return self.record_event(event)

    def record_events(self, events: list[EvidenceEvent]) -> list[KnowledgeState]:
        """Apply a batch of events grouped by concept."""
        if not events:
            return []

        # Group by (student_id, concept_id)
        grouped: dict[tuple[str, str], list[EvidenceEvent]] = {}
        for ev in events:
            self.repository.add_event(ev)
            grouped.setdefault((ev.student_id, ev.concept_id), []).append(ev)

        updated_states: list[KnowledgeState] = []
        for (student_id, concept_id), concept_events in grouped.items():
            current_state = self.repository.get_state(student_id, concept_id)
            concept_events.sort(key=lambda x: x.timestamp)
            updated = self.estimator.batch_update(current_state, concept_events)
            self.repository.save_state(updated)
            updated_states.append(updated)

        return updated_states

    def get_concept_state(
        self,
        student_id: str,
        concept_id: str,
        subject_id: str = "signals_and_systems",
    ) -> KnowledgeState:
        """Retrieve existing state or return a cold-start initial state."""
        state = self.repository.get_state(student_id, concept_id)
        if state is None:
            state = self.estimator.create_initial_state(student_id, concept_id, subject_id)
            self.repository.save_state(state)
        return state

    def get_student_summary(
        self,
        student_id: str,
        subject_id: str = "signals_and_systems",
    ) -> StudentKnowledgeSummary:
        """Compute aggregate summary metrics across all concepts tracked for a student."""
        states = self.repository.get_student_states(student_id, subject_id)
        if not states:
            return StudentKnowledgeSummary(
                student_id=student_id,
                subject_id=subject_id,
                total_concepts_tracked=0,
                mastered_count=0,
                developing_count=0,
                emerging_count=0,
                uncertain_count=0,
                unknown_count=0,
                mean_knowledge_score=0.50,
                mean_confidence=0.10,
            )

        counts = {
            MasteryStatus.MASTERED: 0,
            MasteryStatus.DEVELOPING: 0,
            MasteryStatus.EMERGING: 0,
            MasteryStatus.UNCERTAIN: 0,
            MasteryStatus.UNKNOWN: 0,
        }
        score_sum = 0.0
        conf_sum = 0.0

        for s in states:
            counts[s.mastery_status] = counts.get(s.mastery_status, 0) + 1
            score_sum += s.knowledge_score
            conf_sum += s.confidence

        n = len(states)
        return StudentKnowledgeSummary(
            student_id=student_id,
            subject_id=subject_id,
            total_concepts_tracked=n,
            mastered_count=counts[MasteryStatus.MASTERED],
            developing_count=counts[MasteryStatus.DEVELOPING],
            emerging_count=counts[MasteryStatus.EMERGING],
            uncertain_count=counts[MasteryStatus.UNCERTAIN],
            unknown_count=counts[MasteryStatus.UNKNOWN],
            mean_knowledge_score=round(score_sum / n, 4),
            mean_confidence=round(conf_sum / n, 4),
        )

    def get_weak_concepts(
        self,
        student_id: str,
        subject_id: str = "signals_and_systems",
        max_score: float = 0.50,
        min_confidence: float = 0.30,
    ) -> list[KnowledgeState]:
        """Identify concepts where student demonstrates difficulty supported by evidence.

        Notice: Cold-start concepts (confidence < min_confidence) are excluded
        to prevent treating 'unassessed' as 'weak'.
        """
        states = self.repository.get_student_states(student_id, subject_id)
        return [
            s
            for s in states
            if s.knowledge_score <= max_score
            and s.confidence >= min_confidence
            and s.strong_evidence_count > 0
        ]

    def get_uncertain_concepts(
        self,
        student_id: str,
        subject_id: str = "signals_and_systems",
    ) -> list[KnowledgeState]:
        """Retrieve concepts with conflicting assessment signals."""
        states = self.repository.get_student_states(student_id, subject_id)
        return [
            s
            for s in states
            if s.mastery_status == MasteryStatus.UNCERTAIN
            or s.metadata.get("has_conflicting_evidence", False)
        ]

    def explain_concept_state(
        self,
        student_id: str,
        concept_id: str,
    ) -> ConceptStateExplanation:
        """Provide transparent, auditable justification for a student's concept state."""
        state = self.get_concept_state(student_id, concept_id)
        evidence_summaries = [item.summary for item in state.evidence_history[-10:]]
        has_conflicting = state.metadata.get("has_conflicting_evidence", False)

        # Formulate human-readable explanation reason
        if state.evidence_count == 0:
            reason = (
                f"No assessment or interaction evidence recorded for concept '{concept_id}'. "
                f"Holding default prior belief ({state.knowledge_score:.2f}) with low confidence ({state.confidence:.2f})."
            )
        elif state.mastery_status == MasteryStatus.MASTERED:
            reason = (
                f"Mastery established: Supported by {state.strong_evidence_count} validated assessments "
                f"with high score ({state.knowledge_score:.2f}) and high confidence ({state.confidence:.2f})."
            )
        elif state.mastery_status == MasteryStatus.UNCERTAIN:
            reason = (
                "Uncertain state: Assessment evidence contains conflicting correct and incorrect outcomes. "
                "Additional diagnostic evaluation recommended."
            )
        elif state.strong_evidence_count == 0 and state.weak_evidence_count > 0:
            reason = (
                f"Observational interactions only ({state.weak_evidence_count} questions/views). "
                f"Questions do not constitute proof of mastery or ignorance. Score remains near prior ({state.knowledge_score:.2f})."
            )
        elif state.mastery_status == MasteryStatus.EMERGING:
            if state.knowledge_score < 0.50:
                reason = (
                    f"Emerging competency: Assessed score ({state.knowledge_score:.2f}) indicates need for foundational "
                    f"remediation across {state.strong_evidence_count} assessments."
                )
            else:
                reason = (
                    f"Emerging competency: Early positive assessment signal ({state.knowledge_score:.2f}) with moderate "
                    f"confidence ({state.confidence:.2f}). Additional practice recommended to reach developing status."
                )
        elif state.mastery_status == MasteryStatus.DEVELOPING:
            reason = (
                f"Developing competency: Moderate understanding ({state.knowledge_score:.2f}) supported by "
                f"{state.strong_evidence_count} assessments at confidence {state.confidence:.2f}."
            )
        else:
            reason = (
                f"Estimated score {state.knowledge_score:.2f} at confidence {state.confidence:.2f} based on "
                f"{state.strong_evidence_count} assessments and {state.weak_evidence_count} observational interactions."
            )

        return ConceptStateExplanation(
            student_id=student_id,
            concept_id=concept_id,
            knowledge_score=state.knowledge_score,
            confidence=state.confidence,
            mastery_status=state.mastery_status,
            reason=reason,
            evidence_summary=evidence_summaries,
            strong_evidence_count=state.strong_evidence_count,
            weak_evidence_count=state.weak_evidence_count,
            has_conflicting_evidence=has_conflicting,
        )

    def get_prerequisite_readiness(
        self,
        student_id: str,
        target_concept_id: str,
    ) -> PrerequisiteReadiness:
        """Inspect prerequisite concepts in the Academic Knowledge Graph.

        Core Principle: Inspects graph relationships for pedagogical readiness context,
        WITHOUT mutating or propagating knowledge scores across edges.
        """
        if self.graph_store is None:
            return PrerequisiteReadiness(
                student_id=student_id,
                target_concept_id=target_concept_id,
                is_ready=True,
                prerequisite_states=[],
                unmastered_prerequisites=[],
                recommendation="Knowledge graph not configured; prerequisite check skipped.",
            )

        # Query direct academic prerequisites from the canonical graph
        prereqs = self.graph_store.get_prerequisites(target_concept_id, transitive=False)
        if not prereqs:
            return PrerequisiteReadiness(
                student_id=student_id,
                target_concept_id=target_concept_id,
                is_ready=True,
                prerequisite_states=[],
                unmastered_prerequisites=[],
                recommendation=f"Concept '{target_concept_id}' has no foundational prerequisites in the academic graph.",
            )

        prereq_details: list[dict[str, Any]] = []
        unmastered: list[str] = []

        for p in prereqs:
            p_state = self.get_concept_state(student_id, p.concept_id)
            prereq_details.append({
                "concept_id": p.concept_id,
                "canonical_name": p.canonical_name,
                "knowledge_score": p_state.knowledge_score,
                "confidence": p_state.confidence,
                "mastery_status": p_state.mastery_status.value,
            })
            # Consider unmastered if below developing threshold or unknown
            if p_state.mastery_status in {MasteryStatus.UNKNOWN, MasteryStatus.UNCERTAIN} or (
                p_state.knowledge_score < 0.50 and p_state.confidence >= 0.30
            ):
                unmastered.append(p.canonical_name)

        is_ready = len(unmastered) == 0
        if is_ready:
            rec = f"Student has established foundational understanding for all prerequisites of '{target_concept_id}'."
        else:
            rec = f"Review foundational prerequisites before proceeding to '{target_concept_id}': {', '.join(unmastered)}."

        return PrerequisiteReadiness(
            student_id=student_id,
            target_concept_id=target_concept_id,
            is_ready=is_ready,
            prerequisite_states=prereq_details,
            unmastered_prerequisites=unmastered,
            recommendation=rec,
        )
