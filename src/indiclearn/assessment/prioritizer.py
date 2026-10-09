"""Interpretable multi-signal candidate prioritization policy (Phase 7A).

Evaluates concept candidates based on:
1. Weakness (low knowledge score with evidence)
2. Uncertainty (low confidence or conflicting assessment signals)
3. Prerequisite importance (foundational role for unmastered downstream concepts)
4. Coverage need (prioritizing unassessed / rarely assessed concepts)
5. Recency (penalizing concepts assessed very recently to avoid immediate repetition)

All weights and thresholds are configurable and explicitly designated as
heuristic research defaults rather than calibrated cognitive parameters.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from indiclearn.assessment.models import AssessmentCandidate
from indiclearn.domain.concepts import Concept
from indiclearn.knowledge_graph.base import GraphStore
from indiclearn.student_model.models import KnowledgeState, MasteryStatus

logger = logging.getLogger(__name__)


@runtime_checkable
class CandidatePrioritizer(Protocol):
    """Protocol for scoring and ranking assessment candidates."""

    def prioritize_candidate(
        self,
        concept: Concept,
        state: KnowledgeState,
        graph_store: GraphStore | None = None,
        now: datetime | None = None,
        student_states_lookup: dict[str, KnowledgeState] | None = None,
    ) -> AssessmentCandidate:
        """Compute multi-signal priority and explainability for a single concept."""
        ...

    def rank_candidates(
        self,
        candidates: list[AssessmentCandidate],
    ) -> list[AssessmentCandidate]:
        """Deterministically sort candidates by priority score descending."""
        ...


class MultiSignalCandidatePrioritizer:
    """Interpretable multi-signal candidate prioritizer."""

    def __init__(
        self,
        weight_weakness: float = 0.35,
        weight_uncertainty: float = 0.25,
        weight_prerequisite: float = 0.20,
        weight_coverage: float = 0.12,
        weight_recency: float = 0.08,
        recency_window_minutes: float = 60.0,
    ) -> None:
        self.w_weakness = weight_weakness
        self.w_uncertainty = weight_uncertainty
        self.w_prereq = weight_prerequisite
        self.w_coverage = weight_coverage
        self.w_recency = weight_recency
        self.recency_window_minutes = recency_window_minutes

    def prioritize_candidate(
        self,
        concept: Concept,
        state: KnowledgeState,
        graph_store: GraphStore | None = None,
        now: datetime | None = None,
        student_states_lookup: dict[str, KnowledgeState] | None = None,
    ) -> AssessmentCandidate:
        now_dt = now or datetime.now(timezone.utc)
        reasons: list[str] = []

        # 1. Weakness Signal
        # A concept is truly weak only if evaluative evidence exists.
        has_evaluative = state.strong_evidence_count > 0 or state.medium_evidence_count > 0
        if has_evaluative and state.confidence >= 0.25:
            weakness = max(0.0, min(1.0, 1.0 - state.knowledge_score))
            if state.knowledge_score < 0.40:
                reasons.append(
                    f"Low knowledge score ({state.knowledge_score:.2f}): high weakness priority."
                )
            elif state.knowledge_score < 0.60:
                reasons.append(
                    f"Moderate knowledge score ({state.knowledge_score:.2f}): developing competence priority."
                )
        else:
            # Cold start or observational only: neutral prior, not falsely deemed weak
            weakness = 0.50
            if state.evidence_count == 0:
                reasons.append("Unassessed prior state: neutral weakness baseline.")

        # 2. Uncertainty Signal
        if state.mastery_status == MasteryStatus.UNCERTAIN or state.metadata.get("has_conflicting_evidence", False):
            uncertainty = 1.0
            reasons.append("Conflicting assessment evidence: requires diagnostic resolution.")
        else:
            uncertainty = max(0.0, min(1.0, 1.0 - state.confidence))
            if state.confidence < 0.30:
                reasons.append(
                    f"High estimation uncertainty (confidence {state.confidence:.2f}): diagnostic assessment needed."
                )

        # 3. Prerequisite Importance Signal
        prereq_score = 0.0
        if graph_store is not None:
            dependents = graph_store.get_dependents(concept.concept_id, transitive=False)
            if dependents:
                unmastered_count = 0
                dep_names: list[str] = []
                for dep in dependents:
                    dep_state = None
                    if student_states_lookup is not None:
                        dep_state = student_states_lookup.get(dep.concept_id)

                    # Dependent is unmastered if status is not MASTERED
                    if dep_state is None or dep_state.mastery_status != MasteryStatus.MASTERED:
                        unmastered_count += 1
                        dep_names.append(dep.canonical_name)

                if unmastered_count > 0:
                    prereq_score = min(1.0, 0.40 * unmastered_count)
                    dep_str = ", ".join(dep_names[:3])
                    reasons.append(
                        f"Foundational prerequisite for {unmastered_count} active/unmastered dependent concept(s) ({dep_str})."
                    )

        # 4. Coverage Need Signal
        total_eval_count = state.strong_evidence_count + state.medium_evidence_count
        if total_eval_count == 0:
            coverage_need = 1.0
            reasons.append("Curriculum coverage priority: concept has 0 completed assessments.")
        else:
            coverage_need = max(0.0, min(1.0, 1.0 / (1.0 + 0.5 * total_eval_count)))

        # 5. Recency Penalty Signal
        recency_penalty = 0.0
        if state.last_assessed_at is not None:
            # Ensure timezone-aware comparison
            last_dt = state.last_assessed_at
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            now_aware = now_dt if now_dt.tzinfo is not None else now_dt.replace(tzinfo=timezone.utc)

            delta_sec = max(0.0, (now_aware - last_dt).total_seconds())
            delta_min = delta_sec / 60.0
            if delta_min < self.recency_window_minutes:
                recency_penalty = max(0.0, min(1.0, 1.0 - (delta_min / self.recency_window_minutes)))
                reasons.append(
                    f"Recently assessed ({delta_min:.1f} min ago): recency penalty applied (-{recency_penalty * self.w_recency:.2f})."
                )

        # Compute combined priority
        raw_score = (
            (self.w_weakness * weakness)
            + (self.w_uncertainty * uncertainty)
            + (self.w_prereq * prereq_score)
            + (self.w_coverage * coverage_need)
            - (self.w_recency * recency_penalty)
        )
        priority_score = round(max(0.0, min(1.0, raw_score)), 4)

        return AssessmentCandidate(
            concept_id=concept.concept_id,
            canonical_name=concept.canonical_name,
            subject_id=concept.subject_id,
            priority_score=priority_score,
            knowledge_score=round(state.knowledge_score, 4),
            confidence=round(state.confidence, 4),
            mastery_status=state.mastery_status,
            uncertainty=round(uncertainty, 4),
            weakness=round(weakness, 4),
            prerequisite_relevance=round(prereq_score, 4),
            coverage_need=round(coverage_need, 4),
            recency_penalty=round(recency_penalty, 4),
            last_assessed_at=state.last_assessed_at,
            assessment_count=total_eval_count,
            reasons=reasons,
        )

    def rank_candidates(
        self,
        candidates: list[AssessmentCandidate],
    ) -> list[AssessmentCandidate]:
        """Sort candidates deterministically by priority score descending.

        Ties are broken deterministically by concept_id ascending.
        """
        return sorted(
            candidates,
            key=lambda c: (-c.priority_score, c.concept_id),
        )
