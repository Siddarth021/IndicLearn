"""Recommendation Prioritizer (Phase 7B).

Computes multi-signal priority scores for recommendation candidates using
interpretable, configurable pedagogical heuristics.

Core Principle:
Does NOT use simplistic "lowest score -> recommend".
Evaluates seven interpretable signals:
1. Weakness: Evaluated difficulty based on strong assessments.
2. Uncertainty: Lack of confidence or conflicting assessment outcomes.
3. Prerequisite Deficit: Unmastered foundational prerequisites blocking this concept.
4. Downstream Blocker: Number of downstream concepts in the curriculum blocked by this concept.
5. Review Need: Time elapsed since last study or assessment interaction.
6. Assessment Gap: Absence of strong evaluative evidence.
7. Recent Action Penalty: Damping recently recommended or acted-upon concepts.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from indiclearn.recommendation.models import (
    ReasonCode,
    RecommendationCandidate,
)
from indiclearn.student_model.models import KnowledgeState, MasteryStatus


@runtime_checkable
class RecommendationPrioritizer(Protocol):
    """Protocol for scoring and generating reasons for recommendation candidates."""

    def score_candidate(
        self,
        concept_id: str,
        canonical_name: str,
        state: KnowledgeState,
        unmastered_prereq_names: list[str],
        blocked_dependent_names: list[str],
        source_chunk_ids: list[str],
        source_available: bool,
        last_action_at: datetime | None = None,
        now: datetime | None = None,
    ) -> RecommendationCandidate:
        """Compute candidate priority score, reason codes, and preliminary action."""
        ...


class MultiSignalRecommendationPrioritizer:
    """Configurable multi-signal prioritization policy.

    Note: All default weights are PROPOSED INITIAL HEURISTICS. They are not
    claimed as psychometrically or cognitively optimal parameters.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        mastered_damping_factor: float = 0.35,
        recent_decay_minutes: float = 30.0,
        review_half_life_hours: float = 72.0,
    ) -> None:
        default_weights = {
            "weakness": 0.35,
            "uncertainty": 0.15,
            "prerequisite_deficit": 0.15,
            "downstream_blocker": 0.15,
            "review_need": 0.10,
            "assessment_gap": 0.10,
            "recent_action_penalty": 0.25,
        }
        self.weights = {**default_weights, **(weights or {})}
        self.mastered_damping_factor = mastered_damping_factor
        self.recent_decay_minutes = recent_decay_minutes
        self.review_half_life_hours = review_half_life_hours

    def compute_weakness(self, state: KnowledgeState) -> float:
        """Calculate weakness score in [0.0, 1.0].

        Crucial distinction: If confidence is low and strong evidence is zero,
        lack of assessment does not equate to weakness (defaults to 0.0).
        For evaluated concepts with difficulty, score is scaled by confidence.
        """
        if state.confidence >= 0.30 or state.strong_evidence_count > 0:
            raw_diff = max(0.0, min(1.0, 1.0 - state.knowledge_score))
            # Weight demonstrated difficulty by evidence confidence
            return round(raw_diff * (0.5 + 0.5 * state.confidence), 4)
        return 0.0

    def compute_uncertainty(self, state: KnowledgeState) -> float:
        """Calculate uncertainty score in [0.0, 1.0]."""
        if state.mastery_status == MasteryStatus.UNCERTAIN:
            return 1.0
        if state.evidence_count == 0:
            # Cold-start: nascent uncertainty without overwhelming active failure
            return round(0.50 * (1.0 - state.confidence), 4)
        return max(0.0, min(1.0, 1.0 - state.confidence))

    def compute_prerequisite_deficit(self, unmastered_prereqs: list[str]) -> float:
        """Calculate deficit caused by unmastered upstream prerequisites."""
        if not unmastered_prereqs:
            return 0.0
        return round(min(1.0, len(unmastered_prereqs) / 2.0), 4)

    def compute_downstream_blocker(
        self,
        state: KnowledgeState,
        blocked_dependents: list[str],
    ) -> float:
        """Calculate curriculum blockage caused by this concept if not mastered."""
        if state.mastery_status == MasteryStatus.MASTERED:
            return 0.0
        if not blocked_dependents:
            return 0.0
        return round(min(1.0, len(blocked_dependents) / 3.0), 4)

    def compute_review_need(
        self,
        state: KnowledgeState,
        now: datetime,
    ) -> float:
        """Calculate review need based on elapsed time since last interaction."""
        if state.evidence_count == 0:
            # Unintroduced concepts do not have "neglected review"
            return 0.0

        ref_time = state.last_interaction_at or state.last_assessed_at
        if ref_time is None:
            return 0.5

        if ref_time.tzinfo is None:
            ref_time = ref_time.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        delta_hours = max(0.0, (now - ref_time).total_seconds() / 3600.0)
        decay = 1.0 - math.exp(-delta_hours / self.review_half_life_hours)

        if state.mastery_status == MasteryStatus.MASTERED:
            # Spaced review for mastered concepts grows slower
            return min(1.0, 0.5 * decay)
        return min(1.0, decay)

    def compute_assessment_gap(self, state: KnowledgeState) -> float:
        """Calculate assessment gap based on evaluative evidence volume."""
        if state.strong_evidence_count == 0:
            if state.weak_evidence_count > 0:
                return 0.8  # Student interacted but lacks evaluation
            return 0.3  # Cold start
        if state.strong_evidence_count == 1:
            return 0.4
        return 0.0

    def compute_recent_action_penalty(
        self,
        last_action_at: datetime | None,
        now: datetime,
    ) -> float:
        """Calculate penalty if this concept was acted upon recently."""
        if last_action_at is None:
            return 0.0

        if last_action_at.tzinfo is None:
            last_action_at = last_action_at.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        delta_minutes = max(0.0, (now - last_action_at).total_seconds() / 60.0)
        return math.exp(-delta_minutes / self.recent_decay_minutes)

    def score_candidate(
        self,
        concept_id: str,
        canonical_name: str,
        state: KnowledgeState,
        unmastered_prereq_names: list[str],
        blocked_dependent_names: list[str],
        source_chunk_ids: list[str],
        source_available: bool,
        last_action_at: datetime | None = None,
        now: datetime | None = None,
    ) -> RecommendationCandidate:
        """Evaluate signals, compute priority, and construct candidate."""
        curr_time = now or datetime.now(timezone.utc)

        s_weakness = self.compute_weakness(state)
        s_uncertainty = self.compute_uncertainty(state)
        s_prereq = self.compute_prerequisite_deficit(unmastered_prereq_names)
        s_blocker = self.compute_downstream_blocker(state, blocked_dependent_names)
        s_review = self.compute_review_need(state, curr_time)
        s_gap = self.compute_assessment_gap(state)
        s_recent = self.compute_recent_action_penalty(last_action_at, curr_time)

        # Weighted composite score
        w = self.weights
        raw_priority = (
            w.get("weakness", 0.30) * s_weakness
            + w.get("uncertainty", 0.20) * s_uncertainty
            + w.get("prerequisite_deficit", 0.15) * s_prereq
            + w.get("downstream_blocker", 0.15) * s_blocker
            + w.get("review_need", 0.10) * s_review
            + w.get("assessment_gap", 0.10) * s_gap
            - w.get("recent_action_penalty", 0.25) * s_recent
        )

        # Damping for mastered concepts
        if state.mastery_status == MasteryStatus.MASTERED:
            raw_priority *= self.mastered_damping_factor

        priority = max(0.0, min(1.0, round(raw_priority, 4)))

        # Reason codes and human-readable justifications
        reason_codes: list[ReasonCode] = []
        reasons: list[str] = []

        if s_weakness >= 0.50 and (state.confidence >= 0.30 or state.strong_evidence_count > 0):
            reason_codes.append(ReasonCode.LOW_KNOWLEDGE)
            reasons.append(
                f"Assessed knowledge score ({state.knowledge_score:.2f}) indicates difficulty requiring remediation."
            )

        if state.mastery_status == MasteryStatus.UNCERTAIN:
            reason_codes.append(ReasonCode.UNCERTAIN)
            reasons.append(
                "Assessment history exhibits conflicting performance signals; diagnostic assessment is recommended."
            )
        elif state.confidence < 0.30 and state.strong_evidence_count == 0:
            reason_codes.append(ReasonCode.LOW_CONFIDENCE)
            reasons.append(
                f"Initial knowledge estimate ({state.knowledge_score:.2f}) has low confidence ({state.confidence:.2f}); initial study or diagnostic check needed."
            )

        if unmastered_prereq_names:
            reason_codes.append(ReasonCode.PREREQUISITE_DEFICIT)
            reasons.append(
                f"Has {len(unmastered_prereq_names)} unmastered foundational prerequisite(s): {', '.join(unmastered_prereq_names)}."
            )

        if blocked_dependent_names and state.mastery_status != MasteryStatus.MASTERED:
            reason_codes.append(ReasonCode.DOWNSTREAM_BLOCKER)
            reasons.append(
                f"Prerequisite blocking {len(blocked_dependent_names)} downstream concept(s): {', '.join(blocked_dependent_names)}."
            )

        if s_review >= 0.70:
            reason_codes.append(ReasonCode.NEGLECTED_REVIEW)
            reasons.append("Concept has not been reviewed or assessed recently.")

        if s_gap >= 0.50 and state.strong_evidence_count == 0:
            reason_codes.append(ReasonCode.ASSESSMENT_GAP)
            reasons.append("No validated assessment evidence recorded for this concept.")

        if state.mastery_status == MasteryStatus.MASTERED:
            reason_codes.append(ReasonCode.MASTERED_SPACED_REVIEW)
            reasons.append("Concept is mastered; eligible for periodic spaced retention check.")

        # Source availability reason code
        if source_available:
            reason_codes.append(ReasonCode.SOURCE_AVAILABLE)
        else:
            reason_codes.append(ReasonCode.SOURCE_UNAVAILABLE)
            reasons.append(
                "Source material for this concept is not currently indexed in the academic curriculum."
            )

        return RecommendationCandidate(
            concept_id=concept_id,
            canonical_name=canonical_name,
            subject_id=state.subject_id,
            knowledge_score=state.knowledge_score,
            confidence=state.confidence,
            mastery_status=state.mastery_status,
            weakness_score=round(s_weakness, 4),
            uncertainty_score=round(s_uncertainty, 4),
            prerequisite_deficit_score=round(s_prereq, 4),
            downstream_blocker_score=round(s_blocker, 4),
            review_need_score=round(s_review, 4),
            assessment_gap_score=round(s_gap, 4),
            recent_action_penalty=round(s_recent, 4),
            priority_score=priority,
            reason_codes=reason_codes,
            unmastered_prerequisites=unmastered_prereq_names,
            blocked_dependents=blocked_dependent_names,
            source_chunk_ids=source_chunk_ids,
            source_available=source_available,
            reasons=reasons,
        )
