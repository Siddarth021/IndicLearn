"""Knowledge Estimator Protocol & Evidence-Based Estimator (Phase 6).

Implements the transparent, interpretable, evidence-based estimator:
- Explicit separation of knowledge score from confidence (uncertainty).
- Weak evidence (questions asked) does NOT dominate or imply zero knowledge.
- Strong evidence (assessment results) drives mastery determination.
- Conflicting evidence produces an UNCERTAIN state rather than oscillation.
- Recency weighting with configurable half-life.
- Fully traceable provenance preserved in KnowledgeState.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from indiclearn.config.settings import StudentModelSettings
from indiclearn.student_model.models import (
    EvidenceEvent,
    EvidenceStrength,
    KnowledgeEvidence,
    KnowledgeState,
    MasteryStatus,
)


@runtime_checkable
class KnowledgeEstimator(Protocol):
    """Protocol for student concept knowledge state estimators.

    Allows future replacement by Bayesian Knowledge Tracing (BKT), Item Response
    Theory (IRT), or deep knowledge tracing without changing application boundaries.
    """

    def create_initial_state(
        self,
        student_id: str,
        concept_id: str,
        subject_id: str = "signals_and_systems",
    ) -> KnowledgeState:
        """Create a default cold-start state for a student-concept pair."""
        ...

    def update(
        self,
        current_state: KnowledgeState | None,
        event: EvidenceEvent,
    ) -> KnowledgeState:
        """Apply a single evidence event and return the updated knowledge state."""
        ...

    def batch_update(
        self,
        current_state: KnowledgeState | None,
        events: list[EvidenceEvent],
    ) -> KnowledgeState:
        """Apply a sequence of evidence events in chronological order."""
        ...


class EvidenceBasedKnowledgeEstimator:
    """Interpretable, deterministic, evidence-based knowledge estimator.

    Heuristic Design:
    - Prior belief S_0 = 0.50 with low confidence C_0 = 0.10 (cold start).
    - Asymptotic confidence accumulation: C = 1 - (1 - C_0) * prod(1 - delta_i).
    - Recency-weighted score accumulation: S = sum(w_i * s_i) / sum(w_i).
    - Conflicting evidence detection for mixed assessment signals.
    - Mastered threshold requires both high score (>= 0.80) and high confidence (>= 0.60).
    """

    def __init__(self, settings: StudentModelSettings | None = None) -> None:
        cfg = settings or StudentModelSettings()
        self.prior_score: float = cfg.prior_knowledge_score
        self.prior_confidence: float = cfg.prior_confidence
        self.prior_weight: float = cfg.prior_weight

        self.weights: dict[str, float] = cfg.evidence_weights or {
            "strong": 1.00,
            "medium": 0.40,
            "weak": 0.10,
        }
        self.deltas: dict[str, float] = cfg.confidence_deltas or {
            "strong": 0.20,
            "medium": 0.08,
            "weak": 0.02,
        }
        self.half_life_days: float = cfg.recency_half_life_days

        thresholds = cfg.mastery_thresholds or {}
        self.mastered_min_score: float = thresholds.get("mastered_min_score", 0.80)
        self.mastered_min_confidence: float = thresholds.get("mastered_min_confidence", 0.60)
        self.developing_min_score: float = thresholds.get("developing_min_score", 0.50)
        self.weak_max_score: float = thresholds.get("weak_max_score", 0.40)
        self.weak_min_confidence: float = thresholds.get("weak_min_confidence", 0.40)

    def create_initial_state(
        self,
        student_id: str,
        concept_id: str,
        subject_id: str = "signals_and_systems",
    ) -> KnowledgeState:
        """Initialize cold-start state.

        Core Principle: Unknown does NOT mean unskilled. Score begins at prior belief
        with low confidence and UNKNOWN mastery status.
        """
        return KnowledgeState(
            student_id=student_id,
            concept_id=concept_id,
            subject_id=subject_id,
            knowledge_score=self.prior_score,
            confidence=self.prior_confidence,
            evidence_count=0,
            strong_evidence_count=0,
            medium_evidence_count=0,
            weak_evidence_count=0,
            last_interaction_at=None,
            last_assessed_at=None,
            mastery_status=MasteryStatus.UNKNOWN,
            evidence_history=[],
            model_version="v1_evidence_based",
        )

    def _compute_recency_decay(self, event_time: datetime, reference_time: datetime) -> float:
        """Compute exponential recency decay factor in (0.0, 1.0]."""
        if self.half_life_days <= 0:
            return 1.0
        delta_seconds = max(0.0, (reference_time - event_time).total_seconds())
        delta_days = delta_seconds / 86400.0
        decay_constant = math.log(2.0) / self.half_life_days
        return math.exp(-decay_constant * delta_days)

    def _format_evidence_summary(self, event: EvidenceEvent) -> str:
        """Generate human-readable audit string for evidence item."""
        type_str = event.event_type.value
        source_str = f" via {event.source}" if event.source != "unspecified" else ""
        return f"{type_str} (score={event.score:.2f}, strength={event.strength.value}{source_str})"

    def update(
        self,
        current_state: KnowledgeState | None,
        event: EvidenceEvent,
    ) -> KnowledgeState:
        """Apply a single evidence event."""
        return self.batch_update(current_state, [event])

    def batch_update(
        self,
        current_state: KnowledgeState | None,
        events: list[EvidenceEvent],
    ) -> KnowledgeState:
        """Apply a sequence of evidence events chronologically."""
        if not events and current_state is not None:
            return current_state

        if current_state is None:
            first_event = events[0]
            state = self.create_initial_state(
                student_id=first_event.student_id,
                concept_id=first_event.concept_id,
                subject_id=first_event.subject_id,
            )
        else:
            state = current_state

        # Combine existing history with new events
        all_evidence: list[KnowledgeEvidence] = list(state.evidence_history)
        for ev in events:
            summary = self._format_evidence_summary(ev)
            all_evidence.append(
                KnowledgeEvidence(
                    event_id=ev.event_id,
                    event_type=ev.event_type,
                    strength=ev.strength,
                    score=ev.score,
                    timestamp=ev.timestamp,
                    summary=summary,
                )
            )

        # Sort evidence chronologically
        all_evidence.sort(key=lambda x: x.timestamp)
        now_utc = datetime.now(timezone.utc)
        reference_time = all_evidence[-1].timestamp if all_evidence else now_utc

        # 1. Accumulate weighted knowledge score
        total_weight = self.prior_weight
        weighted_score_sum = self.prior_weight * self.prior_score

        # 2. Accumulate confidence via asymptotic saturation
        unconfidence_factor = 1.0 - self.prior_confidence

        strong_count = 0
        medium_count = 0
        weak_count = 0
        last_assessed_at: datetime | None = state.last_assessed_at
        last_interaction_at: datetime | None = state.last_interaction_at

        recent_strong_scores: list[float] = []

        for item in all_evidence:
            decay = self._compute_recency_decay(item.timestamp, reference_time)
            strength_key = item.strength.value.lower()
            base_w = self.weights.get(strength_key, 0.10)
            base_delta = self.deltas.get(strength_key, 0.02)

            effective_weight = base_w * decay
            total_weight += effective_weight
            weighted_score_sum += effective_weight * item.score

            effective_delta = min(0.99, base_delta * decay)
            unconfidence_factor *= (1.0 - effective_delta)

            if item.strength == EvidenceStrength.STRONG:
                strong_count += 1
                recent_strong_scores.append(item.score)
                if last_assessed_at is None or item.timestamp > last_assessed_at:
                    last_assessed_at = item.timestamp
            elif item.strength == EvidenceStrength.MEDIUM:
                medium_count += 1
            else:
                weak_count += 1

            if last_interaction_at is None or item.timestamp > last_interaction_at:
                last_interaction_at = item.timestamp

        new_score = round(weighted_score_sum / total_weight, 4)
        new_confidence = round(max(0.0, min(1.0, 1.0 - unconfidence_factor)), 4)

        # 3. Detect conflicting evidence in recent strong assessment history
        has_conflicting = False
        if len(recent_strong_scores) >= 2:
            # Check variance in last 5 strong scores
            window = recent_strong_scores[-5:]
            mean_strong = sum(window) / len(window)
            var_strong = sum((x - mean_strong) ** 2 for x in window) / len(window)
            if var_strong >= 0.15:
                has_conflicting = True

        # 4. Determine categorical mastery status
        has_evaluative_evidence = strong_count > 0 or medium_count > 0

        # Rule 1: No evaluative evidence -> UNKNOWN (weak evidence cannot classify mastery)
        if not has_evaluative_evidence or new_confidence < 0.25:
            mastery = MasteryStatus.UNKNOWN
        # Rule 2: Conflicting assessment evidence -> UNCERTAIN
        elif has_conflicting and new_confidence >= 0.40 and 0.40 <= new_score <= 0.65:
            mastery = MasteryStatus.UNCERTAIN
        # Rule 3: Verified Mastery -> MASTERED
        elif new_score >= self.mastered_min_score and new_confidence >= self.mastered_min_confidence:
            mastery = MasteryStatus.MASTERED
        # Rule 4: Developing Competency -> DEVELOPING
        elif new_score >= self.developing_min_score and new_confidence >= 0.40:
            mastery = MasteryStatus.DEVELOPING
        # Rule 5: Nascent/Emerging Competency -> EMERGING
        # Either assessed low score (< 0.50) OR early positive signal (>= 0.50) with low confidence (< 0.40)
        else:
            mastery = MasteryStatus.EMERGING

        return KnowledgeState(
            student_id=state.student_id,
            concept_id=state.concept_id,
            subject_id=state.subject_id,
            knowledge_score=new_score,
            confidence=new_confidence,
            evidence_count=len(all_evidence),
            strong_evidence_count=strong_count,
            medium_evidence_count=medium_count,
            weak_evidence_count=weak_count,
            last_interaction_at=last_interaction_at,
            last_assessed_at=last_assessed_at,
            mastery_status=mastery,
            evidence_history=all_evidence,
            model_version=state.model_version,
            metadata={"has_conflicting_evidence": has_conflicting},
        )
