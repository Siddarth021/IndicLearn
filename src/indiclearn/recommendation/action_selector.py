"""Pedagogical Action Selector (Phase 7B).

Maps student knowledge states, graph prerequisite relationships, and evidence
characteristics to appropriate pedagogical actions and difficulty tiers.

Core Principle:
Separates concept prioritization (urgency of attention) from action selection
(what specific learning activity should be performed).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from indiclearn.assessment.models import ItemDifficulty
from indiclearn.recommendation.models import ActionType, RecommendationCandidate
from indiclearn.student_model.models import MasteryStatus


@runtime_checkable
class ActionSelector(Protocol):
    """Protocol for selecting pedagogical action and difficulty tier."""

    def select_action(
        self,
        candidate: RecommendationCandidate,
    ) -> tuple[ActionType, ItemDifficulty | None]:
        """Determine pedagogical action and suggested difficulty."""
        ...


class PedagogicalActionSelector:
    """Deterministic, policy-driven action selector.

    Rules:
    1. Prerequisite Deficit: If foundational prerequisites are unmastered -> REVIEW_PREREQUISITE.
    2. Conflicting Evidence: If mastery_status is UNCERTAIN -> TAKE_ASSESSMENT (diagnostic).
    3. Mastered Spaced Review: If MASTERED with review need -> REVISIT_SOURCE.
    4. Evaluated Competency / Weakness:
       - High-confidence weakness (S < 0.40, C >= 0.50): PRACTICE_CONCEPT (EASY) or REVIEW_CONCEPT.
       - Developing competency (0.40 <= S < 0.75, C >= 0.40): PRACTICE_CONCEPT (MEDIUM).
    5. Nascent / Cold-Start (C < 0.30, 0 assessments): REVIEW_CONCEPT (initial study).
    """

    def __init__(
        self,
        weakness_threshold: float = 0.40,
        developing_threshold: float = 0.75,
        min_practice_confidence: float = 0.40,
    ) -> None:
        self.weakness_threshold = weakness_threshold
        self.developing_threshold = developing_threshold
        self.min_practice_confidence = min_practice_confidence

    def select_action(
        self,
        candidate: RecommendationCandidate,
    ) -> tuple[ActionType, ItemDifficulty | None]:
        """Select pedagogical action and difficulty for candidate."""
        # Rule 1: Prerequisite Deficit
        if candidate.unmastered_prerequisites:
            return ActionType.REVIEW_PREREQUISITE, None

        # Rule 2: Conflicting Evidence (UNCERTAIN)
        if candidate.mastery_status == MasteryStatus.UNCERTAIN:
            return ActionType.TAKE_ASSESSMENT, ItemDifficulty.MEDIUM

        # Rule 3: Mastered Spaced Review
        if candidate.mastery_status == MasteryStatus.MASTERED:
            return ActionType.REVISIT_SOURCE, None

        # Rule 4: Evaluated Weakness vs Developing
        if candidate.confidence >= self.min_practice_confidence:
            if candidate.knowledge_score < self.weakness_threshold:
                # Evaluated weakness: If source chunks exist, review then practice
                if candidate.source_available:
                    return ActionType.PRACTICE_CONCEPT, ItemDifficulty.EASY
                return ActionType.PRACTICE_CONCEPT, ItemDifficulty.EASY
            elif candidate.knowledge_score < self.developing_threshold:
                return ActionType.PRACTICE_CONCEPT, ItemDifficulty.MEDIUM
            else:
                return ActionType.PRACTICE_CONCEPT, ItemDifficulty.HARD

        # Rule 5: Cold-Start / Nascent Knowledge
        # If student has zero strong assessments and low confidence -> initial review
        return ActionType.REVIEW_CONCEPT, None
