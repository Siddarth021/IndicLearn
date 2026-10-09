"""Assessment item selector protocol and deterministic implementation (Phase 7A).

Selects an appropriate assessment item for a prioritized concept candidate,
enforcing:
1. Difficulty matching policy (aligned with student knowledge score and confidence)
2. Language preference matching (with clean fallback)
3. Repeat avoidance (filtering recently presented items)
4. Full source provenance preservation
5. Explicit, interpretable selection explanations
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from indiclearn.assessment.models import (
    AssessmentCandidate,
    AssessmentItem,
    AssessmentSelectionExplanation,
    ItemDifficulty,
)
from indiclearn.student_model.models import MasteryStatus

logger = logging.getLogger(__name__)


@runtime_checkable
class AssessmentItemSelector(Protocol):
    """Protocol for selecting assessment items for a prioritized candidate."""

    def select_item(
        self,
        candidate: AssessmentCandidate,
        available_items: list[AssessmentItem],
        requested_language: str | None = None,
        seen_item_ids: set[str] | None = None,
    ) -> tuple[AssessmentItem | None, AssessmentSelectionExplanation | None]:
        """Select the best matching item and provide full explainability."""
        ...


class DeterministicItemSelector:
    """Deterministic, policy-driven assessment item selector."""

    def __init__(self, default_language: str = "en") -> None:
        self.default_language = default_language

    def determine_target_difficulty(
        self,
        candidate: AssessmentCandidate,
    ) -> tuple[ItemDifficulty, str]:
        """Determine target item difficulty based on candidate knowledge state."""
        if (
            candidate.knowledge_score < 0.40
            or candidate.confidence < 0.30
            or candidate.mastery_status == MasteryStatus.EMERGING
        ):
            return (
                ItemDifficulty.EASY,
                f"Knowledge score ({candidate.knowledge_score:.2f}) or emerging status indicates need for foundational diagnostic assessment (EASY).",
            )
        elif candidate.knowledge_score >= 0.75 and candidate.confidence >= 0.50:
            return (
                ItemDifficulty.HARD,
                f"High knowledge score ({candidate.knowledge_score:.2f}) and established confidence ({candidate.confidence:.2f}) indicate readiness for advanced synthesis assessment (HARD).",
            )
        else:
            return (
                ItemDifficulty.MEDIUM,
                f"Developing competency (score {candidate.knowledge_score:.2f}, confidence {candidate.confidence:.2f}) indicates standard application-level assessment (MEDIUM).",
            )

    def select_item(
        self,
        candidate: AssessmentCandidate,
        available_items: list[AssessmentItem],
        requested_language: str | None = None,
        seen_item_ids: set[str] | None = None,
    ) -> tuple[AssessmentItem | None, AssessmentSelectionExplanation | None]:
        seen = seen_item_ids or set()
        target_lang = requested_language or self.default_language

        # 1. Filter items strictly belonging to target concept
        matching_items = [it for it in available_items if it.concept_id == candidate.concept_id]
        if not matching_items:
            return None, None

        # 2. Determine target difficulty
        target_diff, diff_reason = self.determine_target_difficulty(candidate)

        # 3. Score each matching item deterministically
        scored_items: list[tuple[float, AssessmentItem, list[str]]] = []

        for it in matching_items:
            score = 0.0
            reasons: list[str] = []

            # Unseen bonus (repeat avoidance)
            if it.item_id not in seen:
                score += 50.0
                reasons.append("Item has not been presented in the current session (repeat avoidance).")
            else:
                reasons.append("Item was previously seen in this session.")

            # Language match
            if it.language == target_lang:
                score += 30.0
                reasons.append(f"Exact language match for requested variety '{target_lang}'.")
            elif target_lang.startswith("te") and it.language.startswith("te"):
                score += 20.0
                reasons.append(f"Acceptable Telugu language family match ('{it.language}').")
            elif it.language == "en":
                score += 10.0
                reasons.append("Fallback to standard English course formulation.")

            # Difficulty match
            if it.difficulty == target_diff:
                score += 20.0
                reasons.append(f"Target difficulty tier matched ({target_diff.value}).")
            elif (
                (target_diff == ItemDifficulty.EASY and it.difficulty == ItemDifficulty.MEDIUM)
                or (target_diff == ItemDifficulty.HARD and it.difficulty == ItemDifficulty.MEDIUM)
                or (target_diff == ItemDifficulty.MEDIUM and it.difficulty != ItemDifficulty.MEDIUM)
            ):
                score += 10.0
                reasons.append(f"Adjacent difficulty tier used ({it.difficulty.value}).")

            # Source provenance note
            if it.source_chunk_ids:
                reasons.append(f"Grounded in academic source chunks: {', '.join(it.source_chunk_ids)}.")

            scored_items.append((score, it, reasons))

        # Sort deterministically: highest score first, tie-break by item_id ascending
        scored_items.sort(key=lambda x: (-x[0], x[1].item_id))

        _best_score, best_item, item_reasons = scored_items[0]

        explanation = AssessmentSelectionExplanation(
            candidate=candidate,
            item=best_item,
            concept_reasons=list(candidate.reasons),
            difficulty_reasons=[diff_reason],
            item_reasons=item_reasons,
        )

        return best_item, explanation
