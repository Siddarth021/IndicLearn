"""Recommendation Explanation Generator (Phase 7B).

Generates transparent, auditable, evidence-based pedagogical explanations.

Core Principle:
Adheres strictly to the communicative principle:
    evidence -> uncertainty -> recommendation -> reason
Avoids blunt deficit labeling ("You are weak in X").
Uses supportive, evidence-grounded scientific formulations.
"""

from __future__ import annotations

from indiclearn.assessment.models import ItemDifficulty
from indiclearn.recommendation.models import ActionType, ReasonCode, RecommendationCandidate


class RecommendationExplanationGenerator:
    """Generates human-readable, auditable justification strings."""

    def __init__(self, default_language: str = "en") -> None:
        self.default_language = default_language

    def generate_explanation(
        self,
        candidate: RecommendationCandidate,
        action: ActionType,
        difficulty: ItemDifficulty | None = None,
    ) -> str:
        """Formulate transparent explanation for a recommended action."""
        name = candidate.canonical_name
        score = candidate.knowledge_score
        conf = candidate.confidence
        chunks = candidate.source_chunk_ids
        unmastered = candidate.unmastered_prerequisites
        blocked = candidate.blocked_dependents

        parts: list[str] = []

        # 1. Action header & Core recommendation
        if action == ActionType.REVIEW_PREREQUISITE:
            prereq_str = ", ".join(unmastered) if unmastered else "foundational topics"
            parts.append(
                f"Before advancing to '{name}', reviewing its foundational prerequisite(s) ({prereq_str}) "
                f"will establish the required conceptual background."
            )
        elif action == ActionType.PRACTICE_CONCEPT:
            diff_str = f" at {difficulty.value} difficulty" if difficulty else ""
            parts.append(
                f"Targeted problem-solving practice on '{name}'{diff_str} is recommended to reinforce conceptual application."
            )
        elif action == ActionType.TAKE_ASSESSMENT:
            parts.append(
                f"Diagnostic assessment on '{name}' is recommended to measure current understanding and clarify mastery state."
            )
        elif action == ActionType.REVISIT_SOURCE:
            parts.append(
                f"Periodic spaced review of '{name}' is suggested to ensure long-term retention of mastered principles."
            )
        else:  # REVIEW_CONCEPT
            parts.append(
                f"Reviewing the curriculum material for '{name}' is recommended to build foundational familiarity."
            )

        # 2. Evidence Context
        if ReasonCode.LOW_KNOWLEDGE in candidate.reason_codes:
            parts.append(
                f"Recent assessment evidence indicates lower demonstrated performance (score: {score:.2f}, confidence: {conf:.2f})."
            )
        elif ReasonCode.UNCERTAIN in candidate.reason_codes:
            parts.append(
                "Prior assessments exhibit mixed performance outcomes, suggesting that key subconcepts require clarification."
            )
        elif ReasonCode.LOW_CONFIDENCE in candidate.reason_codes:
            parts.append(
                f"Current evidence volume is limited (confidence: {conf:.2f}), reflecting initial prior belief rather than confirmed weakness."
            )
        elif ReasonCode.MASTERED_SPACED_REVIEW in candidate.reason_codes:
            parts.append(
                f"High competency has been established (score: {score:.2f}, confidence: {conf:.2f}); spaced retrieval prevents knowledge decay."
            )

        # 3. Downstream impact
        if blocked and candidate.mastery_status.value != "MASTERED":
            blocked_str = ", ".join(blocked[:2])
            if len(blocked) > 2:
                blocked_str += f", and {len(blocked) - 2} other concept(s)"
            parts.append(
                f"Mastering '{name}' is pivotal as it serves as a prerequisite for downstream topics including {blocked_str}."
            )

        # 4. Source Grounding Provenance
        if candidate.source_available and chunks:
            chunk_refs = ", ".join(chunks[:3])
            parts.append(f"Grounded in uploaded curriculum material: [{chunk_refs}].")
        elif not candidate.source_available:
            parts.append(
                "Source material for this concept is not currently indexed in the academic curriculum."
            )

        return " ".join(parts)
