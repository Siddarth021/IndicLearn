"""Recommendation Service Layer (Phase 7B).

Coordinates personalized recommendation planning:
1. Candidate generation from student knowledge state & academic graph
2. Multi-signal heuristic prioritization
3. Pedagogical action selection (review, practice, assessment)
4. Auditable explanation formatting
5. Assessment delegation to AdaptiveAssessmentService
6. Recommendation history recording
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from indiclearn.assessment.models import AssessmentSelectionExplanation
from indiclearn.assessment.service import AdaptiveAssessmentService
from indiclearn.knowledge_graph.base import GraphStore
from indiclearn.recommendation.candidate import (
    HeuristicRecommendationCandidateGenerator,
    RecommendationCandidateGenerator,
)
from indiclearn.recommendation.explanation import RecommendationExplanationGenerator
from indiclearn.recommendation.models import (
    ActionType,
    Recommendation,
    RecommendationCandidate,
    RecommendationHistoryEntry,
)
from indiclearn.recommendation.repository import (
    InMemoryRecommendationRepository,
    RecommendationHistoryRepository,
)
from indiclearn.student_model.service import KnowledgeStateService

logger = logging.getLogger(__name__)


class RecommendationService:
    """Primary service coordinating personalized recommendations and remediation planning."""

    def __init__(
        self,
        knowledge_service: KnowledgeStateService,
        graph_store: GraphStore | None = None,
        candidate_generator: RecommendationCandidateGenerator | None = None,
        explanation_generator: RecommendationExplanationGenerator | None = None,
        history_repo: RecommendationHistoryRepository | None = None,
        assessment_service: AdaptiveAssessmentService | None = None,
        default_top_k: int = 3,
        default_language: str = "en",
    ) -> None:
        self.knowledge_service = knowledge_service
        self.graph_store = graph_store
        self.history_repo = history_repo or InMemoryRecommendationRepository()
        self.candidate_generator = candidate_generator or HeuristicRecommendationCandidateGenerator(
            knowledge_service=knowledge_service,
            graph_store=graph_store,
            history_repo=self.history_repo,
        )
        self.explanation_generator = (
            explanation_generator or RecommendationExplanationGenerator(default_language=default_language)
        )
        self.assessment_service = assessment_service
        self.default_top_k = default_top_k
        self.default_language = default_language

    def get_candidates(
        self,
        student_id: str,
        subject_id: str = "signals_and_systems",
        now: datetime | None = None,
    ) -> list[RecommendationCandidate]:
        """Generate and rank all recommendation candidates for a student."""
        return self.candidate_generator.generate_candidates(
            student_id=student_id,
            subject_id=subject_id,
            now=now,
        )

    def get_recommendations(
        self,
        student_id: str,
        subject_id: str = "signals_and_systems",
        top_k: int | None = None,
        auto_record: bool = False,
        now: datetime | None = None,
    ) -> list[Recommendation]:
        """Generate, rank, and format top-K personalized recommendations."""
        curr_time = now or datetime.now(timezone.utc)
        k = top_k or self.default_top_k

        candidates = self.get_candidates(student_id=student_id, subject_id=subject_id, now=curr_time)
        if not candidates:
            return []

        selected_candidates = candidates[:k]
        recommendations: list[Recommendation] = []

        for cand in selected_candidates:
            explanation = self.explanation_generator.generate_explanation(
                candidate=cand,
                action=cand.action_type,
                difficulty=cand.suggested_difficulty,
            )

            rec = Recommendation(
                student_id=student_id,
                subject_id=subject_id,
                concept_id=cand.concept_id,
                canonical_name=cand.canonical_name,
                action_type=cand.action_type,
                priority_score=cand.priority_score,
                reason_codes=cand.reason_codes,
                explanation=explanation,
                knowledge_score=cand.knowledge_score,
                confidence=cand.confidence,
                mastery_status=cand.mastery_status,
                prerequisite_context={
                    "unmastered_prerequisites": cand.unmastered_prerequisites,
                    "blocked_dependents": cand.blocked_dependents,
                },
                source_chunk_ids=cand.source_chunk_ids,
                source_available=cand.source_available,
                suggested_difficulty=cand.suggested_difficulty,
                created_at=curr_time,
                model_version="v1_heuristic_recommender",
                metadata={
                    "signals": {
                        "weakness": cand.weakness_score,
                        "uncertainty": cand.uncertainty_score,
                        "prerequisite_deficit": cand.prerequisite_deficit_score,
                        "downstream_blocker": cand.downstream_blocker_score,
                        "review_need": cand.review_need_score,
                        "assessment_gap": cand.assessment_gap_score,
                        "recent_action_penalty": cand.recent_action_penalty,
                    },
                    "reasons": cand.reasons,
                },
            )
            recommendations.append(rec)

            if auto_record:
                self.record_recommendation(rec)

        return recommendations

    def record_recommendation(self, recommendation: Recommendation) -> None:
        """Record recommendation in student history to apply recent-action damping."""
        entry = RecommendationHistoryEntry(
            student_id=recommendation.student_id,
            subject_id=recommendation.subject_id,
            concept_id=recommendation.concept_id,
            action_type=recommendation.action_type,
            timestamp=recommendation.created_at,
            metadata={"priority_score": recommendation.priority_score},
        )
        self.history_repo.record_recommendation(entry)

    def delegate_assessment(
        self,
        recommendation: Recommendation,
        requested_language: str | None = None,
        now: datetime | None = None,
    ) -> AssessmentSelectionExplanation | None:
        """Delegate assessment selection to AdaptiveAssessmentService without duplicating logic."""
        if self.assessment_service is None:
            logger.warning("AssessmentService not configured; cannot delegate assessment.")
            return None

        if recommendation.action_type != ActionType.TAKE_ASSESSMENT:
            logger.info("Recommendation action is '%s'; delegation applies to TAKE_ASSESSMENT.", recommendation.action_type)

        lang = requested_language or self.default_language

        # Find matching items for the recommended concept
        all_items = self.assessment_service.item_repository.list_items(
            concept_id=recommendation.concept_id,
            subject_id=recommendation.subject_id,
        )
        if not all_items:
            logger.info("No assessment items found in bank for concept '%s'.", recommendation.concept_id)
            return None

        # Build an AssessmentCandidate from recommendation state to reuse selector
        from indiclearn.assessment.models import AssessmentCandidate as AssCandidate

        ass_cand = AssCandidate(
            concept_id=recommendation.concept_id,
            canonical_name=recommendation.canonical_name,
            subject_id=recommendation.subject_id,
            knowledge_score=recommendation.knowledge_score,
            confidence=recommendation.confidence,
            mastery_status=recommendation.mastery_status,
            priority_score=recommendation.priority_score,
            uncertainty=round(1.0 - recommendation.confidence, 4),
            weakness=round(1.0 - recommendation.knowledge_score, 4),
            prerequisite_relevance=0.0,
            coverage_need=0.5,
            recency_penalty=0.0,
        )

        _item, explanation = self.assessment_service.item_selector.select_item(
            candidate=ass_cand,
            available_items=all_items,
            requested_language=lang,
            seen_item_ids=set(),
        )
        return explanation
