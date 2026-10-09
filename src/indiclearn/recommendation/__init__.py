"""Personalized Recommendations and Remediation Planner (Phase 7B).

Public API exports for explainable, evidence-based recommendation planning.
"""

from indiclearn.recommendation.action_selector import ActionSelector, PedagogicalActionSelector
from indiclearn.recommendation.candidate import (
    HeuristicRecommendationCandidateGenerator,
    RecommendationCandidateGenerator,
)
from indiclearn.recommendation.explanation import RecommendationExplanationGenerator
from indiclearn.recommendation.models import (
    ActionType,
    ReasonCode,
    Recommendation,
    RecommendationCandidate,
    RecommendationHistoryEntry,
)
from indiclearn.recommendation.prioritizer import (
    MultiSignalRecommendationPrioritizer,
    RecommendationPrioritizer,
)
from indiclearn.recommendation.repository import (
    InMemoryRecommendationRepository,
    RecommendationHistoryRepository,
)
from indiclearn.recommendation.service import RecommendationService

__all__ = [
    "ActionSelector",
    "ActionType",
    "HeuristicRecommendationCandidateGenerator",
    "InMemoryRecommendationRepository",
    "MultiSignalRecommendationPrioritizer",
    "PedagogicalActionSelector",
    "ReasonCode",
    "Recommendation",
    "RecommendationCandidate",
    "RecommendationCandidateGenerator",
    "RecommendationExplanationGenerator",
    "RecommendationHistoryEntry",
    "RecommendationHistoryRepository",
    "RecommendationPrioritizer",
    "RecommendationService",
]
