"""IndicLearn Adaptive Assessment Package (Phase 7A).

Provides an explainable, multi-signal adaptive assessment core grounded in academic course materials.
"""

from __future__ import annotations

from indiclearn.assessment.candidate import (
    AssessmentCandidateGenerator,
    HeuristicCandidateGenerator,
)
from indiclearn.assessment.items import evaluate_response
from indiclearn.assessment.models import (
    AssessmentCandidate,
    AssessmentItem,
    AssessmentResponse,
    AssessmentSelectionExplanation,
    AssessmentSession,
    ItemDifficulty,
    QuestionType,
)
from indiclearn.assessment.prioritizer import (
    CandidatePrioritizer,
    MultiSignalCandidatePrioritizer,
)
from indiclearn.assessment.repository import (
    AssessmentItemRepository,
    AssessmentSessionRepository,
    InMemoryAssessmentItemRepository,
    InMemoryAssessmentSessionRepository,
)
from indiclearn.assessment.selector import (
    AssessmentItemSelector,
    DeterministicItemSelector,
)
from indiclearn.assessment.service import AdaptiveAssessmentService
from indiclearn.assessment.session import AssessmentSessionManager

__all__ = [
    "AdaptiveAssessmentService",
    "AssessmentCandidate",
    "AssessmentCandidateGenerator",
    "AssessmentItem",
    "AssessmentItemRepository",
    "AssessmentItemSelector",
    "AssessmentResponse",
    "AssessmentSelectionExplanation",
    "AssessmentSession",
    "AssessmentSessionManager",
    "AssessmentSessionRepository",
    "CandidatePrioritizer",
    "DeterministicItemSelector",
    "HeuristicCandidateGenerator",
    "InMemoryAssessmentItemRepository",
    "InMemoryAssessmentSessionRepository",
    "ItemDifficulty",
    "MultiSignalCandidatePrioritizer",
    "QuestionType",
    "evaluate_response",
]
