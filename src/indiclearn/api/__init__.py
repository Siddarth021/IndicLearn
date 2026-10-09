"""IndicLearn API package (Phase 8)."""

from indiclearn.api.app import app, create_app
from indiclearn.api.schemas import (
    AssessmentItemView,
    AssessmentResultView,
    AssessmentSessionView,
    ChunkDetailView,
    ConceptDetailView,
    EvidenceItemView,
    GraphEdgeView,
    GraphNodeView,
    KnowledgeMapView,
    RecommendationCardView,
    StartAssessmentRequest,
    StudentDashboardOverview,
    SubjectSummary,
    SubmitResponseRequest,
)
from indiclearn.api.service import DashboardApplicationService

__all__ = [
    "AssessmentItemView",
    "AssessmentResultView",
    "AssessmentSessionView",
    "ChunkDetailView",
    "ConceptDetailView",
    "DashboardApplicationService",
    "EvidenceItemView",
    "GraphEdgeView",
    "GraphNodeView",
    "KnowledgeMapView",
    "RecommendationCardView",
    "StartAssessmentRequest",
    "StudentDashboardOverview",
    "SubjectSummary",
    "SubmitResponseRequest",
    "app",
    "create_app",
]
