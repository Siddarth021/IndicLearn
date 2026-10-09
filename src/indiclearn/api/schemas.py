"""API Domain Schemas for Student Learning Dashboard & Knowledge Map (Phase 8).

Defines strongly-typed data transfer objects for the client-server boundary.

Core Principles:
- The frontend is a VIEW over backend models; it never computes scores or recommendations.
- Communicates 'Estimated knowledge' and 'Evidence confidence', avoiding dogmatic cognitive claims.
- Retains academic source chunk provenance and explicitly flags missing sources.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from indiclearn.assessment.models import ItemDifficulty, QuestionType
from indiclearn.recommendation.models import ActionType, ReasonCode
from indiclearn.student_model.models import MasteryStatus


class SubjectSummary(BaseModel):
    """Subject course overview."""

    subject_id: str
    name: str
    concept_count: int
    relation_count: int
    description: str = ""


class GraphNodeView(BaseModel):
    """Visual representation of a concept node with student knowledge overlay."""

    id: str = Field(description="Canonical concept ID")
    name: str = Field(description="Display canonical name")
    type: str = Field(description="Concept type (e.g. concept, property, method)")
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    source_chunk_ids: list[str] = Field(default_factory=list)

    # Student Knowledge Model Overlay (read-only view)
    knowledge_score: float = Field(
        ge=0.0, le=1.0, description="Estimated knowledge score (heuristic, uncalibrated)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Evidence confidence in estimate"
    )
    mastery_status: MasteryStatus = Field(description="Categorical mastery status")
    evidence_count: int = Field(default=0, description="Total evidence events recorded")
    strong_evidence_count: int = Field(default=0)
    weak_evidence_count: int = Field(default=0)
    last_interaction_at: datetime | None = None
    last_assessed_at: datetime | None = None


class GraphEdgeView(BaseModel):
    """Directed edge in the academic knowledge graph."""

    id: str
    source: str
    target: str
    relation_type: str
    label: str


class KnowledgeMapView(BaseModel):
    """Complete academic knowledge map with student overlay."""

    subject_id: str
    student_id: str
    nodes: list[GraphNodeView]
    edges: list[GraphEdgeView]


class EvidenceItemView(BaseModel):
    """Summarized historical evidence event for transparency panel."""

    event_id: str
    event_type: str
    strength: str
    score: float
    timestamp: datetime
    summary: str


class ConceptDetailView(BaseModel):
    """Granular concept view including academic, student, and recommendation data."""

    concept_id: str
    canonical_name: str
    concept_type: str
    description: str | None = None
    subject_id: str
    aliases: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    source_available: bool = True

    # Graph relations
    prerequisites: list[dict[str, str]] = Field(
        default_factory=list, description="Direct prerequisites: [{'id': ..., 'name': ...}]"
    )
    dependents: list[dict[str, str]] = Field(
        default_factory=list, description="Direct dependents: [{'id': ..., 'name': ...}]"
    )

    # Student State
    knowledge_score: float
    confidence: float
    mastery_status: MasteryStatus
    explanation_reason: str
    evidence_count: int
    strong_evidence_count: int
    weak_evidence_count: int
    has_conflicting_evidence: bool = False
    evidence_history: list[EvidenceItemView] = Field(default_factory=list)
    last_interaction_at: datetime | None = None
    last_assessed_at: datetime | None = None

    # Next Recommended Action
    recommended_action: ActionType | None = None
    recommendation_reasons: list[str] = Field(default_factory=list)
    recommendation_priority: float | None = None


class StudentDashboardOverview(BaseModel):
    """High-level learning analytics overview for student dashboard."""

    student_id: str
    subject_id: str
    total_concepts_tracked: int
    concepts_with_evidence: int
    concepts_assessed: int
    mastered_count: int
    developing_count: int
    emerging_count: int
    uncertain_count: int
    unknown_count: int
    mean_knowledge_score: float
    mean_confidence: float
    recent_activity_count: int


class RecommendationCardView(BaseModel):
    """Student-facing recommendation card."""

    recommendation_id: str
    student_id: str
    concept_id: str
    canonical_name: str
    action_type: ActionType
    priority_score: float
    reason_codes: list[ReasonCode]
    explanation: str
    knowledge_score: float
    confidence: float
    mastery_status: MasteryStatus
    source_chunk_ids: list[str]
    source_available: bool
    suggested_difficulty: ItemDifficulty | None = None


class ChunkDetailView(BaseModel):
    """Academic curriculum source chunk for inspection."""

    chunk_id: str
    document_id: str
    subject_id: str = "signals_and_systems"
    page_number: int | None = None
    section: str | None = None
    text: str
    language: str = "en"
    concept_ids: list[str] = Field(default_factory=list)


class StartAssessmentRequest(BaseModel):
    """Request payload to initiate an adaptive assessment session."""

    student_id: str
    subject_id: str = "signals_and_systems"
    concept_id: str | None = None
    requested_language: str = "en"


class SubmitResponseRequest(BaseModel):
    """Request payload for answering an assessment item."""

    session_id: str
    item_id: str
    selected_answer: str
    response_time_ms: int | None = None
    confidence_if_provided: float | None = None
    requested_language: str = "en"


class AssessmentItemView(BaseModel):
    """Assessment item displayed to the student."""

    item_id: str
    concept_id: str
    question: str
    question_type: QuestionType
    difficulty: ItemDifficulty
    language: str
    options: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)


class AssessmentSessionView(BaseModel):
    """Assessment session status view."""

    session_id: str
    student_id: str
    subject_id: str
    is_active: bool
    current_index: int
    total_answered: int
    current_item: AssessmentItemView | None = None
    concept_reasons: list[str] = Field(default_factory=list)


class AssessmentResultView(BaseModel):
    """Feedback and updated state after submitting an assessment answer."""

    session_id: str
    item_id: str
    selected_answer: str
    correct: bool
    feedback: str
    updated_knowledge_score: float
    updated_confidence: float
    updated_mastery_status: MasteryStatus
    next_item: AssessmentItemView | None = None
    session_completed: bool = False
