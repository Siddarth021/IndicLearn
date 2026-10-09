"""Domain models for Phase 7A Adaptive Assessment Core.

Architectural Principles:
- Academic Knowledge Graph (canonical concepts, relations) remains strictly separated and read-only.
- Student Knowledge Model (KnowledgeState, EvidenceEvent) is read-only for candidate scoring;
  updates occur strictly by emitting EvidenceEvent items via KnowledgeStateService.
- Assessment items must preserve source chunk provenance linking to student academic materials.
- Every candidate concept and assessment item selection provides explicit, interpretable justifications.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from indiclearn.student_model.models import MasteryStatus


class QuestionType(str, Enum):
    """Supported assessment question types for Phase 7A."""

    MCQ = "MCQ"
    TRUE_FALSE = "TRUE_FALSE"
    SHORT_ANSWER = "SHORT_ANSWER"


class ItemDifficulty(str, Enum):
    """Categorical item difficulty metadata (heuristic, uncalibrated)."""

    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"


class AssessmentItem(BaseModel):
    """An academic assessment item grounded in source course materials."""

    item_id: str = Field(description="Deterministic or unique item ID")
    concept_id: str = Field(description="Target canonical concept ID from the academic graph")
    subject_id: str = Field(default="signals_and_systems", description="Scoping subject ID")
    question: str = Field(description="The question prompt text")
    question_type: QuestionType = Field(default=QuestionType.MCQ, description="Assessment format")
    difficulty: ItemDifficulty = Field(default=ItemDifficulty.MEDIUM, description="Heuristic difficulty tier")
    language: str = Field(default="en", description="Question language variety (en, te, te-Latn, te-en)")
    options: list[str] = Field(
        default_factory=list,
        description="Selectable option list for MCQ / True-False items",
    )
    correct_answer: str = Field(description="Correct choice identifier or exact canonical answer string")
    explanation: str = Field(description="Pedagogical explanation of why the correct answer is valid")
    source_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Source chunk provenance IDs grounding the question in academic materials",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary extension metadata")


class AssessmentCandidate(BaseModel):
    """A concept evaluated for assessment eligibility and prioritization."""

    concept_id: str = Field(description="Canonical concept ID")
    canonical_name: str = Field(description="Display canonical name of the concept")
    subject_id: str = Field(default="signals_and_systems", description="Subject identifier")
    priority_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Computed selection priority in [0.0, 1.0] (higher = more urgent to assess)",
    )
    knowledge_score: float = Field(ge=0.0, le=1.0, description="Current student knowledge estimate")
    confidence: float = Field(ge=0.0, le=1.0, description="Current student estimation confidence")
    mastery_status: MasteryStatus = Field(description="Categorical student mastery status")
    uncertainty: float = Field(ge=0.0, le=1.0, description="Normalized uncertainty signal")
    weakness: float = Field(ge=0.0, le=1.0, description="Normalized weakness signal")
    prerequisite_relevance: float = Field(
        ge=0.0,
        le=1.0,
        description="Relevance boost based on dependent concepts in the academic graph",
    )
    coverage_need: float = Field(ge=0.0, le=1.0, description="Priority signal for unassessed/rarely assessed concepts")
    recency_penalty: float = Field(ge=0.0, le=1.0, description="Penalty signal for concepts assessed very recently")
    last_assessed_at: datetime | None = Field(default=None, description="Timestamp of most recent assessment")
    assessment_count: int = Field(default=0, ge=0, description="Number of strong/medium assessments completed")
    reasons: list[str] = Field(
        default_factory=list,
        description="Interpretable justifications explaining why this priority score was assigned",
    )


class AssessmentResponse(BaseModel):
    """A student's answer to an assessment item within an active session."""

    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = Field(description="ID of the assessment session")
    item_id: str = Field(description="ID of the answered assessment item")
    student_id: str = Field(description="Student identifier")
    concept_id: str = Field(description="Target concept ID")
    selected_answer: str = Field(description="Option or text provided by the student")
    correct: bool = Field(description="Deterministic evaluation outcome")
    feedback: str = Field(default="", description="Immediate pedagogical feedback or explanation")
    answered_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of response",
    )
    response_time_ms: int | None = Field(default=None, ge=0, description="Optional elapsed time in milliseconds")
    confidence_if_provided: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional student self-reported confidence in [0.0, 1.0]",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class AssessmentSession(BaseModel):
    """Stateful tracking for an adaptive assessment sequence."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    student_id: str = Field(description="Student identifier")
    subject_id: str = Field(default="signals_and_systems", description="Subject identifier")
    item_ids: list[str] = Field(
        default_factory=list,
        description="Sequence of assessment item IDs presented in this session",
    )
    current_index: int = Field(default=0, ge=0, description="Current item index")
    responses: list[AssessmentResponse] = Field(
        default_factory=list,
        description="Recorded student responses in this session",
    )
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = Field(default=None)
    is_active: bool = Field(default=True)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AssessmentSelectionExplanation(BaseModel):
    """Complete explainability record for an adaptive assessment decision."""

    candidate: AssessmentCandidate = Field(description="Prioritized concept candidate")
    item: AssessmentItem = Field(description="Selected assessment item")
    concept_reasons: list[str] = Field(
        default_factory=list,
        description="Why this concept was prioritized over other candidates",
    )
    difficulty_reasons: list[str] = Field(
        default_factory=list,
        description="Why this item difficulty tier was matched to student state",
    )
    item_reasons: list[str] = Field(
        default_factory=list,
        description="Why this specific item was chosen among matching items",
    )
