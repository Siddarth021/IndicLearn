"""Personalized Recommendation & Remediation Planner Domain Models (Phase 7B).

Defines the explicit domain models, action taxonomy, and reason codes for
explainable recommendation planning.

Core Principles:
- Evidence-based: Recommendations are derived strictly from observed evidence and graph relationships.
- Explainable: Every recommendation exposes structured reason codes and human-readable explanations.
- Grounded: Identifies source chunks when available; explicitly flags SOURCE_UNAVAILABLE when missing.
- Non-mutating: Recommendation planning reads knowledge states; it NEVER directly modifies scores.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from indiclearn.assessment.models import ItemDifficulty
from indiclearn.student_model.models import MasteryStatus


class ActionType(str, Enum):
    """Controlled taxonomy of pedagogical actions supported by the system."""

    REVIEW_CONCEPT = "REVIEW_CONCEPT"
    REVIEW_PREREQUISITE = "REVIEW_PREREQUISITE"
    PRACTICE_CONCEPT = "PRACTICE_CONCEPT"
    TAKE_ASSESSMENT = "TAKE_ASSESSMENT"
    REVISIT_SOURCE = "REVISIT_SOURCE"


class ReasonCode(str, Enum):
    """Machine-readable reason codes explaining why a recommendation was made."""

    LOW_KNOWLEDGE = "LOW_KNOWLEDGE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    UNCERTAIN = "UNCERTAIN"
    PREREQUISITE_DEFICIT = "PREREQUISITE_DEFICIT"
    DOWNSTREAM_BLOCKER = "DOWNSTREAM_BLOCKER"
    NEGLECTED_REVIEW = "NEGLECTED_REVIEW"
    ASSESSMENT_GAP = "ASSESSMENT_GAP"
    MASTERED_SPACED_REVIEW = "MASTERED_SPACED_REVIEW"
    SOURCE_AVAILABLE = "SOURCE_AVAILABLE"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"


class RecommendationCandidate(BaseModel):
    """Scored concept candidate before final ranking and top-K selection."""

    concept_id: str = Field(description="Canonical concept ID in the academic graph")
    canonical_name: str = Field(description="Display canonical name of the concept")
    subject_id: str = Field(default="signals_and_systems", description="Scoping subject ID")
    knowledge_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    mastery_status: MasteryStatus = Field(default=MasteryStatus.UNKNOWN)

    # Normalized component signals in [0.0, 1.0]
    weakness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty_score: float = Field(default=0.0, ge=0.0, le=1.0)
    prerequisite_deficit_score: float = Field(default=0.0, ge=0.0, le=1.0)
    downstream_blocker_score: float = Field(default=0.0, ge=0.0, le=1.0)
    review_need_score: float = Field(default=0.0, ge=0.0, le=1.0)
    assessment_gap_score: float = Field(default=0.0, ge=0.0, le=1.0)
    recent_action_penalty: float = Field(default=0.0, ge=0.0, le=1.0)

    # Composite priority in [0.0, 1.0]
    priority_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # Pedagogical decision
    action_type: ActionType = Field(default=ActionType.REVIEW_CONCEPT)
    reason_codes: list[ReasonCode] = Field(default_factory=list)

    # Graph and curriculum context
    unmastered_prerequisites: list[str] = Field(
        default_factory=list, description="IDs of direct prerequisites that lack mastery"
    )
    blocked_dependents: list[str] = Field(
        default_factory=list, description="IDs of downstream concepts blocked by this concept"
    )
    source_chunk_ids: list[str] = Field(
        default_factory=list, description="Academic source chunk IDs grounding this concept"
    )
    source_available: bool = Field(
        default=True, description="Whether academic material is present in the curriculum"
    )
    suggested_difficulty: ItemDifficulty | None = Field(
        default=None, description="Suggested difficulty tier for assessment or practice"
    )
    reasons: list[str] = Field(
        default_factory=list, description="Human-readable justification bullet points"
    )


class Recommendation(BaseModel):
    """Explainable personalized recommendation returned to the learner or instructor."""

    recommendation_id: str = Field(
        default_factory=lambda: f"rec_{uuid.uuid4().hex[:12]}",
        description="Unique recommendation identifier",
    )
    student_id: str = Field(description="Identifier for the student")
    subject_id: str = Field(default="signals_and_systems", description="Scoping subject ID")
    concept_id: str = Field(description="Canonical concept ID in academic graph")
    canonical_name: str = Field(description="Display canonical name")
    action_type: ActionType = Field(description="Recommended pedagogical action")
    priority_score: float = Field(
        ge=0.0, le=1.0, description="Normalized recommendation priority in [0.0, 1.0]"
    )
    reason_codes: list[ReasonCode] = Field(
        default_factory=list, description="Structured categorical reason codes"
    )
    explanation: str = Field(description="Transparent, human-readable justification")
    knowledge_score: float = Field(ge=0.0, le=1.0, description="Estimated knowledge score")
    confidence: float = Field(ge=0.0, le=1.0, description="Estimation confidence")
    mastery_status: MasteryStatus = Field(description="Current mastery status")
    prerequisite_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Details on upstream prerequisites or downstream blocked concepts",
    )
    source_chunk_ids: list[str] = Field(
        default_factory=list, description="Curriculum source chunks for review"
    )
    source_available: bool = Field(
        default=True, description="False if concept has no associated source material"
    )
    suggested_difficulty: ItemDifficulty | None = Field(
        default=None, description="Suggested difficulty for practice/assessment"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of generation",
    )
    model_version: str = Field(
        default="v1_heuristic_recommender",
        description="Recommender model identifier",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecommendationHistoryEntry(BaseModel):
    """Audit log entry tracking when a recommendation was presented or acted upon."""

    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    student_id: str
    subject_id: str = "signals_and_systems"
    concept_id: str
    action_type: ActionType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)
