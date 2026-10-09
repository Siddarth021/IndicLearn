"""Student Knowledge Modeling Domain Entities & Evidence Types (Phase 6).

Implements the explicit architectural separation:
Canonical Academic Knowledge Graph (Subject matter)
       vs
Student Knowledge State (Student × Concept dynamic estimates)

Core Principles:
- A student's question is NOT proof of ignorance (weak evidence only).
- Strong evidence (assessment results) dominates weak observational interactions.
- Uncertainty (confidence) is explicitly separated from knowledge score.
- Every state transition preserves complete evidence provenance.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MasteryStatus(str, Enum):
    """Categorical student mastery status for a concept.

    Note: These are heuristic, interpretable categories based on configurable
    thresholds; they are not claimed as calibrated cognitive probabilities.
    """

    UNKNOWN = "UNKNOWN"          # Insufficient evidence (cold-start / prior belief)
    EMERGING = "EMERGING"        # Early evidence of understanding; low/moderate confidence
    DEVELOPING = "DEVELOPING"    # Moderate evidence of understanding across assessments
    MASTERED = "MASTERED"        # Strong, consistent evidence with high confidence
    UNCERTAIN = "UNCERTAIN"      # Conflicting evidence or high variance in assessments


class EvidenceStrength(str, Enum):
    """Evidence hierarchy classification (Phase 6 Section 4)."""

    STRONG = "STRONG"    # Validated assessments, quiz answers, self-assessments
    MEDIUM = "MEDIUM"    # Evaluated tasks, demonstrated applications
    WEAK = "WEAK"        # Observational events (questions asked, views, explanations)


class EvidenceType(str, Enum):
    """Taxonomy of interaction and assessment events."""

    # Strong assessment evidence
    QUIZ_CORRECT = "QUIZ_CORRECT"
    QUIZ_INCORRECT = "QUIZ_INCORRECT"
    ASSESSMENT_RESULT = "ASSESSMENT_RESULT"
    SELF_ASSESSMENT = "SELF_ASSESSMENT"

    # Medium task evidence
    SUCCESSFUL_TASK = "SUCCESSFUL_TASK"
    UNSUCCESSFUL_TASK = "UNSUCCESSFUL_TASK"

    # Weak observational evidence
    QUESTION_ASKED = "QUESTION_ASKED"
    CONCEPT_VIEWED = "CONCEPT_VIEWED"
    EXPLANATION_REQUESTED = "EXPLANATION_REQUESTED"


def default_strength_for_type(event_type: EvidenceType) -> EvidenceStrength:
    """Return the default evidence strength for a given evidence type."""
    if event_type in {
        EvidenceType.QUIZ_CORRECT,
        EvidenceType.QUIZ_INCORRECT,
        EvidenceType.ASSESSMENT_RESULT,
        EvidenceType.SELF_ASSESSMENT,
    }:
        return EvidenceStrength.STRONG
    if event_type in {EvidenceType.SUCCESSFUL_TASK, EvidenceType.UNSUCCESSFUL_TASK}:
        return EvidenceStrength.MEDIUM
    return EvidenceStrength.WEAK


class EvidenceEvent(BaseModel):
    """An interaction or assessment event providing evidence for a concept.

    Preserves source provenance to explain why a student is estimated
    to be at a particular knowledge state.
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    student_id: str = Field(description="Identifier for the student")
    concept_id: str = Field(description="Canonical concept ID in the academic graph")
    subject_id: str = Field(default="signals_and_systems", description="Course/subject identifier")
    event_type: EvidenceType = Field(description="Type of evidence event")
    strength: EvidenceStrength = Field(description="Evidence hierarchy category")
    score: float = Field(
        ge=0.0, le=1.0, description="Observed performance or signal score in [0.0, 1.0]"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event creation timestamp in UTC",
    )
    source: str = Field(
        default="unspecified",
        description="Originating subsystem (e.g. 'quiz_service', 'rag_pipeline')",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Supplemental context (e.g. query text, quiz item id)",
    )


class KnowledgeEvidence(BaseModel):
    """Summarized evidence item preserved in KnowledgeState for provenance audit."""

    event_id: str
    event_type: EvidenceType
    strength: EvidenceStrength
    score: float
    timestamp: datetime
    summary: str


class KnowledgeState(BaseModel):
    """Estimated state for Student × Concept.

    Separates estimated knowledge score from confidence (uncertainty).
    Maintains a granular evidence log for scientific explainability.
    """

    student_id: str = Field(description="Identifier for the student")
    concept_id: str = Field(description="Canonical concept ID in the academic graph")
    subject_id: str = Field(default="signals_and_systems", description="Scoping subject ID")
    knowledge_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Estimated mastery score in [0.0, 1.0] (heuristic, uncalibrated)",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in estimate based on accumulated evidence volume and recency",
    )
    evidence_count: int = Field(default=0, ge=0)
    strong_evidence_count: int = Field(default=0, ge=0)
    medium_evidence_count: int = Field(default=0, ge=0)
    weak_evidence_count: int = Field(default=0, ge=0)
    last_interaction_at: datetime | None = None
    last_assessed_at: datetime | None = None
    mastery_status: MasteryStatus = Field(default=MasteryStatus.UNKNOWN)
    evidence_history: list[KnowledgeEvidence] = Field(default_factory=list)
    model_version: str = Field(default="v1_heuristic_evidence")
    metadata: dict[str, Any] = Field(default_factory=dict)


class StudentKnowledgeSummary(BaseModel):
    """Subject-level aggregate mastery summary for a student."""

    student_id: str
    subject_id: str
    total_concepts_tracked: int
    mastered_count: int
    developing_count: int
    emerging_count: int
    uncertain_count: int
    unknown_count: int
    mean_knowledge_score: float
    mean_confidence: float


class ConceptStateExplanation(BaseModel):
    """Structured explanation of why a student is in a specific knowledge state."""

    student_id: str
    concept_id: str
    knowledge_score: float
    confidence: float
    mastery_status: MasteryStatus
    reason: str
    evidence_summary: list[str]
    strong_evidence_count: int
    weak_evidence_count: int
    has_conflicting_evidence: bool


class PrerequisiteReadiness(BaseModel):
    """Prerequisite inspection result without knowledge state mutation."""

    student_id: str
    target_concept_id: str
    is_ready: bool
    prerequisite_states: list[dict[str, Any]]
    unmastered_prerequisites: list[str]
    recommendation: str
