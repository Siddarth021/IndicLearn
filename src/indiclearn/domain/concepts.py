"""Concept and Knowledge Graph domain entities (Phase 3).

Strictly academic subject domain entities.
Separate from student knowledge states (see docs/DATA_MODEL.md).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ConceptType(str, Enum):
    """Categorization of academic concepts."""

    CONCEPT = "concept"
    SUBCONCEPT = "subconcept"
    PROPERTY = "property"
    METHOD = "method"
    FORMULA = "formula"
    EXAMPLE = "example"


class RelationType(str, Enum):
    """Controlled vocabulary for typed edges in the academic knowledge graph."""

    RELATED_TO = "RELATED_TO"
    PREREQUISITE_OF = "PREREQUISITE_OF"
    PART_OF = "PART_OF"
    SUBCONCEPT_OF = "SUBCONCEPT_OF"
    HAS_PROPERTY = "HAS_PROPERTY"
    USED_IN = "USED_IN"
    DERIVED_FROM = "DERIVED_FROM"
    CONTRASTS_WITH = "CONTRASTS_WITH"


class PrerequisiteNature(str, Enum):
    """Distinguishes explicit syllabus/text statements from heuristic inferences."""

    EXPLICIT_PREREQUISITE = "explicit_prerequisite"
    INFERRED_PREREQUISITE = "inferred_prerequisite"


class Concept(BaseModel):
    """A canonical academic concept within a specific subject."""

    concept_id: str = Field(description="Deterministic unique ID, e.g. cpt_signals_convolution")
    subject_id: str = Field(description="Scoping subject/course ID")
    canonical_name: str = Field(description="Display canonical name, e.g. 'Convolution'")
    concept_type: ConceptType = Field(default=ConceptType.CONCEPT)
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternative surface forms across English, Telugu script, and Romanized forms",
    )
    description: str | None = Field(
        default=None, description="Extracted or synthesized concept definition"
    )
    source_document_ids: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    sections: list[str] = Field(default_factory=list)
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Heuristic extraction confidence"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def has_alias(self, alias_query: str) -> bool:
        """Case-insensitive check if alias_query matches canonical name or any alias."""
        clean = alias_query.strip().lower()
        if self.canonical_name.strip().lower() == clean:
            return True
        return any(a.strip().lower() == clean for a in self.aliases)


class ConceptRelation(BaseModel):
    """A directed, typed relationship between two concepts with explicit provenance."""

    relation_id: str = Field(
        description="Deterministic ID, e.g. rel_<source>_<relation_type>_<target>"
    )
    subject_id: str = Field(description="Scoping subject/course ID")
    source_concept_id: str = Field(description="Source concept ID")
    target_concept_id: str = Field(description="Target concept ID")
    relation_type: RelationType = Field(description="Controlled relation type")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Heuristic confidence of relationship"
    )
    prerequisite_nature: PrerequisiteNature | None = Field(
        default=None,
        description="For PREREQUISITE_OF edges: explicit statement vs inferred heuristic",
    )
    source_chunk_ids: list[str] = Field(
        default_factory=list, description="Provenance: chunks substantiating this edge"
    )
    source_document_ids: list[str] = Field(default_factory=list)
    extraction_method: str = Field(
        default="heuristic", description="Method: 'heuristic', 'heading', 'grammar', 'llm'"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
