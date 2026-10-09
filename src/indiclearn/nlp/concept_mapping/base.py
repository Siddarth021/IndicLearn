"""Protocol for Cross-Lingual Concept Mapping (Phase 4)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from indiclearn.nlp.models import ConceptMappingResult, TechnicalTerm


@runtime_checkable
class ConceptMapper(Protocol):
    """Protocol for mapping student queries across languages to canonical concept IDs."""

    def map_concept(
        self,
        query: str,
        subject_id: str = "signals_and_systems",
        top_k: int = 3,
        detected_terms: list[TechnicalTerm] | None = None,
    ) -> ConceptMappingResult:
        """Map query to candidate concepts and return structured ranking."""
        ...

    def map_query(
        self,
        query: str,
        subject_id: str = "signals_and_systems",
        top_k: int = 3,
        detected_terms: list[TechnicalTerm] | None = None,
    ) -> ConceptMappingResult:
        """Alias for map_concept."""
        ...
