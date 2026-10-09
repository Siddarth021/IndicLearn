"""Protocol for Technical Term Detection (Phase 4)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from indiclearn.nlp.models import TechnicalTerm


@runtime_checkable
class TechnicalTermDetector(Protocol):
    """Protocol for detecting academic engineering terminology in student queries."""

    def detect_terms(
        self,
        text: str,
        subject_id: str = "signals_and_systems",
    ) -> list[TechnicalTerm]:
        """Identify technical terms grounded in the academic knowledge graph."""
        ...
