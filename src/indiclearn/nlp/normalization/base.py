"""Protocol for general Query Normalization (Phase 4)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from indiclearn.nlp.models import Language, NormalizationResult


@runtime_checkable
class TextNormalizer(Protocol):
    """Protocol for general text normalization."""

    def normalize(
        self,
        text: str,
        primary_language: Language | None = None,
    ) -> NormalizationResult:
        """Normalize student query text."""
        ...
