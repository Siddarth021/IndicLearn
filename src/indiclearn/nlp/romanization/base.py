"""Protocol for Romanized Telugu Normalization (Phase 4)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from indiclearn.nlp.models import NormalizationResult


@runtime_checkable
class RomanizationNormalizer(Protocol):
    """Protocol for Romanized Indic text normalization."""

    def normalize(self, text: str) -> NormalizationResult:
        """Normalize Romanized Telugu text into canonical Romanized forms with alternative candidates."""
        ...
