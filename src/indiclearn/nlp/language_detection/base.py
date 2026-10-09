"""Protocol for Language and Script Identification (Phase 4)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from indiclearn.nlp.models import LanguageDetectionResult


@runtime_checkable
class LanguageDetector(Protocol):
    """Protocol for language and script identification backends."""

    def detect(self, text: str) -> LanguageDetectionResult:
        """Analyze text and determine primary language, script, and distribution."""
        ...
