"""Protocol for token-level Code-Mixing Analysis (Phase 4)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from indiclearn.nlp.models import CodeMixResult


@runtime_checkable
class CodeMixDetector(Protocol):
    """Protocol for token-level code-mix analysis."""

    def analyze(self, text: str) -> CodeMixResult:
        """Analyze text and segment it into language-tagged token spans with CMI."""
        ...

    def detect_spans(self, text: str) -> CodeMixResult:
        """Alias for analyze."""
        ...
