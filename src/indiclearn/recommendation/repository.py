"""Recommendation History and Tracking Repositories (Phase 7B).

Provides interfaces and in-memory persistence for tracking recommendation history,
enabling recent action penalties and repeat-recommendation damping.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from indiclearn.recommendation.models import RecommendationHistoryEntry


@runtime_checkable
class RecommendationHistoryRepository(Protocol):
    """Protocol for recording and querying recommendation exposure history."""

    def record_recommendation(self, entry: RecommendationHistoryEntry) -> None:
        """Log a recommendation presented to or accepted by the student."""
        ...

    def get_history(
        self,
        student_id: str,
        subject_id: str | None = None,
    ) -> list[RecommendationHistoryEntry]:
        """Retrieve full recommendation history for a student."""
        ...

    def get_recent_entries(
        self,
        student_id: str,
        subject_id: str | None = None,
        limit: int = 5,
    ) -> list[RecommendationHistoryEntry]:
        """Retrieve most recent recommendation entries for a student."""
        ...


class InMemoryRecommendationRepository:
    """In-memory implementation of RecommendationHistoryRepository."""

    def __init__(self) -> None:
        # Keyed by student_id -> list of entries in chronological order
        self._history: dict[str, list[RecommendationHistoryEntry]] = {}

    def record_recommendation(self, entry: RecommendationHistoryEntry) -> None:
        """Append entry to student's history."""
        self._history.setdefault(entry.student_id, []).append(entry)

    def get_history(
        self,
        student_id: str,
        subject_id: str | None = None,
    ) -> list[RecommendationHistoryEntry]:
        """Retrieve full recommendation history."""
        entries = self._history.get(student_id, [])
        if subject_id is not None:
            entries = [e for e in entries if e.subject_id == subject_id]
        return list(entries)

    def get_recent_entries(
        self,
        student_id: str,
        subject_id: str | None = None,
        limit: int = 5,
    ) -> list[RecommendationHistoryEntry]:
        """Retrieve most recent entries up to limit."""
        entries = self.get_history(student_id, subject_id)
        return entries[-limit:] if limit > 0 else []

    def clear(self) -> None:
        """Clear all stored recommendation history."""
        self._history.clear()
