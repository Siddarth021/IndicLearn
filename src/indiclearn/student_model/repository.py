"""Knowledge State Repository Interfaces & In-Memory Implementation (Phase 6).

Provides persistence abstractions for student knowledge states and interaction events.
Maintains strict data isolation between students.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Protocol, runtime_checkable

from indiclearn.student_model.models import EvidenceEvent, KnowledgeState


@runtime_checkable
class KnowledgeStateRepository(Protocol):
    """Protocol for student knowledge state and evidence persistence.

    Allows plugging in SQLite, PostgreSQL, or DynamoDB backends in future phases.
    """

    def get_state(self, student_id: str, concept_id: str) -> KnowledgeState | None:
        """Retrieve estimated state for a specific Student × Concept pair."""
        ...

    def get_student_states(
        self,
        student_id: str,
        subject_id: str | None = None,
    ) -> list[KnowledgeState]:
        """Retrieve all concept states for a student."""
        ...

    def save_state(self, state: KnowledgeState) -> None:
        """Upsert a student concept knowledge state."""
        ...

    def add_event(self, event: EvidenceEvent) -> None:
        """Append an interaction or assessment evidence event."""
        ...

    def get_events(
        self,
        student_id: str,
        concept_id: str | None = None,
    ) -> list[EvidenceEvent]:
        """Retrieve interaction events for a student, optionally filtered by concept."""
        ...

    def delete_student_data(self, student_id: str) -> None:
        """Remove all states and events for a specific student."""
        ...

    def clear(self) -> None:
        """Clear all in-memory data (useful for testing)."""
        ...


class InMemoryKnowledgeStateRepository:
    """Thread-safe, in-memory implementation of KnowledgeStateRepository."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # Key: (student_id, concept_id) -> KnowledgeState
        self._states: dict[tuple[str, str], KnowledgeState] = {}
        # Key: student_id -> list[EvidenceEvent]
        self._events: dict[str, list[EvidenceEvent]] = defaultdict(list)

    def get_state(self, student_id: str, concept_id: str) -> KnowledgeState | None:
        with self._lock:
            state = self._states.get((student_id, concept_id))
            return state.model_copy(deep=True) if state else None

    def get_student_states(
        self,
        student_id: str,
        subject_id: str | None = None,
    ) -> list[KnowledgeState]:
        with self._lock:
            results = [
                s.model_copy(deep=True)
                for (s_id, _), s in self._states.items()
                if s_id == student_id and (subject_id is None or s.subject_id == subject_id)
            ]
            return results

    def save_state(self, state: KnowledgeState) -> None:
        with self._lock:
            self._states[(state.student_id, state.concept_id)] = state.model_copy(deep=True)

    def add_event(self, event: EvidenceEvent) -> None:
        with self._lock:
            self._events[event.student_id].append(event.model_copy(deep=True))

    def get_events(
        self,
        student_id: str,
        concept_id: str | None = None,
    ) -> list[EvidenceEvent]:
        with self._lock:
            evs = self._events.get(student_id, [])
            if concept_id:
                evs = [e for e in evs if e.concept_id == concept_id]
            return [e.model_copy(deep=True) for e in evs]

    def delete_student_data(self, student_id: str) -> None:
        with self._lock:
            to_delete = [k for k in self._states if k[0] == student_id]
            for k in to_delete:
                del self._states[k]
            if student_id in self._events:
                del self._events[student_id]

    def clear(self) -> None:
        with self._lock:
            self._states.clear()
            self._events.clear()
