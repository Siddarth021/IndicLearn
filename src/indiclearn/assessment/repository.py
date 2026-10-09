"""Repositories for assessment items and assessment sessions (Phase 7A).

Provides clean data access protocols and in-memory implementations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from indiclearn.assessment.models import (
    AssessmentItem,
    AssessmentSession,
    ItemDifficulty,
    QuestionType,
)


@runtime_checkable
class AssessmentItemRepository(Protocol):
    """Protocol for storing and retrieving assessment items."""

    def add_item(self, item: AssessmentItem) -> None:
        """Store an assessment item."""
        ...

    def get_item(self, item_id: str) -> AssessmentItem | None:
        """Retrieve an item by ID."""
        ...

    def list_items(
        self,
        subject_id: str | None = None,
        concept_id: str | None = None,
    ) -> list[AssessmentItem]:
        """List items matching optional subject and concept filters."""
        ...

    def count(self, subject_id: str | None = None) -> int:
        """Count total items in repository."""
        ...


@runtime_checkable
class AssessmentSessionRepository(Protocol):
    """Protocol for storing and retrieving assessment sessions."""

    def save_session(self, session: AssessmentSession) -> None:
        """Persist or update an assessment session."""
        ...

    def get_session(self, session_id: str) -> AssessmentSession | None:
        """Retrieve a session by unique ID."""
        ...

    def get_student_sessions(
        self,
        student_id: str,
        subject_id: str | None = None,
    ) -> list[AssessmentSession]:
        """Retrieve all sessions for a student."""
        ...


class InMemoryAssessmentItemRepository:
    """In-memory storage for assessment items with JSON loader support."""

    def __init__(self, items: list[AssessmentItem] | None = None) -> None:
        self._items: dict[str, AssessmentItem] = {}
        if items:
            for it in items:
                self.add_item(it)

    def add_item(self, item: AssessmentItem) -> None:
        self._items[item.item_id] = item

    def get_item(self, item_id: str) -> AssessmentItem | None:
        return self._items.get(item_id)

    def list_items(
        self,
        subject_id: str | None = None,
        concept_id: str | None = None,
    ) -> list[AssessmentItem]:
        results: list[AssessmentItem] = []
        for it in self._items.values():
            if subject_id and it.subject_id != subject_id:
                continue
            if concept_id and it.concept_id != concept_id:
                continue
            results.append(it)
        return results

    def count(self, subject_id: str | None = None) -> int:
        if subject_id is None:
            return len(self._items)
        return sum(1 for it in self._items.values() if it.subject_id == subject_id)

    @classmethod
    def from_json_file(cls, path: str | Path) -> InMemoryAssessmentItemRepository:
        """Load items from a structured JSON file."""
        file_path = Path(path)
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        raw_items: list[dict[str, Any]] = data if isinstance(data, list) else data.get("items", [])
        items: list[AssessmentItem] = []
        for raw in raw_items:
            items.append(
                AssessmentItem(
                    item_id=raw["item_id"],
                    concept_id=raw["concept_id"],
                    subject_id=raw.get("subject_id", "signals_and_systems"),
                    question=raw["question"],
                    question_type=QuestionType(raw.get("question_type", "MCQ")),
                    difficulty=ItemDifficulty(raw.get("difficulty", "MEDIUM")),
                    language=raw.get("language", "en"),
                    options=raw.get("options", []),
                    correct_answer=raw["correct_answer"],
                    explanation=raw.get("explanation", ""),
                    source_chunk_ids=raw.get("source_chunk_ids", []),
                    metadata=raw.get("metadata", {}),
                )
            )
        return cls(items)


class InMemoryAssessmentSessionRepository:
    """In-memory storage for assessment sessions with student isolation."""

    def __init__(self) -> None:
        self._sessions: dict[str, AssessmentSession] = {}

    def save_session(self, session: AssessmentSession) -> None:
        self._sessions[session.session_id] = session

    def get_session(self, session_id: str) -> AssessmentSession | None:
        return self._sessions.get(session_id)

    def get_student_sessions(
        self,
        student_id: str,
        subject_id: str | None = None,
    ) -> list[AssessmentSession]:
        results: list[AssessmentSession] = []
        for sess in self._sessions.values():
            if sess.student_id != student_id:
                continue
            if subject_id and sess.subject_id != subject_id:
                continue
            results.append(sess)
        return results
