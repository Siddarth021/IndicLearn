"""Assessment session lifecycle management (Phase 7A).

Tracks item presentations, student responses, and exposure history.
"""

from __future__ import annotations

from datetime import datetime, timezone

from indiclearn.assessment.models import AssessmentResponse, AssessmentSession
from indiclearn.assessment.repository import AssessmentSessionRepository


class AssessmentSessionManager:
    """Manages the state and lifecycle of adaptive assessment sessions."""

    def __init__(self, repository: AssessmentSessionRepository) -> None:
        self.repository = repository

    def create_session(
        self,
        student_id: str,
        subject_id: str = "signals_and_systems",
    ) -> AssessmentSession:
        """Initialize a new active assessment session."""
        session = AssessmentSession(
            student_id=student_id,
            subject_id=subject_id,
            started_at=datetime.now(timezone.utc),
            is_active=True,
        )
        self.repository.save_session(session)
        return session

    def get_session(self, session_id: str) -> AssessmentSession | None:
        """Retrieve an assessment session by ID."""
        return self.repository.get_session(session_id)

    def add_item_to_session(self, session: AssessmentSession, item_id: str) -> None:
        """Register an item presented in this session."""
        session.item_ids.append(item_id)
        self.repository.save_session(session)

    def record_response(
        self,
        session: AssessmentSession,
        response: AssessmentResponse,
    ) -> None:
        """Append a student response and advance the index."""
        session.responses.append(response)
        session.current_index = len(session.responses)
        self.repository.save_session(session)

    def complete_session(self, session: AssessmentSession) -> AssessmentSession:
        """Mark the session as completed."""
        session.is_active = False
        session.completed_at = datetime.now(timezone.utc)
        self.repository.save_session(session)
        return session

    def get_seen_item_ids(self, student_id: str, subject_id: str | None = None) -> set[str]:
        """Aggregate all item IDs presented to the student across recent sessions."""
        sessions = self.repository.get_student_sessions(student_id, subject_id)
        seen: set[str] = set()
        for s in sessions:
            seen.update(s.item_ids)
        return seen
