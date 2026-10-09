"""Assessment candidate generation protocol and implementation (Phase 7A).

Inspects the Academic Knowledge Graph and the Student Knowledge Model
to assemble eligible assessment candidates without mutating either layer.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Protocol, runtime_checkable

from indiclearn.assessment.models import AssessmentCandidate
from indiclearn.assessment.prioritizer import CandidatePrioritizer, MultiSignalCandidatePrioritizer
from indiclearn.domain.concepts import Concept
from indiclearn.knowledge_graph.base import GraphStore
from indiclearn.student_model.models import KnowledgeState, MasteryStatus
from indiclearn.student_model.service import KnowledgeStateService

logger = logging.getLogger(__name__)


@runtime_checkable
class AssessmentCandidateGenerator(Protocol):
    """Protocol for generating prioritized assessment candidates."""

    def generate_candidates(
        self,
        student_id: str,
        subject_id: str = "signals_and_systems",
        now: datetime | None = None,
    ) -> list[AssessmentCandidate]:
        """Generate and rank eligible assessment candidates for a student."""
        ...


class HeuristicCandidateGenerator:
    """Deterministic candidate generator combining Academic KG and Student Knowledge State."""

    def __init__(
        self,
        graph_store: GraphStore | None,
        knowledge_service: KnowledgeStateService,
        prioritizer: CandidatePrioritizer | None = None,
        exclude_mastered: bool = True,
    ) -> None:
        self.graph_store = graph_store
        self.knowledge_service = knowledge_service
        self.prioritizer = prioritizer or MultiSignalCandidatePrioritizer()
        self.exclude_mastered = exclude_mastered

    def generate_candidates(
        self,
        student_id: str,
        subject_id: str = "signals_and_systems",
        now: datetime | None = None,
    ) -> list[AssessmentCandidate]:
        """Generate prioritized concept candidates for the student."""
        # 1. Fetch academic concepts from canonical graph
        concepts: list[Concept] = []
        if self.graph_store is not None:
            concepts = self.graph_store.list_concepts(subject_id=subject_id)

        if not concepts:
            # Fallback: check if student has any states in repository
            tracked_states = self.knowledge_service.repository.get_student_states(student_id, subject_id)
            concepts = [
                Concept(
                    concept_id=s.concept_id,
                    subject_id=subject_id,
                    canonical_name=s.concept_id.replace("_", " ").title(),
                )
                for s in tracked_states
            ]

        if not concepts:
            return []

        # 2. Build student state lookup table
        states_lookup: dict[str, KnowledgeState] = {}
        for c in concepts:
            states_lookup[c.concept_id] = self.knowledge_service.get_concept_state(
                student_id=student_id,
                concept_id=c.concept_id,
                subject_id=subject_id,
            )

        # 3. Score all concepts
        candidates: list[AssessmentCandidate] = []
        for c in concepts:
            st = states_lookup[c.concept_id]
            cand = self.prioritizer.prioritize_candidate(
                concept=c,
                state=st,
                graph_store=self.graph_store,
                now=now,
                student_states_lookup=states_lookup,
            )
            candidates.append(cand)

        # 4. Filter mastered concepts if configured and alternatives exist
        if self.exclude_mastered and len(candidates) > 1:
            non_mastered = [
                c
                for c in candidates
                if not (
                    c.mastery_status == MasteryStatus.MASTERED
                    and c.knowledge_score >= 0.80
                    and c.confidence >= 0.60
                )
            ]
            if non_mastered:
                candidates = non_mastered

        # 5. Deterministically rank candidates
        return self.prioritizer.rank_candidates(candidates)
