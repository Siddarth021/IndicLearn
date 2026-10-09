"""Recommendation Candidate Generator (Phase 7B).

Coordinates retrieval of academic graph concepts, student knowledge states,
prerequisite context, downstream dependency blockage, and source grounding.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from indiclearn.knowledge_graph.base import GraphStore
from indiclearn.recommendation.action_selector import ActionSelector, PedagogicalActionSelector
from indiclearn.recommendation.models import RecommendationCandidate
from indiclearn.recommendation.prioritizer import (
    MultiSignalRecommendationPrioritizer,
    RecommendationPrioritizer,
)
from indiclearn.recommendation.repository import (
    InMemoryRecommendationRepository,
    RecommendationHistoryRepository,
)
from indiclearn.student_model.models import MasteryStatus
from indiclearn.student_model.service import KnowledgeStateService


@runtime_checkable
class RecommendationCandidateGenerator(Protocol):
    """Protocol for generating and scoring recommendation candidates."""

    def generate_candidates(
        self,
        student_id: str,
        subject_id: str = "signals_and_systems",
        now: datetime | None = None,
    ) -> list[RecommendationCandidate]:
        """Generate and rank all recommendation candidates for a student."""
        ...


class HeuristicRecommendationCandidateGenerator:
    """Default implementation coordinating graph, student state, and prioritization."""

    def __init__(
        self,
        knowledge_service: KnowledgeStateService,
        graph_store: GraphStore | None = None,
        prioritizer: RecommendationPrioritizer | None = None,
        action_selector: ActionSelector | None = None,
        history_repo: RecommendationHistoryRepository | None = None,
    ) -> None:
        self.knowledge_service = knowledge_service
        self.graph_store = graph_store
        self.prioritizer = prioritizer or MultiSignalRecommendationPrioritizer()
        self.action_selector = action_selector or PedagogicalActionSelector()
        self.history_repo = history_repo or InMemoryRecommendationRepository()

    def generate_candidates(
        self,
        student_id: str,
        subject_id: str = "signals_and_systems",
        now: datetime | None = None,
    ) -> list[RecommendationCandidate]:
        """Generate candidates across all concepts in the subject curriculum."""
        curr_time = now or datetime.now(timezone.utc)

        # 1. Obtain concepts from graph store if present
        if self.graph_store is not None:
            concepts = self.graph_store.list_concepts(subject_id=subject_id)
        else:
            # Fall back to concepts currently tracked in student states
            states = self.knowledge_service.repository.get_student_states(student_id, subject_id)
            concepts = []
            for st in states:
                from indiclearn.domain.concepts import Concept
                concepts.append(
                    Concept(
                        concept_id=st.concept_id,
                        subject_id=subject_id,
                        canonical_name=st.concept_id.replace("_", " ").title(),
                    )
                )

        if not concepts:
            return []

        # 2. Query student recommendation history to determine recent actions
        history_entries = self.history_repo.get_history(student_id, subject_id)
        last_action_map: dict[str, datetime] = {}
        for entry in history_entries:
            if entry.concept_id not in last_action_map or entry.timestamp > last_action_map[entry.concept_id]:
                last_action_map[entry.concept_id] = entry.timestamp

        candidates: list[RecommendationCandidate] = []

        for cpt in concepts:
            state = self.knowledge_service.get_concept_state(student_id, cpt.concept_id, subject_id)

            # Prerequisite inspection (direct prerequisites)
            unmastered_prereqs: list[str] = []
            if self.graph_store is not None:
                prereqs = self.graph_store.get_prerequisites(cpt.concept_id, transitive=False)
                for p in prereqs:
                    p_state = self.knowledge_service.get_concept_state(student_id, p.concept_id, subject_id)
                    if p_state.mastery_status != MasteryStatus.MASTERED:
                        unmastered_prereqs.append(p.canonical_name)

            # Downstream dependency inspection (direct dependents)
            blocked_dependents: list[str] = []
            if self.graph_store is not None:
                dependents = self.graph_store.get_dependents(cpt.concept_id, transitive=False)
                for d in dependents:
                    d_state = self.knowledge_service.get_concept_state(student_id, d.concept_id, subject_id)
                    if d_state.mastery_status != MasteryStatus.MASTERED:
                        blocked_dependents.append(d.canonical_name)

            # Source availability
            source_chunks = list(cpt.source_chunk_ids)
            source_avail = len(source_chunks) > 0

            last_action_at = last_action_map.get(cpt.concept_id)

            # Score candidate using multi-signal prioritizer
            candidate = self.prioritizer.score_candidate(
                concept_id=cpt.concept_id,
                canonical_name=cpt.canonical_name,
                state=state,
                unmastered_prereq_names=unmastered_prereqs,
                blocked_dependent_names=blocked_dependents,
                source_chunk_ids=source_chunks,
                source_available=source_avail,
                last_action_at=last_action_at,
                now=curr_time,
            )

            # Determine pedagogical action and difficulty
            action, difficulty = self.action_selector.select_action(candidate)
            candidate.action_type = action
            candidate.suggested_difficulty = difficulty

            candidates.append(candidate)

        # 3. Deterministic ranking:
        # 1. priority_score (descending)
        # 2. uncertainty_score (descending)
        # 3. prerequisite deficit + downstream blocker (descending)
        # 4. concept_id (alphabetical ascending)
        candidates.sort(
            key=lambda c: (
                -c.priority_score,
                -c.uncertainty_score,
                -(c.prerequisite_deficit_score + c.downstream_blocker_score),
                c.concept_id,
            )
        )

        return candidates
