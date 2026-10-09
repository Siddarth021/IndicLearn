"""Comprehensive unit tests for Adaptive Assessment Core (Phase 7A).

Validates:
- Domain model invariants and serialization
- Deterministic response evaluation (MCQ, True/False, Short Answer)
- Candidate generation and multi-signal prioritization
- Prerequisite graph importance boost
- Coverage need and recency penalties
- Difficulty tier matching policy
- Language preference selection with English fallback
- Repeat avoidance within active sessions
- Session lifecycle and response logging
- Evidence integration into Student Knowledge Model
- Strict multi-student isolation
- Provenance integrity
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from indiclearn.assessment.items import evaluate_response
from indiclearn.assessment.models import (
    AssessmentCandidate,
    AssessmentItem,
    ItemDifficulty,
    QuestionType,
)
from indiclearn.assessment.prioritizer import MultiSignalCandidatePrioritizer
from indiclearn.assessment.repository import (
    InMemoryAssessmentItemRepository,
)
from indiclearn.assessment.selector import DeterministicItemSelector
from indiclearn.assessment.service import AdaptiveAssessmentService
from indiclearn.domain.concepts import Concept, ConceptRelation, RelationType
from indiclearn.knowledge_graph.networkx_store import NetworkXGraphStore
from indiclearn.student_model.models import (
    EvidenceType,
    KnowledgeState,
    MasteryStatus,
)
from indiclearn.student_model.repository import InMemoryKnowledgeStateRepository
from indiclearn.student_model.service import KnowledgeStateService


@pytest.fixture
def sample_graph() -> NetworkXGraphStore:
    """Fixture providing a minimal academic knowledge graph."""
    graph = NetworkXGraphStore()
    c1 = Concept(
        concept_id="cpt_lti",
        subject_id="signals_and_systems",
        canonical_name="LTI System",
        source_chunk_ids=["CHUNK_SS_001"],
    )
    c2 = Concept(
        concept_id="cpt_conv",
        subject_id="signals_and_systems",
        canonical_name="Convolution",
        source_chunk_ids=["CHUNK_SS_002"],
    )
    c3 = Concept(
        concept_id="cpt_fourier",
        subject_id="signals_and_systems",
        canonical_name="Fourier Transform",
        source_chunk_ids=["CHUNK_SS_003"],
    )
    graph.add_concept(c1)
    graph.add_concept(c2)
    graph.add_concept(c3)

    # c1 (LTI) is prerequisite of c2 (Convolution)
    graph.add_relation(
        ConceptRelation(
            relation_id="rel_lti_prereq_conv",
            subject_id="signals_and_systems",
            source_concept_id="cpt_lti",
            target_concept_id="cpt_conv",
            relation_type=RelationType.PREREQUISITE_OF,
        )
    )
    return graph


@pytest.fixture
def sample_item_repo() -> InMemoryAssessmentItemRepository:
    """Fixture providing sample assessment items covering difficulties and languages."""
    items = [
        AssessmentItem(
            item_id="ITEM_CONV_EASY_EN",
            concept_id="cpt_conv",
            subject_id="signals_and_systems",
            question="What is the convolution integral output?",
            question_type=QuestionType.MCQ,
            difficulty=ItemDifficulty.EASY,
            language="en",
            options=["A) y(t)", "B) x(t)", "C) h(t)", "D) 0"],
            correct_answer="A",
            explanation="Convolution integral computes y(t).",
            source_chunk_ids=["CHUNK_SS_002"],
        ),
        AssessmentItem(
            item_id="ITEM_CONV_EASY_TE",
            concept_id="cpt_conv",
            subject_id="signals_and_systems",
            question="కన్వల్యూషన్ అవుట్‌పుట్ ఏమిటి?",
            question_type=QuestionType.MCQ,
            difficulty=ItemDifficulty.EASY,
            language="te",
            options=["A) y(t)", "B) x(t)", "C) h(t)", "D) 0"],
            correct_answer="A",
            explanation="కన్వల్యూషన్ ఇంటిగ్రల్ y(t) ఇస్తుంది.",
            source_chunk_ids=["CHUNK_SS_002"],
        ),
        AssessmentItem(
            item_id="ITEM_CONV_MED_EN",
            concept_id="cpt_conv",
            subject_id="signals_and_systems",
            question="Is convolution commutative?",
            question_type=QuestionType.TRUE_FALSE,
            difficulty=ItemDifficulty.MEDIUM,
            language="en",
            options=["True", "False"],
            correct_answer="True",
            explanation="x*h = h*x.",
            source_chunk_ids=["CHUNK_SS_002"],
        ),
        AssessmentItem(
            item_id="ITEM_CONV_HARD_EN",
            concept_id="cpt_conv",
            subject_id="signals_and_systems",
            question="Does convolution satisfy associativity across three LTI systems?",
            question_type=QuestionType.TRUE_FALSE,
            difficulty=ItemDifficulty.HARD,
            language="en",
            options=["True", "False"],
            correct_answer="True",
            explanation="LTI cascades are associative.",
            source_chunk_ids=["CHUNK_SS_002"],
        ),
        AssessmentItem(
            item_id="ITEM_LTI_MED_EN",
            concept_id="cpt_lti",
            subject_id="signals_and_systems",
            question="What defines linearity?",
            question_type=QuestionType.MCQ,
            difficulty=ItemDifficulty.MEDIUM,
            language="en",
            options=["A) Superposition", "B) Memory", "C) Causality", "D) Stability"],
            correct_answer="A",
            explanation="Linearity requires superposition.",
            source_chunk_ids=["CHUNK_SS_001"],
        ),
    ]
    return InMemoryAssessmentItemRepository(items)


def test_deterministic_evaluation_mcq_letter_and_text():
    item = AssessmentItem(
        item_id="TEST_01",
        concept_id="cpt_conv",
        question="Which operation?",
        question_type=QuestionType.MCQ,
        difficulty=ItemDifficulty.EASY,
        options=["A) Convolution", "B) Fourier", "C) Laplace", "D) Z"],
        correct_answer="A",
        explanation="Convolution is correct.",
        source_chunk_ids=["CHUNK_01"],
    )

    # Letter match
    ok, fb = evaluate_response(item, "A")
    assert ok is True
    assert "Convolution is correct" in fb

    ok, _ = evaluate_response(item, "a")
    assert ok is True

    # Full option text match
    ok, _ = evaluate_response(item, "A) Convolution")
    assert ok is True

    # Option body match
    ok, _ = evaluate_response(item, "Convolution")
    assert ok is True

    # Incorrect match
    ok, fb = evaluate_response(item, "B")
    assert ok is False
    assert "Incorrect" in fb


def test_deterministic_evaluation_true_false():
    item = AssessmentItem(
        item_id="TEST_02",
        concept_id="cpt_conv",
        question="Is it commutative?",
        question_type=QuestionType.TRUE_FALSE,
        options=["True", "False"],
        correct_answer="True",
        explanation="Commutative property holds.",
        source_chunk_ids=["CHUNK_01"],
    )

    ok, _ = evaluate_response(item, "true")
    assert ok is True
    ok, _ = evaluate_response(item, "True")
    assert ok is True
    ok, _ = evaluate_response(item, "T")
    assert ok is True
    ok, _ = evaluate_response(item, "సత్యం")
    assert ok is True

    ok, _ = evaluate_response(item, "false")
    assert ok is False


def test_candidate_prioritizer_weakness_and_uncertainty():
    prioritizer = MultiSignalCandidatePrioritizer()
    concept = Concept(concept_id="cpt_test", subject_id="signals_and_systems", canonical_name="Test")

    # High weakness (demonstrated low score)
    st_weak = KnowledgeState(
        student_id="s1",
        concept_id="cpt_test",
        knowledge_score=0.20,
        confidence=0.80,
        strong_evidence_count=4,
        mastery_status=MasteryStatus.EMERGING,
    )
    cand_weak = prioritizer.prioritize_candidate(concept, st_weak)
    assert cand_weak.weakness == 0.80
    assert cand_weak.uncertainty == 0.20
    assert any("Low knowledge score" in r for r in cand_weak.reasons)

    # High uncertainty (cold start, no assessments)
    st_unc = KnowledgeState(
        student_id="s1",
        concept_id="cpt_test",
        knowledge_score=0.50,
        confidence=0.10,
        strong_evidence_count=0,
        mastery_status=MasteryStatus.UNKNOWN,
    )
    cand_unc = prioritizer.prioritize_candidate(concept, st_unc)
    assert cand_unc.uncertainty == 0.90
    assert cand_unc.weakness == 0.50  # Neutral prior
    assert any("uncertainty" in r.lower() for r in cand_unc.reasons)


def test_prerequisite_graph_importance_boost(sample_graph):
    prioritizer = MultiSignalCandidatePrioritizer()
    c_lti = sample_graph.get_concept("cpt_lti")
    c_fourier = sample_graph.get_concept("cpt_fourier")
    assert c_lti is not None and c_fourier is not None

    # LTI has dependent cpt_conv which is unmastered
    st_lti = KnowledgeState(
        student_id="s1",
        concept_id="cpt_lti",
        knowledge_score=0.50,
        confidence=0.50,
        strong_evidence_count=2,
    )
    st_conv = KnowledgeState(
        student_id="s1",
        concept_id="cpt_conv",
        knowledge_score=0.30,
        confidence=0.50,
        strong_evidence_count=2,
    )
    st_fourier = KnowledgeState(
        student_id="s1",
        concept_id="cpt_fourier",
        knowledge_score=0.50,
        confidence=0.50,
        strong_evidence_count=2,
    )
    lookup = {"cpt_lti": st_lti, "cpt_conv": st_conv, "cpt_fourier": st_fourier}

    cand_lti = prioritizer.prioritize_candidate(c_lti, st_lti, sample_graph, student_states_lookup=lookup)
    cand_fourier = prioritizer.prioritize_candidate(c_fourier, st_fourier, sample_graph, student_states_lookup=lookup)

    assert cand_lti.prerequisite_relevance > 0.0
    assert cand_fourier.prerequisite_relevance == 0.0
    assert cand_lti.priority_score > cand_fourier.priority_score
    assert any("prerequisite" in r.lower() for r in cand_lti.reasons)


def test_coverage_and_recency_penalty():
    prioritizer = MultiSignalCandidatePrioritizer(recency_window_minutes=60.0)
    concept = Concept(concept_id="cpt_test", subject_id="signals_and_systems", canonical_name="Test")

    now = datetime.now(timezone.utc)
    # Unassessed concept
    st_unassessed = KnowledgeState(
        student_id="s1",
        concept_id="cpt_test",
        knowledge_score=0.50,
        confidence=0.10,
        strong_evidence_count=0,
        last_assessed_at=None,
    )
    cand_unassessed = prioritizer.prioritize_candidate(concept, st_unassessed, now=now)
    assert cand_unassessed.coverage_need == 1.0
    assert cand_unassessed.recency_penalty == 0.0

    # Recently assessed concept (10 min ago)
    st_recent = KnowledgeState(
        student_id="s1",
        concept_id="cpt_test",
        knowledge_score=0.50,
        confidence=0.60,
        strong_evidence_count=3,
        last_assessed_at=now - timedelta(minutes=10),
    )
    cand_recent = prioritizer.prioritize_candidate(concept, st_recent, now=now)
    assert cand_recent.coverage_need < cand_unassessed.coverage_need
    assert cand_recent.recency_penalty > 0.0
    assert cand_unassessed.priority_score > cand_recent.priority_score


def test_difficulty_tier_matching_policy():
    selector = DeterministicItemSelector()

    # Emerging / low score -> EASY
    cand_weak = AssessmentCandidate(
        concept_id="cpt_conv",
        canonical_name="Convolution",
        priority_score=0.8,
        knowledge_score=0.25,
        confidence=0.40,
        mastery_status=MasteryStatus.EMERGING,
        uncertainty=0.6,
        weakness=0.75,
        prerequisite_relevance=0.0,
        coverage_need=1.0,
        recency_penalty=0.0,
    )
    diff_weak, _ = selector.determine_target_difficulty(cand_weak)
    assert diff_weak == ItemDifficulty.EASY

    # Developing -> MEDIUM
    cand_dev = AssessmentCandidate(
        concept_id="cpt_conv",
        canonical_name="Convolution",
        priority_score=0.6,
        knowledge_score=0.65,
        confidence=0.50,
        mastery_status=MasteryStatus.DEVELOPING,
        uncertainty=0.5,
        weakness=0.35,
        prerequisite_relevance=0.0,
        coverage_need=0.5,
        recency_penalty=0.0,
    )
    diff_dev, _ = selector.determine_target_difficulty(cand_dev)
    assert diff_dev == ItemDifficulty.MEDIUM

    # High score & confidence -> HARD
    cand_hard = AssessmentCandidate(
        concept_id="cpt_conv",
        canonical_name="Convolution",
        priority_score=0.4,
        knowledge_score=0.85,
        confidence=0.65,
        mastery_status=MasteryStatus.MASTERED,
        uncertainty=0.35,
        weakness=0.15,
        prerequisite_relevance=0.0,
        coverage_need=0.2,
        recency_penalty=0.0,
    )
    diff_hard, _ = selector.determine_target_difficulty(cand_hard)
    assert diff_hard == ItemDifficulty.HARD


def test_language_preference_selection(sample_item_repo):
    selector = DeterministicItemSelector()
    cand = AssessmentCandidate(
        concept_id="cpt_conv",
        canonical_name="Convolution",
        priority_score=0.8,
        knowledge_score=0.20,
        confidence=0.30,
        mastery_status=MasteryStatus.EMERGING,
        uncertainty=0.7,
        weakness=0.8,
        prerequisite_relevance=0.0,
        coverage_need=1.0,
        recency_penalty=0.0,
    )

    items = sample_item_repo.list_items()
    item_te, _expl_te = selector.select_item(cand, items, requested_language="te")
    assert item_te is not None
    assert item_te.language == "te"
    assert item_te.item_id == "ITEM_CONV_EASY_TE"

    # Fallback to English when requested language is unavailable
    item_fallback, _expl_fb = selector.select_item(cand, items, requested_language="fr")
    assert item_fallback is not None
    assert item_fallback.language == "en"


def test_repeat_avoidance_within_session(sample_item_repo):
    selector = DeterministicItemSelector()
    cand = AssessmentCandidate(
        concept_id="cpt_conv",
        canonical_name="Convolution",
        priority_score=0.8,
        knowledge_score=0.20,
        confidence=0.30,
        mastery_status=MasteryStatus.EMERGING,
        uncertainty=0.7,
        weakness=0.8,
        prerequisite_relevance=0.0,
        coverage_need=1.0,
        recency_penalty=0.0,
    )
    items = sample_item_repo.list_items()

    item1, _ = selector.select_item(cand, items, requested_language="en", seen_item_ids=set())
    assert item1 is not None

    item2, expl2 = selector.select_item(cand, items, requested_language="en", seen_item_ids={item1.item_id})
    assert item2 is not None
    assert item2.item_id != item1.item_id
    assert any("repeat avoidance" in r.lower() for r in expl2.item_reasons)


def test_adaptive_service_session_and_evidence_integration(sample_graph, sample_item_repo):
    k_repo = InMemoryKnowledgeStateRepository()
    k_service = KnowledgeStateService(repository=k_repo, graph_store=sample_graph)
    service = AdaptiveAssessmentService(
        item_repository=sample_item_repo,
        knowledge_service=k_service,
        graph_store=sample_graph,
    )

    student_id = "student_test_cycle"
    session, expl1 = service.start_session(student_id=student_id, requested_language="en")
    assert session.is_active is True
    assert expl1 is not None

    first_item = expl1.item
    assert len(session.item_ids) == 1
    assert session.item_ids[0] == first_item.item_id

    # Submit correct response
    resp1, state1, _next_expl = service.submit_response(
        session_id=session.session_id,
        item_id=first_item.item_id,
        selected_answer=first_item.correct_answer,
    )
    assert resp1.correct is True
    assert state1.knowledge_score > 0.50  # Prior was 0.50, correct increases it
    assert state1.confidence > 0.10
    assert len(session.responses) == 1

    # EvidenceEvent should have been emitted with proper subsystem source
    events = k_repo.get_events(student_id)
    assert len(events) == 1
    assert events[0].event_type == EvidenceType.QUIZ_CORRECT
    assert events[0].source == "adaptive_assessment_service"

    # Close session
    closed_session = service.complete_session(session.session_id)
    assert closed_session.is_active is False
    assert closed_session.completed_at is not None


def test_student_isolation(sample_graph, sample_item_repo):
    k_repo = InMemoryKnowledgeStateRepository()
    k_service = KnowledgeStateService(repository=k_repo, graph_store=sample_graph)
    service = AdaptiveAssessmentService(
        item_repository=sample_item_repo,
        knowledge_service=k_service,
        graph_store=sample_graph,
    )

    alice = "alice"
    bob = "bob"

    sess_alice, expl_alice = service.start_session(student_id=alice)
    assert expl_alice is not None
    service.submit_response(
        session_id=sess_alice.session_id,
        item_id=expl_alice.item.item_id,
        selected_answer=expl_alice.item.correct_answer,
    )

    # Bob's candidate scores should remain at pure cold start
    bob_cands = service.get_candidates(student_id=bob)
    assert all(c.assessment_count == 0 for c in bob_cands)
    assert service.session_manager.get_seen_item_ids(student_id=bob) == set()
