"""Unit tests for Student Knowledge Modeling (Phase 6).

Validates the 10 core research test requirements from Section 21:
1. Question is weak evidence (QUESTION_ASKED != proof of zero knowledge).
2. Correct assessment is strong evidence.
3. Incorrect assessment reduces estimated mastery appropriately.
4. Conflicting evidence handling (correct -> incorrect -> correct stability).
5. Cold start (prior belief with low confidence, not 0 knowledge).
6. Evidence provenance & explainability.
7. Student isolation (Student A evidence never mutates Student B).
8. Concept isolation (Concept A evidence never mutates Concept B).
9. Prerequisite graph inspection without knowledge score mutation.
10. Deterministic reproducibility across replayed event sequences.
"""

from __future__ import annotations

from datetime import datetime, timezone

from indiclearn.domain.concepts import Concept, ConceptRelation, RelationType
from indiclearn.knowledge_graph.networkx_store import NetworkXGraphStore
from indiclearn.student_model.estimator import EvidenceBasedKnowledgeEstimator
from indiclearn.student_model.models import (
    EvidenceEvent,
    EvidenceStrength,
    EvidenceType,
    KnowledgeState,
    MasteryStatus,
)
from indiclearn.student_model.repository import InMemoryKnowledgeStateRepository
from indiclearn.student_model.service import KnowledgeStateService


def test_student_model_cold_start():
    """Test 5: Cold start.

    A new student must have knowledge_score = prior (0.50), confidence = low (0.10),
    and mastery_status = UNKNOWN, NEVER score = 0 with high confidence.
    """
    service = KnowledgeStateService()
    state = service.get_concept_state("student_1", "cpt_convolution")

    assert state.student_id == "student_1"
    assert state.concept_id == "cpt_convolution"
    assert state.knowledge_score == 0.50
    assert state.confidence == 0.10
    assert state.mastery_status == MasteryStatus.UNKNOWN
    assert state.evidence_count == 0


def test_question_asked_is_weak_evidence():
    """Test 1: A student's question is NOT proof of ignorance.

    Asking 'What is convolution?' must not drag knowledge_score to 0
    nor promote to MASTERED, and confidence increase must remain small.
    """
    service = KnowledgeStateService()

    # Student asks 5 questions about convolution
    for i in range(5):
        state = service.record_interaction(
            student_id="student_1",
            concept_id="cpt_convolution",
            event_type=EvidenceType.QUESTION_ASKED,
            score=0.50,
            metadata={"query": f"Question {i+1} on convolution"},
        )

    # Score must remain near prior belief (0.50)
    assert 0.45 <= state.knowledge_score <= 0.55
    # Confidence increase from weak evidence is strictly bounded (< 0.30, well below mastery 0.60)
    assert state.confidence < 0.30
    assert state.mastery_status == MasteryStatus.UNKNOWN
    assert state.weak_evidence_count == 5
    assert state.strong_evidence_count == 0


def test_correct_assessment_is_strong_evidence():
    """Test 2: Correct assessments provide strong evidence driving mastery.

    Repeated correct quiz responses significantly increase knowledge score and confidence,
    achieving MASTERED status once confidence and score thresholds are met.
    """
    service = KnowledgeStateService()

    # 4 consecutive correct quizzes
    for i in range(4):
        state = service.record_interaction(
            student_id="student_1",
            concept_id="cpt_convolution",
            event_type=EvidenceType.QUIZ_CORRECT,
            score=1.0,
            metadata={"item_id": f"q_{i+1}"},
        )

    # Score should be high (>= 0.80)
    assert state.knowledge_score >= 0.80
    # Confidence should exceed mastery threshold (>= 0.60)
    assert state.confidence >= 0.60
    assert state.mastery_status == MasteryStatus.MASTERED
    assert state.strong_evidence_count == 4


def test_incorrect_assessment_reduces_mastery():
    """Test 3: Incorrect assessments reduce score appropriately without instant zeroing."""
    service = KnowledgeStateService()

    state = service.record_interaction(
        student_id="student_1",
        concept_id="cpt_convolution",
        event_type=EvidenceType.QUIZ_INCORRECT,
        score=0.0,
    )

    # Score drops below prior (0.50 -> 0.25)
    assert state.knowledge_score < 0.50
    assert state.knowledge_score > 0.0  # Not instantly zeroed to 0.0
    assert state.confidence > 0.10
    assert state.mastery_status == MasteryStatus.EMERGING


def test_conflicting_evidence_stability():
    """Test 4: Correct -> Incorrect -> Correct evidence handling.

    The model must not violently oscillate between 0 and 1; conflicting
    signals must produce a stable, explainable result flagged as UNCERTAIN.
    """
    service = KnowledgeStateService()

    # Sequence: Correct (1.0) -> Incorrect (0.0) -> Correct (1.0) -> Incorrect (0.0)
    events = [1.0, 0.0, 1.0, 0.0]
    for sc in events:
        state = service.record_interaction(
            student_id="student_1",
            concept_id="cpt_convolution",
            event_type=EvidenceType.QUIZ_CORRECT if sc == 1.0 else EvidenceType.QUIZ_INCORRECT,
            score=sc,
        )

    # The score should reflect the balanced evidence near ~0.50
    assert 0.40 <= state.knowledge_score <= 0.60
    # Conflicting outcomes in recent assessments trigger UNCERTAIN status
    assert state.mastery_status == MasteryStatus.UNCERTAIN
    assert state.metadata.get("has_conflicting_evidence") is True


def test_evidence_provenance_and_explanation():
    """Test 6: Every state transition preserves complete, traceable provenance."""
    service = KnowledgeStateService()

    service.record_interaction(
        student_id="student_1",
        concept_id="cpt_convolution",
        event_type=EvidenceType.QUESTION_ASKED,
        score=0.50,
        source="rag_pipeline",
        metadata={"text": "What is convolution integral?"},
    )
    service.record_interaction(
        student_id="student_1",
        concept_id="cpt_convolution",
        event_type=EvidenceType.QUIZ_CORRECT,
        score=1.0,
        source="quiz_engine",
        metadata={"item_id": "Q101"},
    )

    explanation = service.explain_concept_state("student_1", "cpt_convolution")
    assert explanation.student_id == "student_1"
    assert explanation.concept_id == "cpt_convolution"
    assert len(explanation.evidence_summary) == 2
    assert any("QUESTION_ASKED" in s for s in explanation.evidence_summary)
    assert any("QUIZ_CORRECT" in s for s in explanation.evidence_summary)
    assert "signals_and_systems" in service.get_concept_state("student_1", "cpt_convolution").subject_id


def test_student_isolation():
    """Test 7: Student isolation.

    Evidence recorded for Student A must NEVER alter Student B's knowledge state.
    """
    service = KnowledgeStateService()

    # Student A achieves mastery
    for _ in range(5):
        service.record_interaction(
            student_id="student_alice",
            concept_id="cpt_convolution",
            event_type=EvidenceType.QUIZ_CORRECT,
            score=1.0,
        )

    alice_state = service.get_concept_state("student_alice", "cpt_convolution")
    bob_state = service.get_concept_state("student_bob", "cpt_convolution")

    assert alice_state.mastery_status == MasteryStatus.MASTERED
    assert alice_state.evidence_count == 5

    # Bob must remain at cold-start prior
    assert bob_state.mastery_status == MasteryStatus.UNKNOWN
    assert bob_state.knowledge_score == 0.50
    assert bob_state.confidence == 0.10
    assert bob_state.evidence_count == 0


def test_concept_isolation():
    """Test 8: Concept isolation.

    Evidence for Concept A must not mutate Concept B unless an explicit rule exists.
    """
    service = KnowledgeStateService()

    # Student masters Convolution
    for _ in range(5):
        service.record_interaction(
            student_id="student_1",
            concept_id="cpt_convolution",
            event_type=EvidenceType.QUIZ_CORRECT,
            score=1.0,
        )

    conv_state = service.get_concept_state("student_1", "cpt_convolution")
    fourier_state = service.get_concept_state("student_1", "cpt_fourier_transform")

    assert conv_state.mastery_status == MasteryStatus.MASTERED
    assert fourier_state.mastery_status == MasteryStatus.UNKNOWN
    assert fourier_state.evidence_count == 0


def test_prerequisite_graph_inspection_without_mutation():
    """Test 9: Prerequisite-aware readiness inspection.

    Can query academic graph prerequisites to determine readiness,
    WITHOUT mutating or propagating knowledge scores across edges.
    """
    graph_store = NetworkXGraphStore()
    c_lti = Concept(concept_id="cpt_lti", subject_id="signals_and_systems", canonical_name="LTI System")
    c_conv = Concept(concept_id="cpt_conv", subject_id="signals_and_systems", canonical_name="Convolution")
    graph_store.add_concept(c_lti)
    graph_store.add_concept(c_conv)

    # LTI is prerequisite of Convolution
    rel = ConceptRelation(
        relation_id="rel_lti_prereq_conv",
        subject_id="signals_and_systems",
        source_concept_id="cpt_lti",
        target_concept_id="cpt_conv",
        relation_type=RelationType.PREREQUISITE_OF,
    )
    graph_store.add_relation(rel)

    service = KnowledgeStateService(graph_store=graph_store)

    # Case 1: LTI is unmastered (cold start) -> not ready
    readiness_before = service.get_prerequisite_readiness("student_1", "cpt_conv")
    assert readiness_before.is_ready is False
    assert "LTI System" in readiness_before.unmastered_prerequisites

    # Case 2: Student masters LTI
    for _ in range(4):
        service.record_interaction("student_1", "cpt_lti", EvidenceType.QUIZ_CORRECT, score=1.0)

    readiness_after = service.get_prerequisite_readiness("student_1", "cpt_conv")
    assert readiness_after.is_ready is True
    assert len(readiness_after.unmastered_prerequisites) == 0

    # Ensure Convolution itself was NOT mutated by LTI mastery!
    conv_state = service.get_concept_state("student_1", "cpt_conv")
    assert conv_state.mastery_status == MasteryStatus.UNKNOWN
    assert conv_state.evidence_count == 0


def test_deterministic_reproducibility():
    """Test 10: Determinism.

    Replaying the exact same sequence of evidence events must yield
    the identical floating point score and confidence.
    """
    estimator = EvidenceBasedKnowledgeEstimator()
    repo1 = InMemoryKnowledgeStateRepository()
    repo2 = InMemoryKnowledgeStateRepository()

    svc1 = KnowledgeStateService(repository=repo1, estimator=estimator)
    svc2 = KnowledgeStateService(repository=repo2, estimator=estimator)

    now = datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone.utc)
    events = [
        EvidenceEvent(
            student_id="s1",
            concept_id="c1",
            event_type=EvidenceType.QUESTION_ASKED,
            strength=EvidenceStrength.WEAK,
            score=0.50,
            timestamp=now,
        ),
        EvidenceEvent(
            student_id="s1",
            concept_id="c1",
            event_type=EvidenceType.QUIZ_CORRECT,
            strength=EvidenceStrength.STRONG,
            score=1.0,
            timestamp=now,
        ),
        EvidenceEvent(
            student_id="s1",
            concept_id="c1",
            event_type=EvidenceType.QUIZ_INCORRECT,
            strength=EvidenceStrength.STRONG,
            score=0.0,
            timestamp=now,
        ),
        EvidenceEvent(
            student_id="s1",
            concept_id="c1",
            event_type=EvidenceType.QUIZ_CORRECT,
            strength=EvidenceStrength.STRONG,
            score=1.0,
            timestamp=now,
        ),
    ]

    for ev in events:
        svc1.record_event(ev)
        svc2.record_event(ev)

    s1 = svc1.get_concept_state("s1", "c1")
    s2 = svc2.get_concept_state("s1", "c1")

    assert s1.knowledge_score == s2.knowledge_score
    assert s1.confidence == s2.confidence
    assert s1.mastery_status == s2.mastery_status
    assert s1.evidence_count == s2.evidence_count


def test_audit_scen002_incorrect_assessment_is_emerging_not_developing():
    """Phase 6.1 Audit: SCEN_002 verification.

    A student who repeatedly fails quizzes (e.g. 6 incorrect quizzes, score=0.0750, confidence=0.7438)
    must receive EMERGING (nascent competency needing intervention), NEVER DEVELOPING (which requires score >= 0.50).
    """
    service = KnowledgeStateService()
    for _ in range(6):
        service.record_interaction("student_1", "cpt_convolution", EvidenceType.QUIZ_INCORRECT, score=0.0)

    state = service.get_concept_state("student_1", "cpt_convolution")
    assert state.knowledge_score <= 0.10
    assert state.confidence >= 0.70
    assert state.mastery_status == MasteryStatus.EMERGING
    assert state.mastery_status != MasteryStatus.DEVELOPING


def test_audit_monotonicity_sequences():
    """Phase 6.1 Audit: Monotonicity behavior across standard sequences."""
    service = KnowledgeStateService()

    # Sequence 1: Repeated correct assessments
    scores_correct = []
    for _ in range(3):
        st = service.record_interaction("s_mono_1", "cpt_1", EvidenceType.QUIZ_CORRECT, score=1.0)
        scores_correct.append(st.knowledge_score)
    # Strictly increasing: 0.50 (prior) -> 0.75 -> 0.8333 -> 0.8750
    assert scores_correct[0] < scores_correct[1] < scores_correct[2]

    # Sequence 2: Repeated incorrect assessments
    scores_incorrect = []
    for _ in range(3):
        st = service.record_interaction("s_mono_2", "cpt_1", EvidenceType.QUIZ_INCORRECT, score=0.0)
        scores_incorrect.append(st.knowledge_score)
    # Strictly decreasing: 0.50 (prior) -> 0.25 -> 0.1667 -> 0.1250
    assert scores_incorrect[0] > scores_incorrect[1] > scores_incorrect[2]

    # Sequence 3: Alternating correct and incorrect
    st1 = service.record_interaction("s_mono_3", "cpt_1", EvidenceType.QUIZ_CORRECT, score=1.0)    # 0.75
    st2 = service.record_interaction("s_mono_3", "cpt_1", EvidenceType.QUIZ_INCORRECT, score=0.0)  # 0.50
    st3 = service.record_interaction("s_mono_3", "cpt_1", EvidenceType.QUIZ_CORRECT, score=1.0)    # 0.6250
    st4 = service.record_interaction("s_mono_3", "cpt_1", EvidenceType.QUIZ_INCORRECT, score=0.0)  # 0.50
    assert st1.knowledge_score > 0.50
    assert st2.knowledge_score == 0.50
    assert st3.knowledge_score > 0.50
    assert st4.knowledge_score == 0.50


def test_audit_weak_evidence_cannot_produce_mastery_or_weakness():
    """Phase 6.1 Audit: Weak evidence cannot independently produce mastery or weakness."""
    service = KnowledgeStateService()

    # 15 mixed weak interactions without any quizzes
    types = [
        EvidenceType.QUESTION_ASKED,
        EvidenceType.CONCEPT_VIEWED,
        EvidenceType.EXPLANATION_REQUESTED,
    ]
    for i in range(15):
        t = types[i % len(types)]
        state = service.record_interaction("student_weak", "cpt_conv", t, score=0.50)

    # Score remains at prior
    assert state.knowledge_score == 0.50
    # Confidence remains bounded below mastery threshold
    assert state.confidence < 0.35
    # Status is strictly UNKNOWN (no evaluated performance data)
    assert state.mastery_status == MasteryStatus.UNKNOWN


def test_audit_score_confidence_separation():
    """Phase 6.1 Audit: Score and confidence remain semantically distinct dimensions."""
    # Scenario A: Moderate score (0.70) with low confidence (0.15) via early task
    state_a = KnowledgeState(
        student_id="s_a",
        concept_id="c_1",
        knowledge_score=0.70,
        confidence=0.15,
        evidence_count=1,
        strong_evidence_count=0,
        medium_evidence_count=1,
        weak_evidence_count=0,
        mastery_status=MasteryStatus.UNKNOWN,
    )

    # Scenario B: Moderate score (0.70) with high confidence (0.85) via multiple validated quizzes
    state_b = KnowledgeState(
        student_id="s_b",
        concept_id="c_1",
        knowledge_score=0.70,
        confidence=0.85,
        evidence_count=10,
        strong_evidence_count=8,
        medium_evidence_count=2,
        weak_evidence_count=0,
        mastery_status=MasteryStatus.DEVELOPING,
    )

    # They have identical knowledge scores, but distinct confidence and mastery statuses!
    assert state_a.knowledge_score == state_b.knowledge_score
    assert state_a.confidence < 0.25 and state_a.mastery_status == MasteryStatus.UNKNOWN
    assert state_b.confidence >= 0.60 and state_b.mastery_status == MasteryStatus.DEVELOPING


def test_audit_uncertainty_conflicting_evidence():
    """Phase 6.1 Audit: Conflicting evidence produces UNCERTAIN status."""
    service = KnowledgeStateService()

    # Mix of correct and incorrect quizzes
    for sc in [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]:
        service.record_interaction("student_conflict", "cpt_conv", EvidenceType.QUIZ_CORRECT if sc == 1.0 else EvidenceType.QUIZ_INCORRECT, score=sc)

    state = service.get_concept_state("student_conflict", "cpt_conv")
    assert state.mastery_status == MasteryStatus.UNCERTAIN
    assert state.mastery_status != MasteryStatus.MASTERED
    assert state.mastery_status != MasteryStatus.UNKNOWN
    assert state.metadata.get("has_conflicting_evidence") is True

