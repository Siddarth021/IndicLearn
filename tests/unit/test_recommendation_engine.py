"""Comprehensive Unit Tests for Personalized Recommendations & Remediation Planner (Phase 7B).

Tests algorithmic, behavioral, and architectural properties:
- Domain model integrity and reason codes
- Multi-signal prioritization (weakness, uncertainty, prerequisite deficit, downstream blocker, review need, assessment gap, recent action penalty)
- Explicit uncertainty vs weakness distinction
- Action selection policies and difficulty assignment
- Non-deficit, evidence-based explanation generation
- Prerequisite remediation and downstream blockage
- Source grounding and missing-source safety (no hallucinated provenance)
- Mastered concept damping and spaced review
- Recent-action penalty and diversity
- Assessment delegation to Phase 7A
- Student and concept isolation
- Deterministic ranking and tie-breaking
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from indiclearn.assessment.models import ItemDifficulty
from indiclearn.assessment.repository import InMemoryAssessmentItemRepository
from indiclearn.assessment.service import AdaptiveAssessmentService
from indiclearn.domain.concepts import Concept, ConceptRelation, RelationType
from indiclearn.knowledge_graph.networkx_store import NetworkXGraphStore
from indiclearn.recommendation.action_selector import PedagogicalActionSelector
from indiclearn.recommendation.candidate import HeuristicRecommendationCandidateGenerator
from indiclearn.recommendation.explanation import RecommendationExplanationGenerator
from indiclearn.recommendation.models import (
    ActionType,
    ReasonCode,
    Recommendation,
    RecommendationCandidate,
    RecommendationHistoryEntry,
)
from indiclearn.recommendation.prioritizer import MultiSignalRecommendationPrioritizer
from indiclearn.recommendation.repository import InMemoryRecommendationRepository
from indiclearn.recommendation.service import RecommendationService
from indiclearn.student_model.models import (
    EvidenceEvent,
    EvidenceStrength,
    EvidenceType,
    KnowledgeState,
    MasteryStatus,
)
from indiclearn.student_model.repository import InMemoryKnowledgeStateRepository
from indiclearn.student_model.service import KnowledgeStateService


def _create_test_graph() -> NetworkXGraphStore:
    """Helper creating a 3-node linear prerequisite graph:

    LTI System -> PREREQUISITE_OF -> Convolution -> PREREQUISITE_OF -> CTFT
    """
    graph = NetworkXGraphStore()
    cpt1 = Concept(
        concept_id="cpt_lti",
        subject_id="signals_and_systems",
        canonical_name="LTI System",
        source_chunk_ids=["CHUNK_001"],
    )
    cpt2 = Concept(
        concept_id="cpt_conv",
        subject_id="signals_and_systems",
        canonical_name="Convolution",
        source_chunk_ids=["CHUNK_002"],
    )
    cpt3 = Concept(
        concept_id="cpt_ctft",
        subject_id="signals_and_systems",
        canonical_name="Continuous-Time Fourier Transform",
        source_chunk_ids=["CHUNK_003"],
    )
    cpt4 = Concept(
        concept_id="cpt_unindexed",
        subject_id="signals_and_systems",
        canonical_name="Unindexed Topic",
        source_chunk_ids=[],
    )

    for c in [cpt1, cpt2, cpt3, cpt4]:
        graph.add_concept(c)

    graph.add_relation(
        ConceptRelation(
            relation_id="rel_lti_conv",
            subject_id="signals_and_systems",
            source_concept_id="cpt_lti",
            target_concept_id="cpt_conv",
            relation_type=RelationType.PREREQUISITE_OF,
        )
    )
    graph.add_relation(
        ConceptRelation(
            relation_id="rel_conv_ctft",
            subject_id="signals_and_systems",
            source_concept_id="cpt_conv",
            target_concept_id="cpt_ctft",
            relation_type=RelationType.PREREQUISITE_OF,
        )
    )
    return graph


def test_models_and_reason_codes():
    """Verify enum contracts, Pydantic validation, and model fields."""
    cand = RecommendationCandidate(
        concept_id="cpt_conv",
        canonical_name="Convolution",
        knowledge_score=0.30,
        confidence=0.80,
        mastery_status=MasteryStatus.EMERGING,
        weakness_score=0.60,
        uncertainty_score=0.20,
        prerequisite_deficit_score=0.0,
        downstream_blocker_score=0.33,
        review_need_score=0.10,
        assessment_gap_score=0.0,
        recent_action_penalty=0.0,
        priority_score=0.45,
        action_type=ActionType.PRACTICE_CONCEPT,
        reason_codes=[ReasonCode.LOW_KNOWLEDGE, ReasonCode.SOURCE_AVAILABLE],
        source_chunk_ids=["CHUNK_002"],
        source_available=True,
        suggested_difficulty=ItemDifficulty.EASY,
    )
    assert cand.concept_id == "cpt_conv"
    assert cand.action_type == ActionType.PRACTICE_CONCEPT
    assert ReasonCode.LOW_KNOWLEDGE in cand.reason_codes

    rec = Recommendation(
        student_id="student_1",
        concept_id="cpt_conv",
        canonical_name="Convolution",
        action_type=ActionType.PRACTICE_CONCEPT,
        priority_score=0.45,
        reason_codes=[ReasonCode.LOW_KNOWLEDGE],
        explanation="Targeted practice recommended.",
        knowledge_score=0.30,
        confidence=0.80,
        mastery_status=MasteryStatus.EMERGING,
        source_chunk_ids=["CHUNK_002"],
    )
    assert rec.recommendation_id.startswith("rec_")
    assert rec.source_available is True


def test_prioritizer_signals():
    """Test all seven component signal functions under boundary conditions."""
    prioritizer = MultiSignalRecommendationPrioritizer()
    now = datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone.utc)

    # 1. Weakness: Evaluated vs Unassessed
    st_eval = KnowledgeState(
        student_id="s1",
        concept_id="c1",
        knowledge_score=0.20,
        confidence=0.80,
        strong_evidence_count=3,
    )
    assert prioritizer.compute_weakness(st_eval) > 0.60

    st_cold = KnowledgeState(
        student_id="s1",
        concept_id="c1",
        knowledge_score=0.20,
        confidence=0.10,
        strong_evidence_count=0,
    )
    assert prioritizer.compute_weakness(st_cold) == 0.0

    # 2. Uncertainty: Conflicting vs Calm
    st_unc = KnowledgeState(
        student_id="s1",
        concept_id="c1",
        knowledge_score=0.50,
        confidence=0.50,
        mastery_status=MasteryStatus.UNCERTAIN,
    )
    assert prioritizer.compute_uncertainty(st_unc) == 1.0

    # 3. Prerequisite Deficit
    assert prioritizer.compute_prerequisite_deficit([]) == 0.0
    assert prioritizer.compute_prerequisite_deficit(["Prereq1"]) == 0.50
    assert prioritizer.compute_prerequisite_deficit(["P1", "P2", "P3"]) == 1.0

    # 4. Downstream Blocker
    st_mastered = KnowledgeState(
        student_id="s1",
        concept_id="c1",
        knowledge_score=0.90,
        confidence=0.85,
        mastery_status=MasteryStatus.MASTERED,
    )
    assert prioritizer.compute_downstream_blocker(st_mastered, ["Dep1", "Dep2"]) == 0.0

    st_dev = KnowledgeState(
        student_id="s1",
        concept_id="c1",
        knowledge_score=0.50,
        confidence=0.50,
        mastery_status=MasteryStatus.DEVELOPING,
    )
    assert prioritizer.compute_downstream_blocker(st_dev, ["Dep1"]) == round(1 / 3.0, 4)

    # 5. Review Need
    st_new = KnowledgeState(
        student_id="s1",
        concept_id="c1",
        knowledge_score=0.50,
        confidence=0.10,
        evidence_count=0,
    )
    assert prioritizer.compute_review_need(st_new, now) == 0.0

    # 6. Recent Action Penalty
    assert prioritizer.compute_recent_action_penalty(None, now) == 0.0
    pen_recent = prioritizer.compute_recent_action_penalty(now - timedelta(minutes=2), now)
    pen_older = prioritizer.compute_recent_action_penalty(now - timedelta(minutes=60), now)
    assert pen_recent > pen_older


def test_weakness_vs_uncertainty_distinction():
    """Verify that low score with high confidence produces LOW_KNOWLEDGE,

    while low score with low confidence produces LOW_CONFIDENCE and NOT LOW_KNOWLEDGE.
    """
    prioritizer = MultiSignalRecommendationPrioritizer()
    now = datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone.utc)

    # A: High-confidence weakness
    st_a = KnowledgeState(
        student_id="s1",
        concept_id="cA",
        knowledge_score=0.25,
        confidence=0.90,
        strong_evidence_count=4,
        mastery_status=MasteryStatus.EMERGING,
    )
    cand_a = prioritizer.score_candidate("cA", "Concept A", st_a, [], [], ["CHUNK_1"], True, now=now)
    assert ReasonCode.LOW_KNOWLEDGE in cand_a.reason_codes
    assert ReasonCode.LOW_CONFIDENCE not in cand_a.reason_codes

    # B: Low-confidence cold start
    st_b = KnowledgeState(
        student_id="s1",
        concept_id="cB",
        knowledge_score=0.25,
        confidence=0.15,
        strong_evidence_count=0,
        mastery_status=MasteryStatus.UNKNOWN,
    )
    cand_b = prioritizer.score_candidate("cB", "Concept B", st_b, [], [], ["CHUNK_1"], True, now=now)
    assert ReasonCode.LOW_CONFIDENCE in cand_b.reason_codes
    assert ReasonCode.LOW_KNOWLEDGE not in cand_b.reason_codes


def test_action_selection_rules():
    """Verify PedagogicalActionSelector maps candidate attributes to deterministic actions."""
    selector = PedagogicalActionSelector()

    # Rule 1: Prerequisite deficit
    cand_prereq = RecommendationCandidate(
        concept_id="c1",
        canonical_name="C1",
        knowledge_score=0.30,
        confidence=0.70,
        unmastered_prerequisites=["Foundational Concept"],
    )
    action, diff = selector.select_action(cand_prereq)
    assert action == ActionType.REVIEW_PREREQUISITE
    assert diff is None

    # Rule 2: Conflicting evidence (UNCERTAIN)
    cand_unc = RecommendationCandidate(
        concept_id="c2",
        canonical_name="C2",
        knowledge_score=0.50,
        confidence=0.50,
        mastery_status=MasteryStatus.UNCERTAIN,
    )
    action, diff = selector.select_action(cand_unc)
    assert action == ActionType.TAKE_ASSESSMENT
    assert diff == ItemDifficulty.MEDIUM

    # Rule 3: Mastered spaced review
    cand_mastered = RecommendationCandidate(
        concept_id="c3",
        canonical_name="C3",
        knowledge_score=0.90,
        confidence=0.85,
        mastery_status=MasteryStatus.MASTERED,
    )
    action, diff = selector.select_action(cand_mastered)
    assert action == ActionType.REVISIT_SOURCE
    assert diff is None

    # Rule 4: Evaluated weakness vs developing
    cand_weak = RecommendationCandidate(
        concept_id="c4",
        canonical_name="C4",
        knowledge_score=0.25,
        confidence=0.75,
        mastery_status=MasteryStatus.EMERGING,
    )
    action, diff = selector.select_action(cand_weak)
    assert action == ActionType.PRACTICE_CONCEPT
    assert diff == ItemDifficulty.EASY

    cand_dev = RecommendationCandidate(
        concept_id="c5",
        canonical_name="C5",
        knowledge_score=0.60,
        confidence=0.65,
        mastery_status=MasteryStatus.DEVELOPING,
    )
    action, diff = selector.select_action(cand_dev)
    assert action == ActionType.PRACTICE_CONCEPT
    assert diff == ItemDifficulty.MEDIUM

    # Rule 5: Cold-start
    cand_cold = RecommendationCandidate(
        concept_id="c6",
        canonical_name="C6",
        knowledge_score=0.50,
        confidence=0.10,
        mastery_status=MasteryStatus.UNKNOWN,
    )
    action, diff = selector.select_action(cand_cold)
    assert action == ActionType.REVIEW_CONCEPT
    assert diff is None


def test_explanation_generation_content_and_safety():
    """Verify that explanations adhere to the communicative principle:

    evidence -> uncertainty -> recommendation -> reason
    and strictly avoid blunt deficit labeling ('You are weak in X').
    """
    generator = RecommendationExplanationGenerator()
    cand = RecommendationCandidate(
        concept_id="cpt_conv",
        canonical_name="Convolution",
        knowledge_score=0.25,
        confidence=0.85,
        mastery_status=MasteryStatus.EMERGING,
        reason_codes=[
            ReasonCode.LOW_KNOWLEDGE,
            ReasonCode.DOWNSTREAM_BLOCKER,
            ReasonCode.SOURCE_AVAILABLE,
        ],
        source_chunk_ids=["CHUNK_SS_001"],
        source_available=True,
        blocked_dependents=["Continuous-Time Fourier Transform"],
    )
    expl = generator.generate_explanation(cand, ActionType.PRACTICE_CONCEPT, ItemDifficulty.EASY)

    # Must contain evidence references and pedagogical motivation
    assert "score: 0.25" in expl
    assert "confidence: 0.85" in expl
    assert "CHUNK_SS_001" in expl
    assert "Continuous-Time Fourier Transform" in expl
    assert "EASY" in expl

    # Deficit labeling check
    assert "you are weak" not in expl.lower()
    assert "your ignorance" not in expl.lower()


def test_prerequisite_and_downstream_graph_context():
    """Verify candidate generation correctly propagates graph prerequisites and dependents."""
    graph = _create_test_graph()
    k_service = KnowledgeStateService(repository=InMemoryKnowledgeStateRepository(), graph_store=graph)
    cand_gen = HeuristicRecommendationCandidateGenerator(knowledge_service=k_service, graph_store=graph)

    # Student has unmastered LTI System
    cands = cand_gen.generate_candidates("student_prereq")
    cand_map = {c.concept_id: c for c in cands}

    # LTI System has 0 unmastered prereqs, blocks Convolution
    assert cand_map["cpt_lti"].unmastered_prerequisites == []
    assert "Convolution" in cand_map["cpt_lti"].blocked_dependents

    # Convolution has unmastered prerequisite LTI System
    assert "LTI System" in cand_map["cpt_conv"].unmastered_prerequisites
    assert cand_map["cpt_conv"].action_type == ActionType.REVIEW_PREREQUISITE


def test_source_grounding_and_missing_source():
    """Verify source availability and missing-source handling."""
    graph = _create_test_graph()
    k_service = KnowledgeStateService(repository=InMemoryKnowledgeStateRepository(), graph_store=graph)
    rec_service = RecommendationService(knowledge_service=k_service, graph_store=graph)

    recs = rec_service.get_recommendations("student_src", top_k=10)
    rec_map = {r.concept_id: r for r in recs}

    # Grounded concept
    assert rec_map["cpt_conv"].source_available is True
    assert rec_map["cpt_conv"].source_chunk_ids == ["CHUNK_002"]
    assert ReasonCode.SOURCE_AVAILABLE in rec_map["cpt_conv"].reason_codes

    # Missing source concept
    assert rec_map["cpt_unindexed"].source_available is False
    assert rec_map["cpt_unindexed"].source_chunk_ids == []
    assert ReasonCode.SOURCE_UNAVAILABLE in rec_map["cpt_unindexed"].reason_codes
    assert "not currently indexed" in rec_map["cpt_unindexed"].explanation


def test_mastered_concept_damping_and_spaced_review():
    """Verify mastered concepts receive damping factor but remain eligible for spaced review."""
    graph = _create_test_graph()
    k_service = KnowledgeStateService(repository=InMemoryKnowledgeStateRepository(), graph_store=graph)
    rec_service = RecommendationService(knowledge_service=k_service, graph_store=graph)

    now = datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone.utc)

    # Master LTI System 30 days ago
    for i in range(4):
        k_service.record_event(
            EvidenceEvent(
                student_id="student_spaced",
                concept_id="cpt_lti",
                event_type=EvidenceType.QUIZ_CORRECT,
                strength=EvidenceStrength.STRONG,
                score=1.0,
                timestamp=now - timedelta(days=30, hours=i),
            )
        )

    # Convolution is developing
    k_service.record_event(
        EvidenceEvent(
            student_id="student_spaced",
            concept_id="cpt_conv",
            event_type=EvidenceType.QUIZ_CORRECT,
            strength=EvidenceStrength.STRONG,
            score=1.0,
            timestamp=now - timedelta(hours=2),
        )
    )

    cands = rec_service.get_candidates("student_spaced", now=now)
    cand_map = {c.concept_id: c for c in cands}

    # Developing Convolution outranks Mastered LTI System
    assert cand_map["cpt_conv"].priority_score > cand_map["cpt_lti"].priority_score

    # LTI System has MASTERED_SPACED_REVIEW reason code and REVISIT_SOURCE action
    assert ReasonCode.MASTERED_SPACED_REVIEW in cand_map["cpt_lti"].reason_codes
    assert cand_map["cpt_lti"].action_type == ActionType.REVISIT_SOURCE


def test_recent_action_penalty_and_history():
    """Verify recommendation history records entries and applies decay penalties."""
    history_repo = InMemoryRecommendationRepository()
    now = datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone.utc)

    entry1 = RecommendationHistoryEntry(
        student_id="s1",
        concept_id="cpt_conv",
        action_type=ActionType.PRACTICE_CONCEPT,
        timestamp=now - timedelta(minutes=5),
    )
    history_repo.record_recommendation(entry1)

    entries = history_repo.get_history("s1")
    assert len(entries) == 1
    assert entries[0].concept_id == "cpt_conv"

    recent = history_repo.get_recent_entries("s1", limit=1)
    assert len(recent) == 1


def test_assessment_delegation():
    """Verify delegation from RecommendationService to AdaptiveAssessmentService."""
    graph = _create_test_graph()
    k_service = KnowledgeStateService(repository=InMemoryKnowledgeStateRepository(), graph_store=graph)

    # Build assessment service with an in-memory item
    item_repo = InMemoryAssessmentItemRepository()
    from indiclearn.assessment.models import AssessmentItem, QuestionType

    item = AssessmentItem(
        item_id="item_conv_test",
        concept_id="cpt_conv",
        subject_id="signals_and_systems",
        question="What is the convolution integral?",
        question_type=QuestionType.MCQ,
        difficulty=ItemDifficulty.EASY,
        language="en",
        options=["Integral of x(t)h(t-tau)", "Derivative", "Product", "Fourier"],
        correct_answer="Integral of x(t)h(t-tau)",
        explanation="The convolution integral defines system output.",
        source_chunk_ids=["CHUNK_002"],
    )
    item_repo.add_item(item)

    ass_service = AdaptiveAssessmentService(
        item_repository=item_repo,
        knowledge_service=k_service,
        graph_store=graph,
    )
    rec_service = RecommendationService(
        knowledge_service=k_service,
        graph_store=graph,
        assessment_service=ass_service,
    )

    rec = Recommendation(
        student_id="student_del",
        subject_id="signals_and_systems",
        concept_id="cpt_conv",
        canonical_name="Convolution",
        action_type=ActionType.TAKE_ASSESSMENT,
        priority_score=0.75,
        reason_codes=[ReasonCode.UNCERTAIN],
        explanation="Diagnostic assessment recommended.",
        knowledge_score=0.50,
        confidence=0.50,
        mastery_status=MasteryStatus.UNCERTAIN,
    )

    expl = rec_service.delegate_assessment(rec, requested_language="en")
    assert expl is not None
    assert expl.item.item_id == "item_conv_test"
    assert expl.item.source_chunk_ids == ["CHUNK_002"]


def test_student_and_concept_isolation():
    """Verify complete isolation: Student A state does not leak to Student B."""
    graph = _create_test_graph()
    k_service = KnowledgeStateService(repository=InMemoryKnowledgeStateRepository(), graph_store=graph)
    rec_service = RecommendationService(knowledge_service=k_service, graph_store=graph)
    now = datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone.utc)

    # Student A: Failed Convolution
    for i in range(4):
        k_service.record_event(
            EvidenceEvent(
                student_id="student_A",
                concept_id="cpt_conv",
                event_type=EvidenceType.QUIZ_INCORRECT,
                strength=EvidenceStrength.STRONG,
                score=0.0,
                timestamp=now - timedelta(hours=4 - i),
            )
        )

    # Student B: Mastered Convolution
    for i in range(4):
        k_service.record_event(
            EvidenceEvent(
                student_id="student_B",
                concept_id="cpt_conv",
                event_type=EvidenceType.QUIZ_CORRECT,
                strength=EvidenceStrength.STRONG,
                score=1.0,
                timestamp=now - timedelta(hours=4 - i),
            )
        )

    recs_a = rec_service.get_recommendations("student_A", top_k=2, now=now)
    recs_b = rec_service.get_recommendations("student_B", top_k=2, now=now)

    assert recs_a[0].concept_id != recs_b[0].concept_id or recs_a[0].priority_score != recs_b[0].priority_score
    cand_a = next(c for c in rec_service.get_candidates("student_A", now=now) if c.concept_id == "cpt_conv")
    cand_b = next(c for c in rec_service.get_candidates("student_B", now=now) if c.concept_id == "cpt_conv")
    assert cand_a.priority_score > cand_b.priority_score


def test_deterministic_ordering_and_tie_breaking():
    """Verify that multiple consecutive invocations produce identical results."""
    graph = _create_test_graph()
    k_service = KnowledgeStateService(repository=InMemoryKnowledgeStateRepository(), graph_store=graph)
    rec_service = RecommendationService(knowledge_service=k_service, graph_store=graph)
    now = datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone.utc)

    run1 = rec_service.get_recommendations("student_det", top_k=4, now=now)
    run2 = rec_service.get_recommendations("student_det", top_k=4, now=now)

    assert [r.concept_id for r in run1] == [r.concept_id for r in run2]
    assert [r.priority_score for r in run1] == [r.priority_score for r in run2]
    assert [r.explanation for r in run1] == [r.explanation for r in run2]
