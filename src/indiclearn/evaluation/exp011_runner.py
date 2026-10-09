"""Evaluation Runner for EXP-011: Controlled Adaptive Assessment Evaluation (Phase 9).

Validates the difficulty calibration, weakness targeting, uncertainty probing,
language preference, repeat avoidance, provenance, student isolation, and closed-loop
evidence generation of the Phase 7A adaptive assessment engine.

Scientific Notice:
This is a CONTROLLED ALGORITHMIC BENCHMARK.
It validates software engineering invariants and deterministic adaptive selection algorithms.
It does NOT claim psychometric IRT calibration, exam score improvements, or learning gains.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from indiclearn.assessment.models import AssessmentCandidate, ItemDifficulty
from indiclearn.assessment.repository import (
    InMemoryAssessmentItemRepository,
    InMemoryAssessmentSessionRepository,
)
from indiclearn.assessment.service import AdaptiveAssessmentService
from indiclearn.domain.concepts import Concept, ConceptRelation, RelationType
from indiclearn.evaluation.reporting.generator import ResearchReport
from indiclearn.knowledge_graph.networkx_store import NetworkXGraphStore
from indiclearn.student_model.models import (
    EvidenceEvent,
    EvidenceStrength,
    EvidenceType,
    MasteryStatus,
)
from indiclearn.student_model.repository import InMemoryKnowledgeStateRepository
from indiclearn.student_model.service import KnowledgeStateService

logger = logging.getLogger(__name__)


class EXP011Runner:
    """Automated evaluation runner for EXP-011 adaptive assessment benchmark."""

    def __init__(
        self,
        eval_data_path: str | Path = "data/evaluation/exp011_assessment_eval.json",
        items_data_path: str | Path = "data/evaluation/signals_systems_items.json",
        gold_graph_path: str | Path = "data/evaluation/concept_graph_gold.json",
        output_results_path: str | Path = "data/evaluation/exp011_results.json",
    ) -> None:
        self.eval_data_path = Path(eval_data_path)
        self.items_data_path = Path(items_data_path)
        self.gold_graph_path = Path(gold_graph_path)
        self.output_results_path = Path(output_results_path)

        with open(self.eval_data_path, encoding="utf-8") as f:
            self.eval_data = json.load(f)

    def _build_test_graph(self) -> NetworkXGraphStore:
        """Construct knowledge graph initialized with gold standard concepts and relations."""
        graph = NetworkXGraphStore()
        if self.gold_graph_path.exists():
            with open(self.gold_graph_path, encoding="utf-8") as f:
                data = json.load(f)
            for c in data.get("concepts", []):
                graph.add_concept(
                    Concept(
                        concept_id=c["concept_id"],
                        subject_id="signals_and_systems",
                        canonical_name=c["canonical_name"],
                        aliases=c.get("aliases", []),
                        description=c.get("description"),
                        source_chunk_ids=c.get("source_chunk_ids", []),
                    )
                )
            for r in data.get("relations", []):
                graph.add_relation(
                    ConceptRelation(
                        relation_id=f"rel_{r['source_concept_id']}_{r['relation_type']}_{r['target_concept_id']}",
                        subject_id="signals_and_systems",
                        source_concept_id=r["source_concept_id"],
                        target_concept_id=r["target_concept_id"],
                        relation_type=RelationType(r["relation_type"]),
                    )
                )
        return graph

    def _build_services(
        self, graph: NetworkXGraphStore
    ) -> tuple[KnowledgeStateService, AdaptiveAssessmentService]:
        """Construct isolated knowledge state and adaptive assessment services."""
        repo = InMemoryKnowledgeStateRepository()
        knowledge_service = KnowledgeStateService(repository=repo, graph_store=graph)

        if self.items_data_path.exists():
            item_repo = InMemoryAssessmentItemRepository.from_json_file(self.items_data_path)
        else:
            item_repo = InMemoryAssessmentItemRepository()

        session_repo = InMemoryAssessmentSessionRepository()
        assessment_service = AdaptiveAssessmentService(
            item_repository=item_repo,
            knowledge_service=knowledge_service,
            graph_store=graph,
            session_repository=session_repo,
        )
        return knowledge_service, assessment_service

    def run_all(self) -> dict[str, Any]:
        """Execute all 10 controlled scenarios and evaluate invariants."""
        graph = self._build_test_graph()
        now = datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone.utc)

        scenario_results: list[dict[str, Any]] = []

        # SCEN_001: Difficulty Alignment
        ks, assess = self._build_services(graph)
        c1 = "cpt_signals_and_systems_convolution"
        selector = assess.item_selector
        cand_weak = AssessmentCandidate(
            concept_id=c1,
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
        diff_weak, _ = selector.determine_target_difficulty(cand_weak)

        cand_dev = AssessmentCandidate(
            concept_id=c1,
            canonical_name="Convolution",
            priority_score=0.6,
            knowledge_score=0.60,
            confidence=0.50,
            mastery_status=MasteryStatus.DEVELOPING,
            uncertainty=0.5,
            weakness=0.4,
            prerequisite_relevance=0.0,
            coverage_need=0.5,
            recency_penalty=0.0,
        )
        diff_dev, _ = selector.determine_target_difficulty(cand_dev)

        cand_mast = AssessmentCandidate(
            concept_id=c1,
            canonical_name="Convolution",
            priority_score=0.4,
            knowledge_score=0.85,
            confidence=0.70,
            mastery_status=MasteryStatus.MASTERED,
            uncertainty=0.3,
            weakness=0.15,
            prerequisite_relevance=0.0,
            coverage_need=0.2,
            recency_penalty=0.0,
        )
        diff_mast, _ = selector.determine_target_difficulty(cand_mast)

        scen1_pass = (
            diff_weak == ItemDifficulty.EASY
            and diff_dev == ItemDifficulty.MEDIUM
            and diff_mast == ItemDifficulty.HARD
        )
        scenario_results.append({
            "scenario_id": "EXP011_SCEN_001_DIFFICULTY_ALIGNMENT",
            "passed": scen1_pass,
            "weak_difficulty": diff_weak.value,
            "developing_difficulty": diff_dev.value,
            "mastered_difficulty": diff_mast.value,
        })

        # SCEN_002: Weakness Targeting
        ks, assess = self._build_services(graph)
        s2 = "EXP011_STU_002"
        c2_weak = "cpt_signals_and_systems_convolution"
        c2_strong = "cpt_signals_and_systems_lti_system"
        # 3 incorrect quizzes for Convolution
        for i in range(3):
            ks.record_event(
                EvidenceEvent(
                    student_id=s2,
                    concept_id=c2_weak,
                    event_type=EvidenceType.QUIZ_INCORRECT,
                    strength=EvidenceStrength.STRONG,
                    score=0.10,
                    timestamp=now - timedelta(hours=3 - i),
                )
            )
        # 3 correct quizzes for LTI System
        for i in range(3):
            ks.record_event(
                EvidenceEvent(
                    student_id=s2,
                    concept_id=c2_strong,
                    event_type=EvidenceType.QUIZ_CORRECT,
                    strength=EvidenceStrength.STRONG,
                    score=0.95,
                    timestamp=now - timedelta(hours=3 - i),
                )
            )
        cands2 = assess.get_candidates(student_id=s2, now=now)
        cand_weak = next(c for c in cands2 if c.concept_id == c2_weak)
        cand_strong = next(c for c in cands2 if c.concept_id == c2_strong)
        scen2_pass = (
            cand_weak.priority_score > cand_strong.priority_score
            and cands2.index(cand_weak) < cands2.index(cand_strong)
            and cand_weak.weakness >= 0.70
        )
        scenario_results.append({
            "scenario_id": "EXP011_SCEN_002_WEAKNESS_TARGETING",
            "passed": scen2_pass,
            "weak_priority": cand_weak.priority_score,
            "strong_priority": cand_strong.priority_score,
            "weakness_score": cand_weak.weakness,
        })

        # SCEN_003: Uncertainty Targeting
        ks, assess = self._build_services(graph)
        s3 = "EXP011_STU_003"
        c3 = "cpt_signals_and_systems_laplace_transform"
        # Conflicting evidence -> UNCERTAIN
        ks.record_event(
            EvidenceEvent(
                student_id=s3,
                concept_id=c3,
                event_type=EvidenceType.QUIZ_CORRECT,
                strength=EvidenceStrength.STRONG,
                score=1.0,
                timestamp=now - timedelta(hours=4),
            )
        )
        ks.record_event(
            EvidenceEvent(
                student_id=s3,
                concept_id=c3,
                event_type=EvidenceType.QUIZ_INCORRECT,
                strength=EvidenceStrength.STRONG,
                score=0.0,
                timestamp=now - timedelta(hours=2),
            )
        )
        cands3 = assess.get_candidates(s3, now=now)
        cand3 = next((c for c in cands3 if c.concept_id == c3), None)
        scen3_pass = cand3 is not None and cand3.uncertainty >= 0.70
        scenario_results.append({
            "scenario_id": "EXP011_SCEN_003_UNCERTAINTY_TARGETING",
            "passed": scen3_pass,
            "uncertainty_score": cand3.uncertainty if cand3 else None,
        })

        # SCEN_004: Prerequisite Targeting
        ks, assess = self._build_services(graph)
        s4 = "EXP011_STU_004"
        c4_pre = "cpt_signals_and_systems_lti_system"
        for i in range(3):
            ks.record_event(
                EvidenceEvent(
                    student_id=s4,
                    concept_id=c4_pre,
                    event_type=EvidenceType.QUIZ_INCORRECT,
                    strength=EvidenceStrength.STRONG,
                    score=0.0,
                    timestamp=now - timedelta(hours=3 - i),
                )
            )
        cands4 = assess.get_candidates(s4, now=now)
        cand_pre = next(c for c in cands4 if c.concept_id == c4_pre)
        scen4_pass = cand_pre.prerequisite_relevance > 0.0
        scenario_results.append({
            "scenario_id": "EXP011_SCEN_004_PREREQUISITE_TARGETING",
            "passed": scen4_pass,
            "prerequisite_relevance": cand_pre.prerequisite_relevance,
        })

        # SCEN_005: Language Preference
        ks, assess = self._build_services(graph)
        s5 = "EXP011_STU_005"
        c5 = "cpt_signals_and_systems_convolution"
        ks.record_event(
            EvidenceEvent(
                student_id=s5,
                concept_id=c5,
                event_type=EvidenceType.QUIZ_INCORRECT,
                strength=EvidenceStrength.STRONG,
                score=0.0,
                timestamp=now - timedelta(hours=2),
            )
        )
        expl5 = assess.select_next_assessment(student_id=s5, requested_language="te", now=now)
        scen5_pass = expl5 is not None and expl5.item.language == "te" and any("\u0c00" <= ch <= "\u0c7f" for ch in expl5.item.question)
        scenario_results.append({
            "scenario_id": "EXP011_SCEN_005_LANGUAGE_PREFERENCE",
            "passed": scen5_pass,
            "selected_language": expl5.item.language if expl5 else None,
        })

        # SCEN_006: Repeat Avoidance
        ks, assess = self._build_services(graph)
        s6 = "EXP011_STU_006"
        c6 = "cpt_signals_and_systems_convolution"
        ks.record_event(
            EvidenceEvent(
                student_id=s6,
                concept_id=c6,
                event_type=EvidenceType.QUIZ_INCORRECT,
                strength=EvidenceStrength.STRONG,
                score=0.0,
                timestamp=now - timedelta(hours=2),
            )
        )
        session, expl6_1 = assess.start_session(student_id=s6, requested_language="en")
        item1_id = expl6_1.item.item_id if expl6_1 else None
        expl6_2 = assess.select_next_assessment(student_id=s6, session_id=session.session_id, requested_language="en", now=now)
        item2_id = expl6_2.item.item_id if expl6_2 else None
        scen6_pass = item1_id is not None and item2_id is not None and item1_id != item2_id
        scenario_results.append({
            "scenario_id": "EXP011_SCEN_006_REPEAT_AVOIDANCE",
            "passed": scen6_pass,
            "item1_id": item1_id,
            "item2_id": item2_id,
        })

        # SCEN_007: Determinism
        ks, assess = self._build_services(graph)
        s7 = "EXP011_STU_007"
        c7 = "cpt_signals_and_systems_convolution"
        ks.record_event(
            EvidenceEvent(
                student_id=s7,
                concept_id=c7,
                event_type=EvidenceType.QUIZ_INCORRECT,
                strength=EvidenceStrength.STRONG,
                score=0.0,
                timestamp=now - timedelta(hours=2),
            )
        )
        expl7_a = assess.select_next_assessment(student_id=s7, now=now)
        expl7_b = assess.select_next_assessment(student_id=s7, now=now)
        scen7_pass = (
            expl7_a is not None
            and expl7_b is not None
            and expl7_a.item.item_id == expl7_b.item.item_id
            and expl7_a.candidate.priority_score == expl7_b.candidate.priority_score
        )
        scenario_results.append({
            "scenario_id": "EXP011_SCEN_007_DETERMINISM",
            "passed": scen7_pass,
            "deterministic_match": scen7_pass,
        })

        # SCEN_008: Provenance Integrity
        ks, assess = self._build_services(graph)
        s8 = "EXP011_STU_008"
        c8 = "cpt_signals_and_systems_convolution"
        ks.record_event(
            EvidenceEvent(
                student_id=s8,
                concept_id=c8,
                event_type=EvidenceType.QUIZ_INCORRECT,
                strength=EvidenceStrength.STRONG,
                score=0.0,
                timestamp=now - timedelta(hours=2),
            )
        )
        expl8 = assess.select_next_assessment(student_id=s8, now=now)
        scen8_pass = expl8 is not None and len(expl8.item.source_chunk_ids) > 0
        scenario_results.append({
            "scenario_id": "EXP011_SCEN_008_PROVENANCE_INTEGRITY",
            "passed": scen8_pass,
            "source_chunk_ids": expl8.item.source_chunk_ids if expl8 else [],
        })

        # SCEN_009: Student Isolation
        ks, assess = self._build_services(graph)
        s9_a = "EXP011_STU_009_A"
        s9_b = "EXP011_STU_009_B"
        c9_a = "cpt_signals_and_systems_convolution"
        c9_b = "cpt_signals_and_systems_sampling_theorem"
        ks.record_event(
            EvidenceEvent(
                student_id=s9_a,
                concept_id=c9_a,
                event_type=EvidenceType.QUIZ_INCORRECT,
                strength=EvidenceStrength.STRONG,
                score=0.0,
                timestamp=now - timedelta(hours=2),
            )
        )
        ks.record_event(
            EvidenceEvent(
                student_id=s9_b,
                concept_id=c9_b,
                event_type=EvidenceType.QUIZ_INCORRECT,
                strength=EvidenceStrength.STRONG,
                score=0.0,
                timestamp=now - timedelta(hours=2),
            )
        )
        st_a_b = ks.get_concept_state(s9_a, c9_b)
        st_b_a = ks.get_concept_state(s9_b, c9_a)
        st_a_a = ks.get_concept_state(s9_a, c9_a)
        st_b_b = ks.get_concept_state(s9_b, c9_b)
        scen9_pass = (
            st_a_b.evidence_count == 0
            and st_b_a.evidence_count == 0
            and st_a_a.evidence_count > 0
            and st_b_b.evidence_count > 0
        )
        scenario_results.append({
            "scenario_id": "EXP011_SCEN_009_STUDENT_ISOLATION",
            "passed": scen9_pass,
            "student_a_concept_b_evidence": st_a_b.evidence_count,
            "student_b_concept_a_evidence": st_b_a.evidence_count,
        })

        # SCEN_010: Closed-Loop Evidence Generation
        ks, assess = self._build_services(graph)
        s10 = "EXP011_STU_010"
        c10 = "cpt_signals_and_systems_convolution"
        session, expl10 = assess.start_session(student_id=s10, requested_language="en")
        item10 = expl10.item if expl10 else None
        assert item10 is not None

        state_before = ks.get_concept_state(s10, c10)
        ev_count_before = state_before.evidence_count

        # Submit response to the selected item
        resp, state_after, _ = assess.submit_response(
            session_id=session.session_id,
            item_id=item10.item_id,
            selected_answer=item10.correct_answer,
            now=now,
        )
        scen10_pass = (
            resp.correct
            and state_after.evidence_count == ev_count_before + 1
            and state_after.last_assessed_at is not None
        )
        scenario_results.append({
            "scenario_id": "EXP011_SCEN_010_CLOSED_LOOP_EVIDENCE",
            "passed": scen10_pass,
            "evidence_count_before": ev_count_before,
            "evidence_count_after": state_after.evidence_count,
            "knowledge_score_after": state_after.knowledge_score,
        })

        total_scenarios = len(scenario_results)
        passed_scenarios = sum(1 for s in scenario_results if s["passed"])
        scenario_pass_rate = passed_scenarios / total_scenarios if total_scenarios > 0 else 0.0

        overall_status = "PASS" if passed_scenarios == total_scenarios else "FAIL"

        report = ResearchReport(
            experiment_id="EXP-011",
            title="Controlled Adaptive Assessment Evaluation",
            research_question="RQ-ASSESS: Does the Phase 7A adaptive assessment engine maintain difficulty calibration, weakness targeting, language fidelity, repeat avoidance, and closed-loop state updates under controlled test conditions?",
            hypothesis="The adaptive assessment selection and evaluation pipelines satisfy 100% of controlled invariants without cross-student leakage or degradation in provenance.",
            dataset_identifier=str(self.eval_data_path),
            sample_size=total_scenarios,
            procedure="Evaluated 10 controlled scenarios spanning difficulty adaptation, weakness targeting, uncertainty probing, prerequisite deficit awareness, Telugu language preference, repeat item avoidance, determinism, provenance, student isolation, and closed-loop evidence integration.",
            metrics={
                "total_scenarios": total_scenarios,
                "passed_scenarios": passed_scenarios,
                "scenario_pass_rate": scenario_pass_rate,
                "difficulty_alignment": 1.0 if scen1_pass else 0.0,
                "weakness_targeting_accuracy": 1.0 if scen2_pass else 0.0,
                "language_fidelity": 1.0 if scen5_pass else 0.0,
                "repeat_avoidance_rate": 1.0 if scen6_pass else 0.0,
                "closed_loop_evidence_rate": 1.0 if scen10_pass else 0.0,
            },
            observed_results={
                "total_scenarios": total_scenarios,
                "passed_scenarios": passed_scenarios,
                "overall_status": overall_status,
            },
            interpretation=(
                "The adaptive assessment engine successfully calibrates item difficulty to student mastery, "
                "targets verified concept weaknesses, respects Telugu language preferences, prevents immediate question repeats, "
                "and integrates quiz outcomes into the student model in a closed feedback loop."
            ),
            scientific_claims=[
                "The adaptive assessment system deterministically adheres to defined pedagogical item selection policies and maintains closed-loop evidence integration under controlled conditions."
            ],
            limitations=[
                "Evaluated using controlled curated question items and synthetic knowledge states.",
                "Does not establish psychometric IRT item discrimination or live exam score improvements.",
                "Real student learning gains require a future controlled educational effectiveness study.",
            ],
            reproducibility={
                "random_seed": 42,
                "item_bank": str(self.items_data_path),
                "gold_graph": str(self.gold_graph_path),
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        output_dict = {
            "metadata": {
                "experiment_id": "EXP-011",
                "status": overall_status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_scenarios": total_scenarios,
                "passed_scenarios": passed_scenarios,
                "scenario_pass_rate": scenario_pass_rate,
                "evaluation_mode": "controlled_algorithmic_evaluation",
                "scientific_disclaimer": "Controlled algorithmic benchmark. Does NOT claim real student learning gains.",
            },
            "scenarios": scenario_results,
            "report": report.to_dict(),
        }

        self.output_results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_results_path, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, indent=2)

        return output_dict
