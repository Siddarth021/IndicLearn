"""Evaluation Runner for EXP-007: Adaptive Assessment Selection Benchmark (Phase 7A).

Validates the mathematical, behavioral, and architectural contracts of the
Adaptive Assessment Core under controlled, reproducible evidence scenarios.

Scientific Policy:
- Synthetic/controlled benchmark dataset.
- Does NOT claim real student learning gains, cognitive retention, or psychometric IRT validity.
- Strictly measures software and algorithmic properties: candidate selection consistency,
  multi-signal prioritization, difficulty adaptation, repeat avoidance, language preference,
  provenance integrity, evidence integration, and student isolation.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from indiclearn.assessment.candidate import HeuristicCandidateGenerator
from indiclearn.assessment.models import (
    AssessmentCandidate,
    ItemDifficulty,
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
    EvidenceEvent,
    EvidenceStrength,
    EvidenceType,
    MasteryStatus,
)
from indiclearn.student_model.repository import InMemoryKnowledgeStateRepository
from indiclearn.student_model.service import KnowledgeStateService

logger = logging.getLogger(__name__)


class EXP007Runner:
    """Automated evaluation runner for EXP-007 benchmark."""

    def __init__(
        self,
        eval_data_path: str | Path = "data/evaluation/exp007_assessment_eval.json",
        items_data_path: str | Path = "data/evaluation/signals_systems_items.json",
        gold_graph_path: str | Path = "data/evaluation/concept_graph_gold.json",
        output_results_path: str | Path = "data/evaluation/exp007_results.json",
    ) -> None:
        self.eval_data_path = Path(eval_data_path)
        self.items_data_path = Path(items_data_path)
        self.gold_graph_path = Path(gold_graph_path)
        self.output_results_path = Path(output_results_path)

        with open(self.eval_data_path, encoding="utf-8") as f:
            self.eval_data = json.load(f)

    def _build_test_graph(self) -> NetworkXGraphStore:
        """Construct graph store initialized with gold standard concepts and relations."""
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

    def run_all(self) -> dict[str, Any]:
        """Execute all 10 scenarios and evaluate invariants."""
        results: list[dict[str, Any]] = []
        total_invariants = 0
        passed_invariants = 0

        scenarios = self.eval_data.get("scenarios", [])

        for sc in scenarios:
            sc_id = sc["scenario_id"]
            handler = getattr(self, f"_run_{sc_id.lower()}", None)
            if handler is None:
                raise ValueError(f"No test handler implemented for scenario '{sc_id}'")

            sc_result = handler(sc)
            results.append(sc_result)

            for val in sc_result["invariant_checks"].values():
                total_invariants += 1
                if val:
                    passed_invariants += 1

        all_scenarios_passed = all(r["all_invariants_passed"] for r in results)
        summary = {
            "experiment_id": "EXP-007",
            "name": "Adaptive Assessment Selection Benchmark",
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "total_scenarios": len(results),
            "scenarios_passed": sum(1 for r in results if r["all_invariants_passed"]),
            "scenario_pass_rate": round(sum(1 for r in results if r["all_invariants_passed"]) / len(results), 4),
            "total_invariants_evaluated": total_invariants,
            "invariants_passed": passed_invariants,
            "invariant_pass_rate": round(passed_invariants / total_invariants, 4),
            "scientific_disclaimer": self.eval_data["metadata"]["scientific_disclaimer"],
            "scenario_details": results,
            "status": "COMPLETE" if all_scenarios_passed else "FAILED",
        }

        self.output_results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_results_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    def _run_scen_001_low_knowledge_prioritized(self, sc: dict[str, Any]) -> dict[str, Any]:
        graph = self._build_test_graph()
        repo = InMemoryKnowledgeStateRepository()
        k_service = KnowledgeStateService(repository=repo, graph_store=graph)

        base_time = datetime.now(timezone.utc) - timedelta(days=2)
        # Setup student with low score on Convolution (0.20) and high on LTI System (0.80)
        student_id = "student_s1"
        for _ in range(4):
            k_service.record_event(
                EvidenceEvent(
                    student_id=student_id,
                    concept_id="cpt_signals_and_systems_convolution",
                    subject_id="signals_and_systems",
                    event_type=EvidenceType.QUIZ_INCORRECT,
                    strength=EvidenceStrength.STRONG,
                    score=0.10,
                    timestamp=base_time,
                )
            )
            k_service.record_event(
                EvidenceEvent(
                    student_id=student_id,
                    concept_id="cpt_signals_and_systems_lti_system",
                    subject_id="signals_and_systems",
                    event_type=EvidenceType.QUIZ_CORRECT,
                    strength=EvidenceStrength.STRONG,
                    score=0.95,
                    timestamp=base_time,
                )
            )

        prioritizer = MultiSignalCandidatePrioritizer()
        gen = HeuristicCandidateGenerator(
            graph_store=graph,
            knowledge_service=k_service,
            prioritizer=prioritizer,
            exclude_mastered=False,
        )
        candidates = gen.generate_candidates(student_id=student_id)

        cand_map = {c.concept_id: c for c in candidates}
        cand_a = cand_map["cpt_signals_and_systems_convolution"]
        cand_b = cand_map["cpt_signals_and_systems_lti_system"]

        p_greater = cand_a.priority_score > cand_b.priority_score
        rank_a = candidates.index(cand_a)
        rank_b = candidates.index(cand_b)
        a_ranked_above_b = rank_a < rank_b

        invariants = {
            "priority_a_greater_than_b": bool(p_greater),
            "concept_a_ranked_above_b": bool(a_ranked_above_b),
        }
        return {
            "scenario_id": sc["scenario_id"],
            "name": sc["name"],
            "all_invariants_passed": all(invariants.values()),
            "priority_a": cand_a.priority_score,
            "priority_b": cand_b.priority_score,
            "invariant_checks": invariants,
        }

    def _run_scen_002_uncertainty_handling(self, sc: dict[str, Any]) -> dict[str, Any]:
        graph = self._build_test_graph()
        repo = InMemoryKnowledgeStateRepository()
        k_service = KnowledgeStateService(repository=repo, graph_store=graph)

        student_id = "student_s2"
        # Concept A: Sampling theorem (cold start, no assessments -> C=0.10)
        # Concept B: Convolution (4 assessments -> score=0.20, C=0.74)
        base_time = datetime.now(timezone.utc) - timedelta(days=2)
        for _ in range(4):
            k_service.record_event(
                EvidenceEvent(
                    student_id=student_id,
                    concept_id="cpt_signals_and_systems_convolution",
                    subject_id="signals_and_systems",
                    event_type=EvidenceType.QUIZ_INCORRECT,
                    strength=EvidenceStrength.STRONG,
                    score=0.10,
                    timestamp=base_time,
                )
            )

        prioritizer = MultiSignalCandidatePrioritizer()
        gen = HeuristicCandidateGenerator(graph_store=graph, knowledge_service=k_service, prioritizer=prioritizer, exclude_mastered=False)
        candidates = gen.generate_candidates(student_id=student_id)
        cand_map = {c.concept_id: c for c in candidates}

        cand_a = cand_map["cpt_signals_and_systems_sampling_theorem"]
        cand_b = cand_map["cpt_signals_and_systems_convolution"]

        uncertainty_a_high = cand_a.uncertainty >= 0.70
        weakness_b_high = cand_b.weakness >= 0.60
        distinct_reasons = any("uncertainty" in r.lower() for r in cand_a.reasons) and any("weakness" in r.lower() or "score" in r.lower() for r in cand_b.reasons)

        invariants = {
            "uncertainty_a_high": bool(uncertainty_a_high),
            "weakness_b_high": bool(weakness_b_high),
            "distinct_reasons": bool(distinct_reasons),
        }
        return {
            "scenario_id": sc["scenario_id"],
            "name": sc["name"],
            "all_invariants_passed": all(invariants.values()),
            "cand_a_uncertainty": cand_a.uncertainty,
            "cand_b_weakness": cand_b.weakness,
            "invariant_checks": invariants,
        }

    def _run_scen_003_prerequisite_relevance(self, sc: dict[str, Any]) -> dict[str, Any]:
        graph = self._build_test_graph()
        repo = InMemoryKnowledgeStateRepository()
        k_service = KnowledgeStateService(repository=repo, graph_store=graph)

        student_id = "student_s3"
        # Concept B (Convolution) is unmastered
        k_service.record_event(
            EvidenceEvent(
                student_id=student_id,
                concept_id="cpt_signals_and_systems_convolution",
                subject_id="signals_and_systems",
                event_type=EvidenceType.QUIZ_INCORRECT,
                strength=EvidenceStrength.STRONG,
                score=0.20,
            )
        )

        prioritizer = MultiSignalCandidatePrioritizer()
        gen = HeuristicCandidateGenerator(graph_store=graph, knowledge_service=k_service, prioritizer=prioritizer, exclude_mastered=False)
        candidates = gen.generate_candidates(student_id=student_id)
        cand_map = {c.concept_id: c for c in candidates}

        # LTI System is prerequisite of Convolution
        cand_prereq = cand_map["cpt_signals_and_systems_lti_system"]
        # Linearity has no unmastered dependents
        cand_neutral = cand_map["cpt_signals_and_systems_linearity"]

        prereq_greater = cand_prereq.prerequisite_relevance > cand_neutral.prerequisite_relevance
        reason_present = any("prerequisite" in r.lower() and "convolution" in r.lower() for r in cand_prereq.reasons)

        invariants = {
            "prereq_score_greater_than_neutral": bool(prereq_greater),
            "prereq_reasons_contain_dependent": bool(reason_present),
        }
        return {
            "scenario_id": sc["scenario_id"],
            "name": sc["name"],
            "all_invariants_passed": all(invariants.values()),
            "prereq_score": cand_prereq.prerequisite_relevance,
            "neutral_score": cand_neutral.prerequisite_relevance,
            "invariant_checks": invariants,
        }

    def _run_scen_004_coverage_advantage(self, sc: dict[str, Any]) -> dict[str, Any]:
        graph = self._build_test_graph()
        repo = InMemoryKnowledgeStateRepository()
        k_service = KnowledgeStateService(repository=repo, graph_store=graph)

        student_id = "student_s4"
        now = datetime.now(timezone.utc)
        recent_time = now - timedelta(minutes=5)

        # Concept B assessed 5 times recently
        for _ in range(5):
            k_service.record_event(
                EvidenceEvent(
                    student_id=student_id,
                    concept_id="cpt_signals_and_systems_convolution",
                    subject_id="signals_and_systems",
                    event_type=EvidenceType.QUIZ_CORRECT,
                    strength=EvidenceStrength.STRONG,
                    score=0.50,
                    timestamp=recent_time,
                )
            )

        prioritizer = MultiSignalCandidatePrioritizer()
        gen = HeuristicCandidateGenerator(graph_store=graph, knowledge_service=k_service, prioritizer=prioritizer, exclude_mastered=False)
        candidates = gen.generate_candidates(student_id=student_id, now=now)
        cand_map = {c.concept_id: c for c in candidates}

        cand_unassessed = cand_map["cpt_signals_and_systems_sampling_theorem"]
        cand_recent = cand_map["cpt_signals_and_systems_convolution"]

        cov_greater = cand_unassessed.coverage_need > cand_recent.coverage_need
        recency_penalty_applied = cand_recent.recency_penalty > 0.0
        priority_greater = cand_unassessed.priority_score > cand_recent.priority_score

        invariants = {
            "coverage_need_a_greater_than_b": bool(cov_greater),
            "recency_penalty_b_applied": bool(recency_penalty_applied),
            "priority_a_greater_than_b": bool(priority_greater),
        }
        return {
            "scenario_id": sc["scenario_id"],
            "name": sc["name"],
            "all_invariants_passed": all(invariants.values()),
            "coverage_a": cand_unassessed.coverage_need,
            "coverage_b": cand_recent.coverage_need,
            "recency_b": cand_recent.recency_penalty,
            "invariant_checks": invariants,
        }

    def _run_scen_005_difficulty_adaptation(self, sc: dict[str, Any]) -> dict[str, Any]:
        selector = DeterministicItemSelector()

        # Weak candidate
        cand_weak = AssessmentCandidate(
            concept_id="cpt_signals_and_systems_convolution",
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

        # Developing candidate
        cand_dev = AssessmentCandidate(
            concept_id="cpt_signals_and_systems_convolution",
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

        # Mastered candidate
        cand_mast = AssessmentCandidate(
            concept_id="cpt_signals_and_systems_convolution",
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

        all_match = (
            diff_weak == ItemDifficulty.EASY
            and diff_dev == ItemDifficulty.MEDIUM
            and diff_mast == ItemDifficulty.HARD
        )

        invariants = {
            "all_difficulty_tiers_match": bool(all_match),
        }
        return {
            "scenario_id": sc["scenario_id"],
            "name": sc["name"],
            "all_invariants_passed": all(invariants.values()),
            "weak_diff": diff_weak.value,
            "dev_diff": diff_dev.value,
            "mast_diff": diff_mast.value,
            "invariant_checks": invariants,
        }

    def _run_scen_006_repeat_avoidance(self, sc: dict[str, Any]) -> dict[str, Any]:
        item_repo = InMemoryAssessmentItemRepository.from_json_file(self.items_data_path)
        selector = DeterministicItemSelector()

        cand = AssessmentCandidate(
            concept_id="cpt_signals_and_systems_convolution",
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

        all_items = item_repo.list_items()
        item1, _expl1 = selector.select_item(cand, all_items, requested_language="en", seen_item_ids=set())
        assert item1 is not None

        # Next selection with item1 marked seen
        item2, expl2 = selector.select_item(cand, all_items, requested_language="en", seen_item_ids={item1.item_id})
        assert item2 is not None

        differs = item2.item_id != item1.item_id
        repeat_reason = any("repeat avoidance" in r.lower() for r in expl2.item_reasons)

        invariants = {
            "second_item_differs_from_first": bool(differs),
            "repeat_avoidance_reason_present": bool(repeat_reason),
        }
        return {
            "scenario_id": sc["scenario_id"],
            "name": sc["name"],
            "all_invariants_passed": all(invariants.values()),
            "item1_id": item1.item_id,
            "item2_id": item2.item_id,
            "invariant_checks": invariants,
        }

    def _run_scen_007_language_preference(self, sc: dict[str, Any]) -> dict[str, Any]:
        item_repo = InMemoryAssessmentItemRepository.from_json_file(self.items_data_path)
        selector = DeterministicItemSelector()

        cand = AssessmentCandidate(
            concept_id="cpt_signals_and_systems_convolution",
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

        all_items = item_repo.list_items()
        item, _expl = selector.select_item(cand, all_items, requested_language="te")
        assert item is not None

        lang_match = item.language == "te"

        invariants = {
            "selected_item_matches_language": bool(lang_match),
        }
        return {
            "scenario_id": sc["scenario_id"],
            "name": sc["name"],
            "all_invariants_passed": all(invariants.values()),
            "selected_item_id": item.item_id,
            "selected_language": item.language,
            "invariant_checks": invariants,
        }

    def _run_scen_008_provenance_integrity(self, sc: dict[str, Any]) -> dict[str, Any]:
        item_repo = InMemoryAssessmentItemRepository.from_json_file(self.items_data_path)
        items = item_repo.list_items()

        has_provenance = all(len(it.source_chunk_ids) > 0 for it in items)
        all_chunks_valid = all(
            all(cid.startswith("CHUNK_SS_") for cid in it.source_chunk_ids)
            for it in items
        )

        invariants = {
            "has_provenance_chunks": bool(has_provenance),
            "chunks_belong_to_corpus": bool(all_chunks_valid),
        }
        return {
            "scenario_id": sc["scenario_id"],
            "name": sc["name"],
            "all_invariants_passed": all(invariants.values()),
            "total_items_checked": len(items),
            "invariant_checks": invariants,
        }

    def _run_scen_009_evidence_integration(self, sc: dict[str, Any]) -> dict[str, Any]:
        graph = self._build_test_graph()
        item_repo = InMemoryAssessmentItemRepository.from_json_file(self.items_data_path)
        repo = InMemoryKnowledgeStateRepository()
        k_service = KnowledgeStateService(repository=repo, graph_store=graph)
        service = AdaptiveAssessmentService(
            item_repository=item_repo,
            knowledge_service=k_service,
            graph_store=graph,
        )

        student_id = "student_s9"
        session, expl = service.start_session(student_id=student_id)
        assert expl is not None
        first_item = expl.item

        # 1. Submit correct answer
        _resp1, state1, next_expl1 = service.submit_response(
            session_id=session.session_id,
            item_id=first_item.item_id,
            selected_answer=first_item.correct_answer,
        )
        correct_increased = state1.knowledge_score > 0.50 and state1.confidence > 0.10

        # 2. Submit incorrect answer on second item
        assert next_expl1 is not None
        second_item = next_expl1.item
        prev_score = state1.knowledge_score

        # Pick wrong answer
        wrong_choice = "D" if second_item.correct_answer != "D" else "A"
        _resp2, state2, _ = service.submit_response(
            session_id=session.session_id,
            item_id=second_item.item_id,
            selected_answer=wrong_choice,
        )

        if second_item.concept_id == first_item.concept_id:
            score_decreased = state2.knowledge_score < prev_score
        else:
            score_decreased = state2.knowledge_score < 0.50

        # Check evidence history source
        logged_source = any(
            ev.summary and "adaptive_assessment_service" in ev.summary.lower()
            for ev in state2.evidence_history
        ) or any(
            e.source == "adaptive_assessment_service"
            for e in repo.get_events(student_id)
        )

        invariants = {
            "correct_answer_increases_score": bool(correct_increased),
            "incorrect_answer_decreases_score": bool(score_decreased),
            "event_logged_in_history": bool(logged_source),
        }
        return {
            "scenario_id": sc["scenario_id"],
            "name": sc["name"],
            "all_invariants_passed": all(invariants.values()),
            "score_after_correct": state1.knowledge_score,
            "score_after_incorrect": state2.knowledge_score,
            "invariant_checks": invariants,
        }

    def _run_scen_010_student_isolation(self, sc: dict[str, Any]) -> dict[str, Any]:
        graph = self._build_test_graph()
        item_repo = InMemoryAssessmentItemRepository.from_json_file(self.items_data_path)
        repo = InMemoryKnowledgeStateRepository()
        k_service = KnowledgeStateService(repository=repo, graph_store=graph)
        service = AdaptiveAssessmentService(
            item_repository=item_repo,
            knowledge_service=k_service,
            graph_store=graph,
        )

        alice_id = sc["alice_id"]
        bob_id = sc["bob_id"]

        # Alice completes an assessment session
        session_a, expl_a = service.start_session(student_id=alice_id)
        assert expl_a is not None
        service.submit_response(
            session_id=session_a.session_id,
            item_id=expl_a.item.item_id,
            selected_answer=expl_a.item.correct_answer,
        )

        # Bob gets candidates at cold start
        cands_bob = service.get_candidates(student_id=bob_id)
        seen_bob = service.session_manager.get_seen_item_ids(student_id=bob_id)

        # Check that Bob has 0 seen items and his candidates are all cold start
        bob_seen_empty = len(seen_bob) == 0
        bob_cold_start = all(c.assessment_count == 0 for c in cands_bob)

        invariants = {
            "alice_history_does_not_affect_bob_ranking": bool(bob_cold_start),
            "bob_seen_items_empty": bool(bob_seen_empty),
        }
        return {
            "scenario_id": sc["scenario_id"],
            "name": sc["name"],
            "all_invariants_passed": all(invariants.values()),
            "bob_seen_count": len(seen_bob),
            "invariant_checks": invariants,
        }
