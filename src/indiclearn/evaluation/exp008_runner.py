"""Evaluation Runner for EXP-008: Personalized Recommendation Benchmark (Phase 7B).

Validates the mathematical, behavioral, and architectural contracts of the
Personalized Recommendation & Remediation Planner under controlled scenarios.

Scientific Policy:
- Controlled synthetic benchmark dataset.
- Does NOT claim real student learning gains, exam score improvements, or validated pedagogical efficacy.
- Strictly evaluates software engineering invariants and algorithmic properties:
  multi-signal prioritization, uncertainty distinction, prerequisite remediation,
  downstream blocker awareness, recent-action damping, mastered concept deprioritization,
  conflicting evidence handling, source grounding and missing-source safety,
  assessment delegation, student isolation, and determinism.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from indiclearn.assessment.repository import InMemoryAssessmentItemRepository
from indiclearn.assessment.service import AdaptiveAssessmentService
from indiclearn.domain.concepts import Concept, ConceptRelation, RelationType
from indiclearn.knowledge_graph.networkx_store import NetworkXGraphStore
from indiclearn.recommendation.models import ActionType, ReasonCode
from indiclearn.recommendation.service import RecommendationService
from indiclearn.student_model.models import (
    EvidenceEvent,
    EvidenceStrength,
    EvidenceType,
    MasteryStatus,
)
from indiclearn.student_model.repository import InMemoryKnowledgeStateRepository
from indiclearn.student_model.service import KnowledgeStateService

logger = logging.getLogger(__name__)


class EXP008Runner:
    """Automated evaluation runner for EXP-008 benchmark."""

    def __init__(
        self,
        eval_data_path: str | Path = "data/evaluation/exp008_recommendation_eval.json",
        items_data_path: str | Path = "data/evaluation/signals_systems_items.json",
        gold_graph_path: str | Path = "data/evaluation/concept_graph_gold.json",
        output_results_path: str | Path = "data/evaluation/exp008_results.json",
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

    def _build_assessment_service(
        self,
        knowledge_service: KnowledgeStateService,
        graph: NetworkXGraphStore,
    ) -> AdaptiveAssessmentService:
        """Construct AdaptiveAssessmentService with controlled items."""
        if self.items_data_path.exists():
            item_repo = InMemoryAssessmentItemRepository.from_json_file(self.items_data_path)
        else:
            item_repo = InMemoryAssessmentItemRepository()
        return AdaptiveAssessmentService(
            item_repository=item_repo,
            knowledge_service=knowledge_service,
            graph_store=graph,
        )

    def run_all(self) -> dict[str, Any]:
        """Execute all 12 controlled scenarios and evaluate invariants."""
        results: list[dict[str, Any]] = []
        total_invariants = 0
        passed_invariants = 0

        scenarios = self.eval_data.get("scenarios", [])
        for scen in scenarios:
            scen_id = scen["scenario_id"]
            name = scen["name"]
            expected_invariants = scen["invariants"]

            eval_res = self._run_single_scenario(scen_id, expected_invariants)
            eval_res["scenario_id"] = scen_id
            eval_res["name"] = name
            results.append(eval_res)

            for passed in eval_res["invariant_results"].values():
                total_invariants += 1
                if passed:
                    passed_invariants += 1

        all_passed = (passed_invariants == total_invariants) and (total_invariants > 0)
        overall_compliance = round(passed_invariants / total_invariants, 4) if total_invariants > 0 else 0.0

        output = {
            "metadata": {
                "experiment_id": "EXP-008",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_scenarios": len(results),
                "total_invariants": total_invariants,
                "passed_invariants": passed_invariants,
                "overall_compliance_rate": overall_compliance,
                "status": "PASS" if all_passed else "FAIL",
                "scientific_disclaimer": self.eval_data["metadata"]["scientific_disclaimer"],
            },
            "scenarios": results,
        }

        self.output_results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_results_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        return output

    def _run_single_scenario(
        self,
        scenario_id: str,
        expected_invariants: dict[str, bool],
    ) -> dict[str, Any]:
        """Dispatch scenario by ID."""
        graph = self._build_test_graph()
        now = datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone.utc)

        if scenario_id == "SCEN_001_HIGH_CONFIDENCE_WEAKNESS":
            return self._run_scen_001(graph, expected_invariants, now)
        elif scenario_id == "SCEN_002_LOW_CONFIDENCE_UNCERTAINTY":
            return self._run_scen_002(graph, expected_invariants, now)
        elif scenario_id == "SCEN_003_PREREQUISITE_REMEDIATION":
            return self._run_scen_003(graph, expected_invariants, now)
        elif scenario_id == "SCEN_004_DOWNSTREAM_RELEVANCE":
            return self._run_scen_004(graph, expected_invariants, now)
        elif scenario_id == "SCEN_005_RECENT_ACTION_PENALTY":
            return self._run_scen_005(graph, expected_invariants, now)
        elif scenario_id == "SCEN_006_MASTERED_CONCEPT_DEPRIORITIZATION":
            return self._run_scen_006(graph, expected_invariants, now)
        elif scenario_id == "SCEN_007_CONFLICTING_EVIDENCE":
            return self._run_scen_007(graph, expected_invariants, now)
        elif scenario_id == "SCEN_008_SOURCE_GROUNDING":
            return self._run_scen_008(graph, expected_invariants, now)
        elif scenario_id == "SCEN_009_MISSING_SOURCE_SAFETY":
            return self._run_scen_009(graph, expected_invariants, now)
        elif scenario_id == "SCEN_010_ASSESSMENT_DELEGATION":
            return self._run_scen_010(graph, expected_invariants, now)
        elif scenario_id == "SCEN_011_STUDENT_ISOLATION":
            return self._run_scen_011(graph, expected_invariants, now)
        elif scenario_id == "SCEN_012_DETERMINISTIC_RANKING":
            return self._run_scen_012(graph, expected_invariants, now)
        else:
            raise ValueError(f"Unknown scenario ID: {scenario_id}")

    def _run_scen_001(
        self, graph: NetworkXGraphStore, expected_invariants: dict[str, bool], now: datetime
    ) -> dict[str, Any]:
        """SCEN_001: High-Confidence Weakness Triggers Strong Remediation."""
        k_service = KnowledgeStateService(repository=InMemoryKnowledgeStateRepository(), graph_store=graph)
        rec_service = RecommendationService(knowledge_service=k_service, graph_store=graph)

        # Prerequisite LTI System is mastered
        cpt_lti = "cpt_signals_and_systems_lti_system"
        for i in range(4):
            k_service.record_event(
                EvidenceEvent(
                    student_id="student_scen1",
                    concept_id=cpt_lti,
                    event_type=EvidenceType.QUIZ_CORRECT,
                    strength=EvidenceStrength.STRONG,
                    score=1.0,
                    timestamp=now - timedelta(hours=8 - i),
                )
            )

        # 4 consecutive incorrect quiz events for Convolution
        cpt_id = "cpt_signals_and_systems_convolution"
        for i in range(4):
            k_service.record_event(
                EvidenceEvent(
                    student_id="student_scen1",
                    concept_id=cpt_id,
                    event_type=EvidenceType.QUIZ_INCORRECT,
                    strength=EvidenceStrength.STRONG,
                    score=0.0,
                    timestamp=now - timedelta(hours=4 - i),
                )
            )

        cands = rec_service.get_candidates("student_scen1", now=now)
        conv_cand = next(c for c in cands if c.concept_id == cpt_id)

        inv_results = {
            "priority_high": conv_cand.priority_score >= 0.30,
            "reason_has_low_knowledge": ReasonCode.LOW_KNOWLEDGE in conv_cand.reason_codes,
            "action_is_remediation": conv_cand.action_type in {
                ActionType.PRACTICE_CONCEPT,
                ActionType.REVIEW_CONCEPT,
            },
        }

        return {
            "evaluated_concept": cpt_id,
            "knowledge_score": conv_cand.knowledge_score,
            "confidence": conv_cand.confidence,
            "priority_score": conv_cand.priority_score,
            "action_type": conv_cand.action_type.value,
            "reason_codes": [r.value for r in conv_cand.reason_codes],
            "invariant_results": inv_results,
            "passed": all(inv_results.values()),
        }

    def _run_scen_002(
        self, graph: NetworkXGraphStore, expected_invariants: dict[str, bool], now: datetime
    ) -> dict[str, Any]:
        """SCEN_002: Low-Confidence Uncertainty Distinguished from Weakness."""
        k_service = KnowledgeStateService(repository=InMemoryKnowledgeStateRepository(), graph_store=graph)
        rec_service = RecommendationService(knowledge_service=k_service, graph_store=graph)

        cpt_id = "cpt_signals_and_systems_sampling_theorem"
        # Zero strong assessments; default initial belief
        cands = rec_service.get_candidates("student_scen2", now=now)
        samp_cand = next(c for c in cands if c.concept_id == cpt_id)

        inv_results = {
            "reason_has_low_confidence": ReasonCode.LOW_CONFIDENCE in samp_cand.reason_codes,
            "reason_lacks_low_knowledge": ReasonCode.LOW_KNOWLEDGE not in samp_cand.reason_codes,
            "action_is_initial_review": samp_cand.action_type in {
                ActionType.REVIEW_CONCEPT,
                ActionType.TAKE_ASSESSMENT,
            },
        }

        return {
            "evaluated_concept": cpt_id,
            "knowledge_score": samp_cand.knowledge_score,
            "confidence": samp_cand.confidence,
            "action_type": samp_cand.action_type.value,
            "reason_codes": [r.value for r in samp_cand.reason_codes],
            "invariant_results": inv_results,
            "passed": all(inv_results.values()),
        }

    def _run_scen_003(
        self, graph: NetworkXGraphStore, expected_invariants: dict[str, bool], now: datetime
    ) -> dict[str, Any]:
        """SCEN_003: Prerequisite Remediation Unblocks Downstream Learning."""
        k_service = KnowledgeStateService(repository=InMemoryKnowledgeStateRepository(), graph_store=graph)
        rec_service = RecommendationService(knowledge_service=k_service, graph_store=graph)

        # LTI System is unmastered prerequisite of Convolution
        cpt_lti = "cpt_signals_and_systems_lti_system"
        cpt_conv = "cpt_signals_and_systems_convolution"

        # LTI: assessed with 2 incorrect quizzes (weakness)
        for i in range(2):
            k_service.record_event(
                EvidenceEvent(
                    student_id="student_scen3",
                    concept_id=cpt_lti,
                    event_type=EvidenceType.QUIZ_INCORRECT,
                    strength=EvidenceStrength.STRONG,
                    score=0.0,
                    timestamp=now - timedelta(hours=3 - i),
                )
            )

        cands = rec_service.get_candidates("student_scen3", now=now)
        conv_cand = next(c for c in cands if c.concept_id == cpt_conv)
        lti_cand = next(c for c in cands if c.concept_id == cpt_lti)

        inv_results = {
            "downstream_concept_action_review_prerequisite": conv_cand.action_type == ActionType.REVIEW_PREREQUISITE,
            "upstream_prerequisite_has_downstream_blocker": ReasonCode.DOWNSTREAM_BLOCKER in lti_cand.reason_codes,
            "prerequisite_deficit_reason_assigned": ReasonCode.PREREQUISITE_DEFICIT in conv_cand.reason_codes,
        }

        return {
            "conv_action": conv_cand.action_type.value,
            "conv_unmastered_prereqs": conv_cand.unmastered_prerequisites,
            "lti_reason_codes": [r.value for r in lti_cand.reason_codes],
            "lti_blocked_dependents": lti_cand.blocked_dependents,
            "invariant_results": inv_results,
            "passed": all(inv_results.values()),
        }

    def _run_scen_004(
        self, graph: NetworkXGraphStore, expected_invariants: dict[str, bool], now: datetime
    ) -> dict[str, Any]:
        """SCEN_004: Downstream Blocker Weight Reflects Curriculum Impact."""
        k_service = KnowledgeStateService(repository=InMemoryKnowledgeStateRepository(), graph_store=graph)
        rec_service = RecommendationService(knowledge_service=k_service, graph_store=graph)

        # In the gold graph:
        # LTI System is PREREQUISITE_OF -> Convolution (1 dependent)
        # Energy Signal has 0 outgoing prerequisite edges (0 dependents)
        cpt_lti = "cpt_signals_and_systems_lti_system"
        cpt_energy = "cpt_signals_and_systems_energy_signal"

        cands = rec_service.get_candidates("student_scen4", now=now)
        cand_lti = next(c for c in cands if c.concept_id == cpt_lti)
        cand_energy = next(c for c in cands if c.concept_id == cpt_energy)

        inv_results = {
            "blocker_score_a_greater_than_b": cand_lti.downstream_blocker_score > cand_energy.downstream_blocker_score,
        }

        return {
            "lti_blocker_score": cand_lti.downstream_blocker_score,
            "energy_blocker_score": cand_energy.downstream_blocker_score,
            "invariant_results": inv_results,
            "passed": all(inv_results.values()),
        }

    def _run_scen_005(
        self, graph: NetworkXGraphStore, expected_invariants: dict[str, bool], now: datetime
    ) -> dict[str, Any]:
        """SCEN_005: Recent-Action Damping Prevents Repetitive Domination."""
        k_service = KnowledgeStateService(repository=InMemoryKnowledgeStateRepository(), graph_store=graph)
        rec_service = RecommendationService(knowledge_service=k_service, graph_store=graph)

        cpt_a = "cpt_signals_and_systems_convolution"
        cpt_b = "cpt_signals_and_systems_lti_system"

        # Give both concepts identical weak scores
        for cid in [cpt_a, cpt_b]:
            for i in range(3):
                k_service.record_event(
                    EvidenceEvent(
                        student_id="student_scen5",
                        concept_id=cid,
                        event_type=EvidenceType.QUIZ_INCORRECT,
                        strength=EvidenceStrength.STRONG,
                        score=0.0,
                        timestamp=now - timedelta(hours=3 - i),
                    )
                )

        # Concept A was acted on 2 minutes ago
        from indiclearn.recommendation.models import RecommendationHistoryEntry

        entry_a = RecommendationHistoryEntry(
            student_id="student_scen5",
            concept_id=cpt_a,
            action_type=ActionType.PRACTICE_CONCEPT,
            timestamp=now - timedelta(minutes=2),
        )
        rec_service.history_repo.record_recommendation(entry_a)

        # Generate candidates now
        cands = rec_service.get_candidates("student_scen5", now=now)
        cand_map = {c.concept_id: c for c in cands}

        inv_results = {
            "priority_b_greater_than_a": cand_map[cpt_b].priority_score > cand_map[cpt_a].priority_score,
            "concept_b_ranked_above_a": cands.index(cand_map[cpt_b]) < cands.index(cand_map[cpt_a]),
        }

        return {
            "priority_a": cand_map[cpt_a].priority_score,
            "priority_b": cand_map[cpt_b].priority_score,
            "penalty_a": cand_map[cpt_a].recent_action_penalty,
            "penalty_b": cand_map[cpt_b].recent_action_penalty,
            "invariant_results": inv_results,
            "passed": all(inv_results.values()),
        }

    def _run_scen_006(
        self, graph: NetworkXGraphStore, expected_invariants: dict[str, bool], now: datetime
    ) -> dict[str, Any]:
        """SCEN_006: Mastered Concepts Deprioritized Below Unresolved Topics."""
        k_service = KnowledgeStateService(repository=InMemoryKnowledgeStateRepository(), graph_store=graph)
        rec_service = RecommendationService(knowledge_service=k_service, graph_store=graph)

        cpt_mastered = "cpt_signals_and_systems_lti_system"
        cpt_developing = "cpt_signals_and_systems_convolution"

        # Mastered concept: 4 correct quizzes (S >= 0.85, C >= 0.80)
        for i in range(4):
            k_service.record_event(
                EvidenceEvent(
                    student_id="student_scen6",
                    concept_id=cpt_mastered,
                    event_type=EvidenceType.QUIZ_CORRECT,
                    strength=EvidenceStrength.STRONG,
                    score=1.0,
                    timestamp=now - timedelta(hours=4 - i),
                )
            )

        # Developing concept: 1 correct, 1 incorrect
        k_service.record_event(
            EvidenceEvent(
                student_id="student_scen6",
                concept_id=cpt_developing,
                event_type=EvidenceType.QUIZ_CORRECT,
                strength=EvidenceStrength.STRONG,
                score=1.0,
                timestamp=now - timedelta(hours=2),
            )
        )
        k_service.record_event(
            EvidenceEvent(
                student_id="student_scen6",
                concept_id=cpt_developing,
                event_type=EvidenceType.QUIZ_INCORRECT,
                strength=EvidenceStrength.STRONG,
                score=0.0,
                timestamp=now - timedelta(hours=1),
            )
        )

        cands = rec_service.get_candidates("student_scen6", now=now)
        cand_map = {c.concept_id: c for c in cands}

        inv_results = {
            "priority_b_greater_than_a": cand_map[cpt_developing].priority_score > cand_map[cpt_mastered].priority_score,
            "concept_b_ranked_above_a": cands.index(cand_map[cpt_developing]) < cands.index(cand_map[cpt_mastered]),
            "mastered_concept_damped": cand_map[cpt_mastered].mastery_status == MasteryStatus.MASTERED,
        }

        return {
            "mastered_priority": cand_map[cpt_mastered].priority_score,
            "developing_priority": cand_map[cpt_developing].priority_score,
            "invariant_results": inv_results,
            "passed": all(inv_results.values()),
        }

    def _run_scen_007(
        self, graph: NetworkXGraphStore, expected_invariants: dict[str, bool], now: datetime
    ) -> dict[str, Any]:
        """SCEN_007: Conflicting Evidence (UNCERTAIN) Triggers Diagnostic Action."""
        k_service = KnowledgeStateService(repository=InMemoryKnowledgeStateRepository(), graph_store=graph)
        rec_service = RecommendationService(knowledge_service=k_service, graph_store=graph)

        cpt_id = "cpt_signals_and_systems_impulse_response"
        # Alternating outcomes to trigger UNCERTAIN status
        events = [1.0, 0.0, 1.0, 0.0, 0.5]
        for i, sc in enumerate(events):
            k_service.record_event(
                EvidenceEvent(
                    student_id="student_scen7",
                    concept_id=cpt_id,
                    event_type=EvidenceType.QUIZ_CORRECT if sc > 0.5 else EvidenceType.QUIZ_INCORRECT,
                    strength=EvidenceStrength.STRONG,
                    score=sc,
                    timestamp=now - timedelta(hours=len(events) - i),
                )
            )

        cands = rec_service.get_candidates("student_scen7", now=now)
        cand = next(c for c in cands if c.concept_id == cpt_id)

        inv_results = {
            "reason_has_uncertain": ReasonCode.UNCERTAIN in cand.reason_codes,
            "action_is_diagnostic": cand.action_type in {
                ActionType.TAKE_ASSESSMENT,
                ActionType.REVIEW_CONCEPT,
            },
        }

        return {
            "concept_id": cpt_id,
            "mastery_status": cand.mastery_status.value,
            "action_type": cand.action_type.value,
            "reason_codes": [r.value for r in cand.reason_codes],
            "invariant_results": inv_results,
            "passed": all(inv_results.values()),
        }

    def _run_scen_008(
        self, graph: NetworkXGraphStore, expected_invariants: dict[str, bool], now: datetime
    ) -> dict[str, Any]:
        """SCEN_008: Review Recommendations Grounded in Curriculum Chunks."""
        k_service = KnowledgeStateService(repository=InMemoryKnowledgeStateRepository(), graph_store=graph)
        rec_service = RecommendationService(knowledge_service=k_service, graph_store=graph)

        cpt_id = "cpt_signals_and_systems_convolution"
        recs = rec_service.get_recommendations("student_scen8", top_k=5, now=now)
        conv_rec = next(r for r in recs if r.concept_id == cpt_id)

        inv_results = {
            "source_available_is_true": conv_rec.source_available is True,
            "source_chunks_preserved": len(conv_rec.source_chunk_ids) > 0 and "CHUNK_SS_001" in conv_rec.source_chunk_ids,
            "reason_has_source_available": ReasonCode.SOURCE_AVAILABLE in conv_rec.reason_codes,
        }

        return {
            "concept_id": cpt_id,
            "source_available": conv_rec.source_available,
            "source_chunk_ids": conv_rec.source_chunk_ids,
            "reason_codes": [r.value for r in conv_rec.reason_codes],
            "invariant_results": inv_results,
            "passed": all(inv_results.values()),
        }

    def _run_scen_009(
        self, graph: NetworkXGraphStore, expected_invariants: dict[str, bool], now: datetime
    ) -> dict[str, Any]:
        """SCEN_009: Missing Source Concepts Flag Explicit SOURCE_UNAVAILABLE."""
        # Add a concept with no source chunks to graph
        unindexed_cpt = Concept(
            concept_id="cpt_signals_and_systems_quantum_signal",
            subject_id="signals_and_systems",
            canonical_name="Quantum Signal",
            source_chunk_ids=[],
        )
        graph.add_concept(unindexed_cpt)

        k_service = KnowledgeStateService(repository=InMemoryKnowledgeStateRepository(), graph_store=graph)
        rec_service = RecommendationService(knowledge_service=k_service, graph_store=graph)

        cands = rec_service.get_candidates("student_scen9", now=now)
        cand = next(c for c in cands if c.concept_id == unindexed_cpt.concept_id)

        inv_results = {
            "source_available_is_false": cand.source_available is False,
            "source_chunks_empty": len(cand.source_chunk_ids) == 0,
            "reason_has_source_unavailable": ReasonCode.SOURCE_UNAVAILABLE in cand.reason_codes,
            "no_hallucinated_provenance": cand.source_chunk_ids == [],
        }

        return {
            "concept_id": unindexed_cpt.concept_id,
            "source_available": cand.source_available,
            "source_chunk_ids": cand.source_chunk_ids,
            "reason_codes": [r.value for r in cand.reason_codes],
            "invariant_results": inv_results,
            "passed": all(inv_results.values()),
        }

    def _run_scen_010(
        self, graph: NetworkXGraphStore, expected_invariants: dict[str, bool], now: datetime
    ) -> dict[str, Any]:
        """SCEN_010: Assessment Action Delegates to Adaptive Assessment Service."""
        k_service = KnowledgeStateService(repository=InMemoryKnowledgeStateRepository(), graph_store=graph)
        ass_service = self._build_assessment_service(k_service, graph)
        rec_service = RecommendationService(
            knowledge_service=k_service,
            graph_store=graph,
            assessment_service=ass_service,
        )

        # Create recommendation with TAKE_ASSESSMENT
        from indiclearn.recommendation.models import Recommendation

        rec = Recommendation(
            student_id="student_scen10",
            subject_id="signals_and_systems",
            concept_id="cpt_signals_and_systems_convolution",
            canonical_name="Convolution",
            action_type=ActionType.TAKE_ASSESSMENT,
            priority_score=0.85,
            reason_codes=[ReasonCode.UNCERTAIN],
            explanation="Diagnostic assessment recommended.",
            knowledge_score=0.50,
            confidence=0.50,
            mastery_status=MasteryStatus.UNCERTAIN,
        )

        explanation = rec_service.delegate_assessment(rec, requested_language="en", now=now)

        inv_results = {
            "delegation_succeeds": explanation is not None,
            "delegated_item_valid": explanation is not None and explanation.item is not None and len(explanation.item.question) > 0,
            "delegated_item_provenance_intact": explanation is not None and len(explanation.item.source_chunk_ids) > 0,
        }

        return {
            "delegation_succeeded": explanation is not None,
            "item_id": explanation.item.item_id if explanation else None,
            "item_question": explanation.item.question if explanation else None,
            "item_chunks": explanation.item.source_chunk_ids if explanation else None,
            "invariant_results": inv_results,
            "passed": all(inv_results.values()),
        }

    def _run_scen_011(
        self, graph: NetworkXGraphStore, expected_invariants: dict[str, bool], now: datetime
    ) -> dict[str, Any]:
        """SCEN_011: Student Recommendation Isolation."""
        k_service = KnowledgeStateService(repository=InMemoryKnowledgeStateRepository(), graph_store=graph)
        rec_service = RecommendationService(knowledge_service=k_service, graph_store=graph)

        cpt_conv = "cpt_signals_and_systems_convolution"

        # Student A is weak in Convolution (4 incorrect quizzes)
        for i in range(4):
            k_service.record_event(
                EvidenceEvent(
                    student_id="student_A",
                    concept_id=cpt_conv,
                    event_type=EvidenceType.QUIZ_INCORRECT,
                    strength=EvidenceStrength.STRONG,
                    score=0.0,
                    timestamp=now - timedelta(hours=4 - i),
                )
            )

        # Student B has mastered Convolution (4 correct quizzes)
        for i in range(4):
            k_service.record_event(
                EvidenceEvent(
                    student_id="student_B",
                    concept_id=cpt_conv,
                    event_type=EvidenceType.QUIZ_CORRECT,
                    strength=EvidenceStrength.STRONG,
                    score=1.0,
                    timestamp=now - timedelta(hours=4 - i),
                )
            )

        recs_a = rec_service.get_recommendations("student_A", top_k=1, now=now)
        recs_b = rec_service.get_recommendations("student_B", top_k=1, now=now)

        inv_results = {
            "student_a_top_convolution": recs_a[0].concept_id == cpt_conv,
            "student_b_top_not_convolution": recs_b[0].concept_id != cpt_conv,
            "student_isolation_preserved": recs_a[0].priority_score != recs_b[0].priority_score,
        }

        return {
            "student_a_top": recs_a[0].concept_id,
            "student_a_priority": recs_a[0].priority_score,
            "student_b_top": recs_b[0].concept_id,
            "student_b_priority": recs_b[0].priority_score,
            "invariant_results": inv_results,
            "passed": all(inv_results.values()),
        }

    def _run_scen_012(
        self, graph: NetworkXGraphStore, expected_invariants: dict[str, bool], now: datetime
    ) -> dict[str, Any]:
        """SCEN_012: Deterministic Ranking Invariant."""
        k_service = KnowledgeStateService(repository=InMemoryKnowledgeStateRepository(), graph_store=graph)
        rec_service = RecommendationService(knowledge_service=k_service, graph_store=graph)

        recs_run1 = rec_service.get_recommendations("student_scen12", top_k=5, now=now)
        recs_run2 = rec_service.get_recommendations("student_scen12", top_k=5, now=now)

        ids1 = [r.concept_id for r in recs_run1]
        ids2 = [r.concept_id for r in recs_run2]

        scores1 = [r.priority_score for r in recs_run1]
        scores2 = [r.priority_score for r in recs_run2]

        expls1 = [r.explanation for r in recs_run1]
        expls2 = [r.explanation for r in recs_run2]

        inv_results = {
            "exact_ranking_identical": ids1 == ids2,
            "priority_scores_identical": scores1 == scores2,
            "explanations_identical": expls1 == expls2,
        }

        return {
            "ranking_run1": ids1,
            "ranking_run2": ids2,
            "scores_run1": scores1,
            "scores_run2": scores2,
            "invariant_results": inv_results,
            "passed": all(inv_results.values()),
        }
