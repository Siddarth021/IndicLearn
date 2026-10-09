"""Evaluation Runner for EXP-006: Student Knowledge Modeling & Evidence-Based Personalization.

Validates the mathematical, behavioral, and architectural contracts of the Phase 6
Student Knowledge Model under controlled, reproducible evidence scenarios.

Scientific Policy:
- Synthetic/controlled benchmark dataset.
- Does NOT claim real student ground truth or cognitive predictive validity.
- Measures mathematical invariants: monotonicity, evidence strength dominance,
  weak evidence restraint, uncertainty gating, provenance completeness, isolation,
  and deterministic reproducibility.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from indiclearn.student_model.estimator import EvidenceBasedKnowledgeEstimator
from indiclearn.student_model.models import (
    EvidenceEvent,
    EvidenceStrength,
    EvidenceType,
    MasteryStatus,
)
from indiclearn.student_model.repository import InMemoryKnowledgeStateRepository
from indiclearn.student_model.service import KnowledgeStateService

logger = logging.getLogger(__name__)


class EXP006Runner:
    """Automated evaluation runner for EXP-006 benchmark."""

    def __init__(
        self,
        eval_data_path: str | Path = "data/evaluation/exp006_student_eval.json",
        output_results_path: str | Path = "data/evaluation/exp006_results.json",
    ) -> None:
        self.eval_data_path = Path(eval_data_path)
        self.output_results_path = Path(output_results_path)

        with open(self.eval_data_path, encoding="utf-8") as f:
            self.eval_data = json.load(f)

    def _convert_event(self, raw: dict[str, Any], student_id: str, concept_id: str, base_time: datetime) -> EvidenceEvent:
        days = raw.get("time_offset_days", 0)
        ev_time = base_time - timedelta(days=days)
        return EvidenceEvent(
            student_id=student_id,
            concept_id=concept_id,
            subject_id="signals_and_systems",
            event_type=EvidenceType(raw["event_type"]),
            strength=EvidenceStrength(raw["strength"]),
            score=raw["score"],
            timestamp=ev_time,
            source="exp006_runner",
            metadata={"raw": raw},
        )

    def run_all(self) -> dict[str, Any]:
        """Execute all benchmark scenarios and verify mathematical invariants."""
        scenarios = self.eval_data.get("scenarios", [])
        scenario_results: list[dict[str, Any]] = []
        total_invariants = 0
        passed_invariants = 0

        base_time = datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone.utc)

        for sc in scenarios:
            sc_id = sc["scenario_id"]
            name = sc["name"]
            student_id = sc["student_id"]
            concept_id = sc["concept_id"]
            raw_events = sc.get("events", [])
            invariants = sc.get("expected_invariants", {})

            service = KnowledgeStateService(
                repository=InMemoryKnowledgeStateRepository(),
                estimator=EvidenceBasedKnowledgeEstimator(),
            )

            # Record events step by step while recording trajectory
            score_trajectory: list[float] = []
            confidence_trajectory: list[float] = []

            # Initialize cold start if no events
            if not raw_events:
                initial = service.get_concept_state(student_id, concept_id)
                score_trajectory.append(initial.knowledge_score)
                confidence_trajectory.append(initial.confidence)
            else:
                for raw_ev in raw_events:
                    ev = self._convert_event(raw_ev, student_id, concept_id, base_time)
                    state = service.record_event(ev)
                    score_trajectory.append(state.knowledge_score)
                    confidence_trajectory.append(state.confidence)

            final_state = service.get_concept_state(student_id, concept_id)
            invariant_checks: dict[str, bool] = {}

            # 1. Monotonicity checks
            if invariants.get("monotonic_score_increase"):
                is_mono = all(score_trajectory[i] >= score_trajectory[i - 1] for i in range(1, len(score_trajectory)))
                invariant_checks["monotonic_score_increase"] = is_mono

            if invariants.get("monotonic_score_decrease"):
                is_mono = all(score_trajectory[i] <= score_trajectory[i - 1] for i in range(1, len(score_trajectory)))
                invariant_checks["monotonic_score_decrease"] = is_mono

            # 2. Score bounds checks
            if "final_min_score" in invariants:
                invariant_checks["final_min_score"] = final_state.knowledge_score >= invariants["final_min_score"]

            if "final_max_score" in invariants:
                invariant_checks["final_max_score"] = final_state.knowledge_score <= invariants["final_max_score"]

            if invariants.get("not_instantly_zeroed"):
                invariant_checks["not_instantly_zeroed"] = final_state.knowledge_score > 0.0

            if "final_min_confidence" in invariants:
                invariant_checks["final_min_confidence"] = final_state.confidence >= invariants["final_min_confidence"]

            if "max_confidence" in invariants:
                invariant_checks["max_confidence"] = final_state.confidence <= invariants["max_confidence"]

            if invariants.get("score_near_prior"):
                dev = abs(final_state.knowledge_score - 0.50)
                max_dev = invariants.get("max_score_deviation_from_prior", 0.05)
                invariant_checks["score_near_prior"] = dev <= max_dev

            if invariants.get("never_mastered"):
                invariant_checks["never_mastered"] = final_state.mastery_status != MasteryStatus.MASTERED

            if invariants.get("strong_evidence_dominance"):
                # Score was pulled down despite 4 prior weak events
                invariant_checks["strong_evidence_dominance"] = final_state.knowledge_score <= 0.40

            if "balanced_score_range" in invariants:
                low, high = invariants["balanced_score_range"]
                invariant_checks["balanced_score_range"] = low <= final_state.knowledge_score <= high

            if invariants.get("has_conflicting_flag"):
                invariant_checks["has_conflicting_flag"] = final_state.metadata.get("has_conflicting_evidence", False)

            if "expected_final_status" in invariants:
                invariant_checks["expected_final_status"] = (
                    final_state.mastery_status.value == invariants["expected_final_status"]
                )

            # 3. Isolation checks
            if "secondary_student_id" in sc:
                sec_id = sc["secondary_student_id"]
                sec_state = service.get_concept_state(sec_id, concept_id)
                invariant_checks["alice_status"] = (
                    final_state.mastery_status.value == invariants.get("alice_status")
                )
                invariant_checks["bob_status"] = (
                    sec_state.mastery_status.value == invariants.get("bob_status")
                )
                invariant_checks["bob_evidence_count"] = (
                    sec_state.evidence_count == invariants.get("bob_evidence_count", 0)
                )

            if "secondary_concept_id" in sc:
                sec_cid = sc["secondary_concept_id"]
                sec_cstate = service.get_concept_state(student_id, sec_cid)
                invariant_checks["target_concept_status"] = (
                    final_state.mastery_status.value == invariants.get("target_concept_status")
                )
                invariant_checks["secondary_concept_status"] = (
                    sec_cstate.mastery_status.value == invariants.get("secondary_concept_status")
                )
                invariant_checks["secondary_concept_evidence_count"] = (
                    sec_cstate.evidence_count == invariants.get("secondary_concept_evidence_count", 0)
                )

            # 4. Provenance and determinism checks
            if "evidence_history_length" in invariants:
                invariant_checks["evidence_history_length"] = (
                    len(final_state.evidence_history) == invariants["evidence_history_length"]
                )

            if invariants.get("deterministic_exact_match"):
                # Run duplicate pipeline and verify identical floating point values
                duplicate_service = KnowledgeStateService(
                    repository=InMemoryKnowledgeStateRepository(),
                    estimator=EvidenceBasedKnowledgeEstimator(),
                )
                for raw_ev in raw_events:
                    ev_dup = self._convert_event(raw_ev, student_id, concept_id, base_time)
                    duplicate_service.record_event(ev_dup)
                dup_state = duplicate_service.get_concept_state(student_id, concept_id)
                is_exact = (
                    final_state.knowledge_score == dup_state.knowledge_score
                    and final_state.confidence == dup_state.confidence
                    and final_state.mastery_status == dup_state.mastery_status
                )
                invariant_checks["deterministic_exact_match"] = is_exact

            # Count passed invariants
            sc_passed = all(invariant_checks.values())
            for passed in invariant_checks.values():
                total_invariants += 1
                if passed:
                    passed_invariants += 1

            scenario_results.append({
                "scenario_id": sc_id,
                "name": name,
                "all_invariants_passed": sc_passed,
                "final_knowledge_score": final_state.knowledge_score,
                "final_confidence": final_state.confidence,
                "final_mastery_status": final_state.mastery_status.value,
                "evidence_count": final_state.evidence_count,
                "invariant_checks": invariant_checks,
            })

        scenario_pass_rate = round(
            sum(1 for s in scenario_results if s["all_invariants_passed"]) / len(scenario_results), 4
        )
        invariant_pass_rate = round(passed_invariants / total_invariants, 4) if total_invariants > 0 else 1.0

        results = {
            "experiment_id": "EXP-006",
            "name": "Student Knowledge Modeling & Evidence-Based Personalization Benchmark",
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "total_scenarios": len(scenario_results),
            "scenarios_passed": sum(1 for s in scenario_results if s["all_invariants_passed"]),
            "scenario_pass_rate": scenario_pass_rate,
            "total_invariants_evaluated": total_invariants,
            "invariants_passed": passed_invariants,
            "invariant_pass_rate": invariant_pass_rate,
            "scientific_disclaimer": self.eval_data.get("scientific_disclaimer"),
            "scenario_details": scenario_results,
            "status": "COMPLETE",
        }

        self.output_results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info("EXP-006 results written to %s", self.output_results_path)
        return results
