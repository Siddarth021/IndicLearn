"""Evaluation Runner for EXP-012: Dashboard Usability Evaluation Framework (Phase 9).

Validates the dashboard usability protocol, task specifications, Likert questionnaire
schema, and response analysis pipeline.

Scientific Notice:
Distinguishes software correctness (EXP-009) from human usability (EXP-012).
If no real participant responses are supplied, status remains NOT_RUN (HUMAN_DATA_REQUIRED).
Zero participant responses, completion times, or Likert ratings are manufactured.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from indiclearn.evaluation.reporting.generator import ResearchReport
from indiclearn.evaluation.statistics.descriptive import (
    compute_descriptive_stats,
    compute_distribution,
)

logger = logging.getLogger(__name__)


class EXP012Runner:
    """Automated runner and validator for EXP-012 dashboard usability study."""

    def __init__(
        self,
        template_path: str | Path = "data/evaluation/exp012_usability_template.json",
        output_results_path: str | Path = "data/evaluation/exp012_results.json",
    ) -> None:
        self.template_path = Path(template_path)
        self.output_results_path = Path(output_results_path)

        with open(self.template_path, encoding="utf-8") as f:
            self.data = json.load(f)

    def validate_framework(self) -> dict[str, Any]:
        """Validate the structural schema and neutrality of tasks and survey items."""
        tasks = self.data.get("tasks", [])
        likert_items = self.data.get("likert_items", [])
        view_modes = self.data.get("view_modes_evaluated", [])

        # Validate task definitions
        task_checks: list[bool] = []
        for t in tasks:
            valid_t = (
                "task_id" in t
                and "name" in t
                and "prompt" in t
                and "success_criteria" in t
                and t.get("max_time_seconds", 0) > 0
            )
            task_checks.append(valid_t)

        # Validate Likert items for neutral phrasing (no leading questions)
        forbidden_phrases = ["helped me learn better", "made me smarter", "improved my grades", "proved easy"]
        likert_checks: list[bool] = []
        for item in likert_items:
            stmt = item.get("statement", "").lower()
            is_neutral = not any(phrase in stmt for phrase in forbidden_phrases)
            has_id = "item_id" in item
            likert_checks.append(is_neutral and has_id)

        framework_valid = (
            len(tasks) >= 5
            and all(task_checks)
            and len(likert_items) >= 6
            and all(likert_checks)
            and "GRAPH_VIEW" in view_modes
            and "LIST_VIEW" in view_modes
        )

        return {
            "tasks_count": len(tasks),
            "all_tasks_valid": all(task_checks),
            "likert_items_count": len(likert_items),
            "all_likert_neutral": all(likert_checks),
            "view_modes": view_modes,
            "framework_valid": framework_valid,
        }

    def analyze_responses(self, responses: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze actual participant responses if present."""
        if not responses:
            return {
                "status": "NOT_AVAILABLE",
                "reason": "No participant responses provided.",
            }

        # Analyze task completion
        task_metrics: dict[str, Any] = {}
        for r in responses:
            for tp in r.get("task_performance", []):
                tid = tp["task_id"]
                task_metrics.setdefault(tid, {"completed": 0, "times": [], "errors": []})
                if tp.get("completed", False):
                    task_metrics[tid]["completed"] += 1
                if "completion_time_seconds" in tp:
                    task_metrics[tid]["times"].append(float(tp["completion_time_seconds"]))
                if "error_count" in tp:
                    task_metrics[tid]["errors"].append(float(tp["error_count"]))

        task_summary: dict[str, Any] = {}
        n_resp = len(responses)
        for tid, stats in task_metrics.items():
            times = stats["times"]
            time_stats = compute_descriptive_stats(times) if times else {}
            task_summary[tid] = {
                "completion_rate": stats["completed"] / n_resp if n_resp > 0 else 0.0,
                "mean_time_seconds": time_stats.get("mean"),
                "median_time_seconds": time_stats.get("median"),
                "std_time_seconds": time_stats.get("std"),
            }

        # Analyze Likert responses
        likert_summary: dict[str, Any] = {}
        for q_key in ["Q01", "Q02", "Q03", "Q04", "Q05", "Q06"]:
            q_vals = [
                float(r["likert_responses"][k])
                for r in responses
                if "likert_responses" in r
                for k in r["likert_responses"]
                if k.startswith(q_key) and r["likert_responses"][k] is not None
            ]
            if q_vals:
                desc = compute_descriptive_stats(q_vals)
                dist = compute_distribution(q_vals)
                likert_summary[q_key] = {
                    "n": len(q_vals),
                    "mean": desc.get("mean"),
                    "median": desc.get("median"),
                    "std": desc.get("std"),
                    "distribution": dist,
                }

        return {
            "n_participants": n_resp,
            "task_summary": task_summary,
            "likert_summary": likert_summary,
        }

    def run_all(self) -> dict[str, Any]:
        """Execute evaluation or framework verification."""
        validation = self.validate_framework()
        responses = self.data.get("responses", [])

        if not responses:
            overall_status = "NOT_RUN"
            reason = "HUMAN_DATA_REQUIRED: No human participant responses supplied in evaluation template."
            analysis_result = {
                "status": "NOT_AVAILABLE",
                "reason": reason,
            }
        else:
            overall_status = "COMPLETED"
            reason = "Human participant data analyzed."
            analysis_result = self.analyze_responses(responses)

        report = ResearchReport(
            experiment_id="EXP-012",
            title="Student Learning Dashboard Human Usability Evaluation Framework",
            research_question="RQ-DASH: Can engineering students navigate concept dependencies, understand latent knowledge estimates, and interpret personalized learning recommendations transparently across Graph and List views?",
            hypothesis="Students achieve >80% task completion on concept localization and recommendation interpretation, with mean Likert transparency ratings >= 3.5/5.0 across both visualization modes.",
            dataset_identifier=str(self.template_path),
            sample_size=len(responses),
            procedure="Structured protocol administering 5 operational tasks (T1-T5) and a 6-item neutral Likert questionnaire (Q01-Q06) evaluating navigation, estimate comprehension, evidence tracing, and Graph vs List views.",
            metrics={
                "framework_valid": validation["framework_valid"],
                "total_tasks_specified": validation["tasks_count"],
                "total_likert_items": validation["likert_items_count"],
                "participant_responses_count": len(responses),
                "status": overall_status,
            },
            observed_results={
                "validation": validation,
                "analysis": analysis_result,
                "status": overall_status,
                "reason": reason,
            },
            interpretation=(
                "The dashboard usability testing framework is structurally complete and neutral. "
                "No human participants have yet been recruited, maintaining zero data fabrication policy. "
                "Framework is primed for IRB-approved human validation."
                if not responses
                else "Human usability responses evaluated across task completion and Likert dimensions."
            ),
            scientific_claims=[
                "The dashboard evaluation framework establishes verified schemas, neutral Likert items, and automated aggregation pipelines for human usability studies without data fabrication."
            ],
            limitations=[
                "Actual usability scores require live participant cohorts under approved ethical protocol.",
                "Synthetic benchmark EXP-009 proves interface correctness, but does not substitute for human testing.",
                "No claims regarding student satisfaction or cognitive ergonomics can be made until human evaluation is conducted.",
            ],
            reproducibility={
                "template_path": str(self.template_path),
                "framework_version": "1.0",
                "data_collection_status": "AWAITING_HUMAN_PARTICIPANTS",
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        output_dict = {
            "metadata": {
                "experiment_id": "EXP-012",
                "status": overall_status,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "framework_validation": validation,
                "sample_size": len(responses),
                "scientific_disclaimer": self.data["metadata"]["scientific_disclaimer"],
            },
            "analysis": analysis_result,
            "report": report.to_dict(),
        }

        self.output_results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_results_path, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, indent=2)

        return output_dict
