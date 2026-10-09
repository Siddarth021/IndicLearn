"""Evaluation Runner for EXP-014: Multilingual Human Evaluation (Phase 9).

Validates and analyzes multilingual human evaluations of educational NLP responses
across four strictly disaggregated language modalities:
1. English (en)
2. Telugu native script (te)
3. Romanized Telugu (te-Latn)
4. Telugu-English code-mix (te-en)

Scientific Notice:
Language categories must NEVER be collapsed into a single blended aggregate that
masks low-resource or code-mixed degradation.
If no real multilingual participant data is provided, status remains strictly HUMAN_DATA_REQUIRED.
Zero human ratings, transliteration scores, or fluency ratings are fabricated.
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


class EXP014Runner:
    """Automated runner and disaggregated multilingual analysis pipeline for EXP-014."""

    def __init__(
        self,
        template_path: str | Path = "data/evaluation/exp014_multilingual_template.json",
        output_results_path: str | Path = "data/evaluation/exp014_results.json",
    ) -> None:
        self.template_path = Path(template_path)
        self.output_results_path = Path(output_results_path)

        with open(self.template_path, encoding="utf-8") as f:
            self.data = json.load(f)

    def analyze_evaluations(self, evaluations: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute disaggregated per-language statistics."""
        if not evaluations:
            return {
                "status": "NOT_AVAILABLE",
                "reason": "No multilingual human evaluation data supplied",
            }

        modalities = ["en", "te", "te-Latn", "te-en"]
        dimensions = [
            "factual_correctness",
            "evidence_grounding",
            "linguistic_naturalness",
            "technical_term_preservation",
            "pedagogical_clarity",
        ]

        per_modality_analysis: dict[str, Any] = {}

        for mod in modalities:
            mod_evals = [e for e in evaluations if e.get("modality") == mod or e.get("language") == mod]
            n = len(mod_evals)
            if n == 0:
                per_modality_analysis[mod] = {
                    "sample_size": 0,
                    "status": "NO_DATA_FOR_MODALITY",
                }
                continue

            dim_stats: dict[str, Any] = {}
            for dim in dimensions:
                vals = [
                    float(e["scores"][dim])
                    for e in mod_evals
                    if "scores" in e and dim in e["scores"] and e["scores"][dim] is not None
                ]
                if vals:
                    stats = compute_descriptive_stats(vals)
                    dist = compute_distribution(vals)
                    dim_stats[dim] = {
                        "n": len(vals),
                        "mean": stats.get("mean"),
                        "median": stats.get("median"),
                        "std": stats.get("std"),
                        "distribution": dist,
                    }

            mod_record: dict[str, Any] = {
                "sample_size": n,
                "dimensions": dim_stats,
            }

            # Modality-specific research annotations
            if mod == "te-Latn":
                mod_record["linguistic_considerations"] = (
                    "Subject to phonetic transliteration variability, inconsistent vowel length representation, "
                    "and non-standardized orthography across engineering students."
                )
            elif mod == "te-en":
                mod_record["linguistic_considerations"] = (
                    "Features technical English nouns (e.g., 'convolution', 'transfer function') "
                    "embedded within Telugu morphosyntactic postpositional frames (e.g., 'lo', 'tho', 'ki', 'kabatti')."
                )

            per_modality_analysis[mod] = mod_record

        return {
            "total_evaluations": len(evaluations),
            "disaggregated_modalities": per_modality_analysis,
        }

    def run_all(self) -> dict[str, Any]:
        """Execute multilingual evaluation or return safe HUMAN_DATA_REQUIRED status."""
        evaluations = self.data.get("evaluations", [])

        if not evaluations:
            overall_status = "HUMAN_DATA_REQUIRED"
            reason = "No multilingual human evaluation data supplied. Framework verified and ready for multilingual panel administration."
            analysis = {
                "status": "NOT_AVAILABLE",
                "reason": reason,
            }
        else:
            overall_status = "COMPLETED"
            reason = "Multilingual human evaluations analyzed across English, Telugu, Romanized Telugu, and Code-mixed modalities."
            analysis = self.analyze_evaluations(evaluations)

        report = ResearchReport(
            experiment_id="EXP-014",
            title="Multilingual Human Evaluation of Educational NLP Responses",
            research_question="RQ-MULTI-HUMAN: How do bilingual Telugu-English engineering students evaluate the correctness, grounding, and pedagogical clarity of answers generated across English, Telugu native script, Romanized Telugu, and code-mixed modalities?",
            hypothesis="IndicLearn maintains high factual grounding (>=4.0/5.0) across both English and Telugu modalities, with code-mixed explanations achieving higher perceived pedagogical clarity among bilingual students than purely translated technical Telugu.",
            dataset_identifier=str(self.template_path),
            sample_size=len(evaluations),
            procedure="Administer 4-modality comparative evaluation (en, te, te-Latn, te-en) across computer science concepts to bilingual engineering students, measuring correctness, grounding, naturalness, and technical terminology preservation.",
            metrics={
                "status": overall_status,
                "total_evaluations": len(evaluations),
                "modalities_configured": [m["code"] for m in self.data.get("modalities", [])],
            },
            observed_results={
                "analysis": analysis,
                "status": overall_status,
                "reason": reason,
            },
            interpretation=(
                "The multilingual human evaluation framework is established and disaggregated. "
                "Because real student evaluators have not yet been recruited, zero multilingual ratings are manufactured. "
                "The pipeline prevents conflating standard English performance with low-resource or code-mixed performance."
                if not evaluations
                else "Multilingual responses evaluated with strict disaggregation across modalities."
            ),
            scientific_claims=[
                "The multilingual evaluation framework enforces strict disaggregation across English, Telugu, Romanized Telugu, and code-mixed settings to prevent masked degradation."
            ],
            limitations=[
                "Automated retrieval and scope benchmarks (EXP-005A, EXP-005B) evaluate lexical/embedding retrieval, not human linguistic acceptability.",
                "Romanized Telugu exhibits substantial dialectal and spelling variation across regional demographics.",
                "Empirical conclusions require formal execution with enrolled engineering students under approved IRB protocol.",
            ],
            reproducibility={
                "template_path": str(self.template_path),
                "data_status": "AWAITING_MULTILINGUAL_EVALUATORS",
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        output_dict = {
            "metadata": {
                "experiment_id": "EXP-014",
                "status": overall_status,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sample_size": len(evaluations),
                "scientific_disclaimer": self.data["metadata"]["scientific_disclaimer"],
            },
            "analysis": analysis,
            "report": report.to_dict(),
        }

        self.output_results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_results_path, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, indent=2)

        return output_dict
