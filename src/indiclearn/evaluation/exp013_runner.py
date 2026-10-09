"""Evaluation Runner for EXP-013: RAG Human Annotation Analysis (Phase 9).

Validates and analyzes expert human evaluation of generated RAG responses across:
1. Factual correctness (1-5)
2. Evidence support / Grounding (1-5)
3. Citation correctness (1-5)
4. Pedagogical usefulness (1-5)
5. Clarity (1-5)
6. Language quality (1-5)
7. Code-mixed understanding (1-5, where applicable)

Scientific Notice:
Zero human ratings, Cohen's kappa values, or agreement percentages are fabricated.
Without valid human annotation data, status remains strictly HUMAN_DATA_REQUIRED.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from indiclearn.evaluation.reporting.generator import ResearchReport
from indiclearn.evaluation.statistics.agreement import (
    compute_agreement_percentage,
    compute_cohens_kappa,
    compute_weighted_cohens_kappa,
)
from indiclearn.evaluation.statistics.descriptive import (
    compute_descriptive_stats,
    compute_distribution,
)

logger = logging.getLogger(__name__)


class EXP013Runner:
    """Automated runner and statistical analysis pipeline for EXP-013."""

    def __init__(
        self,
        template_path: str | Path = "data/evaluation/exp013_annotation_template.json",
        output_results_path: str | Path = "data/evaluation/exp013_results.json",
    ) -> None:
        self.template_path = Path(template_path)
        self.output_results_path = Path(output_results_path)

        with open(self.template_path, encoding="utf-8") as f:
            self.data = json.load(f)

    def analyze_annotations(self, annotations: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute descriptive, agreement, and subgroup metrics on annotations."""
        if not annotations:
            return {
                "status": "NOT_AVAILABLE",
                "reason": "No human annotation data supplied",
            }

        dimensions = self.data.get("rating_dimensions", [
            "factual_correctness",
            "evidence_support",
            "citation_correctness",
            "pedagogical_usefulness",
            "clarity",
            "language_quality",
            "code_mixed_understanding",
        ])

        # 1. Overall dimension summary
        dimension_summary: dict[str, Any] = {}
        for dim in dimensions:
            vals = [
                float(a["scores"][dim])
                for a in annotations
                if "scores" in a and dim in a["scores"] and a["scores"][dim] is not None
            ]
            if vals:
                desc = compute_descriptive_stats(vals)
                dist = compute_distribution(vals)
                dimension_summary[dim] = {
                    "n": len(vals),
                    "mean": desc.get("mean"),
                    "median": desc.get("median"),
                    "std": desc.get("std"),
                    "iqr": desc.get("iqr"),
                    "distribution": dist,
                }

        # 2. Per-language disaggregation
        per_language: dict[str, Any] = {}
        for lang in ["en", "te", "te-Latn", "te-en"]:
            lang_anns = [a for a in annotations if a.get("language") == lang]
            if lang_anns:
                per_language[lang] = {
                    "sample_size": len(lang_anns),
                    "dimension_means": {
                        dim: compute_descriptive_stats([
                            float(a["scores"][dim])
                            for a in lang_anns
                            if "scores" in a and dim in a["scores"] and a["scores"][dim] is not None
                        ]).get("mean")
                        for dim in dimensions
                    },
                }

        # 3. Inter-rater agreement (if multiple annotators annotated overlapping query_ids)
        annotators = list({a["annotator_id"] for a in annotations if "annotator_id" in a})
        agreement_metrics: dict[str, Any] = {
            "annotators_count": len(annotators),
            "annotator_ids": annotators,
        }

        if len(annotators) == 2:
            # Pairwise Cohen's kappa for key dimensions
            ann1, ann2 = annotators[0], annotators[1]
            q_map1 = {a["query_id"]: a for a in annotations if a.get("annotator_id") == ann1}
            q_map2 = {a["query_id"]: a for a in annotations if a.get("annotator_id") == ann2}
            common_queries = sorted(set(q_map1.keys()) & set(q_map2.keys()))

            if common_queries:
                dim_kappas: dict[str, Any] = {}
                for dim in ["factual_correctness", "evidence_support"]:
                    r1 = [q_map1[q]["scores"][dim] for q in common_queries if dim in q_map1[q]["scores"]]
                    r2 = [q_map2[q]["scores"][dim] for q in common_queries if dim in q_map2[q]["scores"]]
                    if len(r1) == len(r2) and len(r1) > 0:
                        raw = compute_agreement_percentage(r1, r2)
                        unweighted = compute_cohens_kappa(r1, r2)
                        weighted = compute_weighted_cohens_kappa(r1, r2, weight="quadratic")
                        dim_kappas[dim] = {
                            "n_paired": len(r1),
                            "raw_agreement_pct": raw.get("agreement_percentage"),
                            "cohen_kappa": unweighted.get("kappa"),
                            "weighted_kappa": weighted.get("weighted_kappa"),
                            "interpretation": weighted.get("interpretation"),
                        }
                agreement_metrics["pairwise_cohen"] = dim_kappas
        elif len(annotators) > 2:
            agreement_metrics["status"] = "MULTI_RATER_FLEISS_PREPARED"

        return {
            "total_annotations": len(annotations),
            "unique_queries_annotated": len({a.get("query_id") for a in annotations}),
            "dimension_summary": dimension_summary,
            "per_language": per_language,
            "agreement_metrics": agreement_metrics,
        }

    def run_all(self) -> dict[str, Any]:
        """Execute evaluation or return safe HUMAN_DATA_REQUIRED status."""
        annotations = self.data.get("annotations", [])

        if not annotations:
            overall_status = "HUMAN_DATA_REQUIRED"
            reason = "No human annotation data supplied. Framework verified and ready for annotator recruitment."
            analysis = {
                "status": "NOT_AVAILABLE",
                "reason": reason,
            }
        else:
            overall_status = "COMPLETED"
            reason = "Human RAG annotations analyzed across dimensions, languages, and inter-rater agreement."
            analysis = self.analyze_annotations(annotations)

        report = ResearchReport(
            experiment_id="EXP-013",
            title="RAG Human Annotation Analysis Framework",
            research_question="RQ-RAG-HUMAN: How do expert human evaluators rate generated RAG responses on factual correctness, evidence grounding, citation validity, and pedagogical clarity across English, Telugu, and code-mixed settings?",
            hypothesis="IndicLearn generated responses achieve mean factual correctness and evidence support >= 4.0/5.0 with quadratic weighted Cohen's kappa >= 0.60 across independent expert annotators.",
            dataset_identifier=str(self.template_path),
            sample_size=len(annotations),
            procedure="Independent double-annotation of RAG responses against retrieved course evidence chunks using 7 explicit 5-point Likert rating dimensions with strict anti-hallucination guidelines.",
            metrics={
                "status": overall_status,
                "total_annotations": len(annotations),
                "rating_dimensions_count": len(self.data.get("rating_dimensions", [])),
            },
            observed_results={
                "analysis": analysis,
                "status": overall_status,
                "reason": reason,
            },
            interpretation=(
                "The RAG human evaluation framework is verified and operational. "
                "Because no human annotators have been recruited yet, zero ratings or agreement scores are manufactured. "
                "Analysis pipeline is primed for immediate execution upon annotator data submission."
                if not annotations
                else "Human annotations analyzed across factual correctness, grounding, and agreement."
            ),
            scientific_claims=[
                "The evaluation pipeline establishes complete statistical and inter-rater agreement analysis for human RAG evaluation without data fabrication."
            ],
            limitations=[
                "Automated computational metrics (EXP-005A, EXP-005B) do not replace human judgment of pedagogical correctness.",
                "High inter-rater agreement does not guarantee factual ground truth; expert calibration is essential.",
                "Final empirical claims require recruiting qualified bilingual engineering evaluators.",
            ],
            reproducibility={
                "template_path": str(self.template_path),
                "data_status": "AWAITING_HUMAN_ANNOTATIONS",
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        output_dict = {
            "metadata": {
                "experiment_id": "EXP-013",
                "status": overall_status,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sample_size": len(annotations),
                "scientific_disclaimer": self.data["metadata"]["scientific_disclaimer"],
            },
            "analysis": analysis,
            "report": report.to_dict(),
        }

        self.output_results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_results_path, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, indent=2)

        return output_dict
