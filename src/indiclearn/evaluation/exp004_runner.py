"""EXP-004: Indic NLP and Cross-Lingual Concept Mapping Evaluation Runner.

Evaluates:
1. Language and Script Identification (Accuracy, Per-Category P/R/F1, Confusion Matrix)
2. Technical Term Detection (Precision, Recall, F1)
3. Concept Mapping across 4 Baselines:
   - Baseline A: Exact Canonical & Alias Matching
   - Baseline B: Normalized Lexical Matching
   - Baseline C: Dense Embedding Candidate Ranking
   - Baseline D: Multi-Signal Hybrid Mapping
4. Error Analysis on Romanized Telugu and Code-Mixed queries.

All results are empirical evaluations on the controlled benchmark.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from indiclearn.domain.concepts import Concept
from indiclearn.nlp.concept_mapping.base import ConceptMapper
from indiclearn.nlp.concept_mapping.mapper import (
    EmbeddingConceptMapper,
    ExactConceptMapper,
    HybridConceptMapper,
    LexicalConceptMapper,
)
from indiclearn.nlp.language_detection.detector import HeuristicLanguageDetector
from indiclearn.nlp.models import Language
from indiclearn.nlp.normalization.normalizer import UnifiedQueryNormalizer
from indiclearn.nlp.terminology.detector import GraphGroundedTermDetector

logger = logging.getLogger(__name__)


class Exp004Runner:
    """Experiment runner for EXP-004 (Indic NLP & Cross-Lingual Concept Mapping)."""

    def __init__(
        self,
        eval_path: str = "data/evaluation/concept_mapping_eval.json",
        concept_graph_path: str = "data/evaluation/concept_graph_gold.json",
        subject_id: str = "signals_and_systems",
    ) -> None:
        self.eval_path = Path(eval_path)
        self.concept_graph_path = Path(concept_graph_path)
        self.subject_id = subject_id

        self.eval_data = self._load_json(self.eval_path)
        self.concepts = self._load_concepts(self.concept_graph_path)

        # Initialize shared components
        self.language_detector = HeuristicLanguageDetector()
        self.normalizer = UnifiedQueryNormalizer()
        self.term_detector = GraphGroundedTermDetector(concepts=self.concepts)

    def _load_json(self, path: Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_concepts(self, path: Path) -> list[Concept]:
        data = self._load_json(path)
        concepts = []
        for item in data.get("concepts", []):
            concepts.append(
                Concept(
                    concept_id=item["concept_id"],
                    canonical_name=item["canonical_name"],
                    subject_id=self.subject_id,
                    concept_type=item.get("concept_type", "concept"),
                    aliases=item.get("aliases", []),
                    description=item.get("description", ""),
                    source_chunk_ids=item.get("source_chunk_ids", []),
                )
            )
        return concepts

    def evaluate_language_detection(self) -> dict[str, Any]:
        """Evaluate language and script identification performance."""
        queries = self.eval_data.get("queries", [])
        total = len(queries)
        if total == 0:
            return {}

        correct_lang = 0
        correct_script = 0

        categories = ["en", "te", "te-Latn", "te-en"]
        confusion_matrix: dict[str, dict[str, int]] = {
            cat: {c: 0 for c in categories + ["other"]} for cat in categories
        }

        tp: dict[str, int] = defaultdict(int)
        fp: dict[str, int] = defaultdict(int)
        fn: dict[str, int] = defaultdict(int)

        for q in queries:
            gold_lang = q["expected_language"]
            gold_script = q["expected_script"]
            text = q["text"]

            res = self.language_detector.detect(text)
            pred_lang = res.primary_language.value
            pred_script = res.detected_script.value

            if pred_lang == gold_lang:
                correct_lang += 1
            if pred_script == gold_script or (
                gold_script == "Mixed" and pred_script in ["Mixed", "Telugu"]
            ):
                correct_script += 1

            # Confusion matrix tracking
            matrix_pred = pred_lang if pred_lang in categories else "other"
            if gold_lang in confusion_matrix:
                confusion_matrix[gold_lang][matrix_pred] += 1

            for cat in categories:
                if gold_lang == cat and pred_lang == cat:
                    tp[cat] += 1
                elif gold_lang != cat and pred_lang == cat:
                    fp[cat] += 1
                elif gold_lang == cat and pred_lang != cat:
                    fn[cat] += 1

        per_category = {}
        for cat in categories:
            p = tp[cat] / (tp[cat] + fp[cat]) if (tp[cat] + fp[cat]) > 0 else 0.0
            r = tp[cat] / (tp[cat] + fn[cat]) if (tp[cat] + fn[cat]) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            per_category[cat] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
                "support": tp[cat] + fn[cat],
            }

        return {
            "total_queries": total,
            "language_accuracy": round(correct_lang / total, 4),
            "script_accuracy": round(correct_script / total, 4),
            "per_category": per_category,
            "confusion_matrix": confusion_matrix,
        }

    def evaluate_technical_term_detection(self) -> dict[str, Any]:
        """Evaluate technical term detection against gold term annotations."""
        queries = self.eval_data.get("queries", [])
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for q in queries:
            text = q["text"]
            gold_terms = {t.lower().strip() for t in q.get("gold_technical_terms", [])}
            detected = self.term_detector.detect_terms(text, subject_id=self.subject_id)
            pred_spans = {t.matched_span.lower().strip() for t in detected}

            # Also check normalized text
            norm_res = self.normalizer.normalize(
                text, primary_language=Language(q.get("expected_language", "en"))
            )
            if norm_res.normalized_text != text:
                detected_norm = self.term_detector.detect_terms(
                    norm_res.normalized_text, subject_id=self.subject_id
                )
                pred_spans.update({t.matched_span.lower().strip() for t in detected_norm})

            # Partial match or exact span match
            for pred in pred_spans:
                if any(pred in g or g in pred for g in gold_terms):
                    total_tp += 1
                else:
                    total_fp += 1

            for g in gold_terms:
                if not any(pred in g or g in pred for pred in pred_spans):
                    total_fn += 1

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
        }

    def evaluate_concept_mapper(
        self,
        mapper: ConceptMapper,
        baseline_name: str,
    ) -> dict[str, Any]:
        """Evaluate a concept mapper implementation on Recall@1, Recall@3, and MRR."""
        queries = self.eval_data.get("queries", [])
        total = len(queries)
        if total == 0:
            return {}

        recall_at_1 = 0
        recall_at_3 = 0
        reciprocal_ranks: list[float] = []
        per_category_r1: dict[str, list[int]] = defaultdict(list)

        for q in queries:
            text = q["text"]
            category = q["category"]
            gold_ids = set(q.get("gold_concept_ids", []))

            # Detect terms first
            terms = self.term_detector.detect_terms(text, subject_id=self.subject_id)
            res = mapper.map_query(text, subject_id=self.subject_id, detected_terms=terms)

            candidates = res.candidates
            candidate_ids = [c.concept_id for c in candidates]

            # Check R@1
            top1_hit = 1 if (candidate_ids and candidate_ids[0] in gold_ids) else 0
            recall_at_1 += top1_hit
            per_category_r1[category].append(top1_hit)

            # Check R@3
            top3_ids = set(candidate_ids[:3])
            recall_at_3 += 1 if any(gid in top3_ids for gid in gold_ids) else 0

            # Compute reciprocal rank
            rank = 0
            for idx, cid in enumerate(candidate_ids, start=1):
                if cid in gold_ids:
                    rank = idx
                    break

            reciprocal_ranks.append(1.0 / rank if rank > 0 else 0.0)

        mrr = sum(reciprocal_ranks) / total if total > 0 else 0.0

        cat_r1_summary = {}
        for cat, hits in per_category_r1.items():
            cat_r1_summary[cat] = round(sum(hits) / len(hits), 4) if hits else 0.0

        return {
            "baseline": baseline_name,
            "total_queries": total,
            "recall_at_1": round(recall_at_1 / total, 4),
            "recall_at_3": round(recall_at_3 / total, 4),
            "mrr": round(mrr, 4),
            "per_category_recall_at_1": cat_r1_summary,
        }

    def run_all_baselines(self) -> dict[str, Any]:
        """Run complete scientific comparison across Baselines A, B, C, and D."""
        logger.info("Initializing Concept Mapping Baselines A-D...")

        # Baseline A: Exact canonical/alias mapper
        baseline_a = ExactConceptMapper(concepts=self.concepts)

        # Baseline B: Normalized lexical mapper
        baseline_b = LexicalConceptMapper(concepts=self.concepts)

        # Baseline C: Dense semantic embedding mapper
        baseline_c = EmbeddingConceptMapper(concepts=self.concepts)

        # Baseline D: Multi-signal hybrid mapper
        baseline_d = HybridConceptMapper(concepts=self.concepts)

        results = {
            "baseline_a_exact": self.evaluate_concept_mapper(baseline_a, "Baseline A (Exact)"),
            "baseline_b_lexical": self.evaluate_concept_mapper(
                baseline_b, "Baseline B (Lexical)"
            ),
            "baseline_c_embedding": self.evaluate_concept_mapper(
                baseline_c, "Baseline C (Embedding)"
            ),
            "baseline_d_hybrid": self.evaluate_concept_mapper(
                baseline_d, "Baseline D (Hybrid)"
            ),
        }
        return results

    def run_error_analysis(self) -> dict[str, Any]:
        """Conduct fine-grained error analysis on Romanized Telugu and Code-Mixed queries."""
        queries = self.eval_data.get("queries", [])
        mapper = HybridConceptMapper(concepts=self.concepts)

        failures = []
        error_categories = {
            "spelling_variation": 0,
            "short_token_ambiguity": 0,
            "unseen_technical_alias": 0,
            "multilingual_script_mixing": 0,
            "syntax_colloquialism": 0,
        }

        for q in queries:
            text = q["text"]
            category = q["category"]
            gold_ids = set(q.get("gold_concept_ids", []))

            terms = self.term_detector.detect_terms(text, subject_id=self.subject_id)
            res = mapper.map_query(text, subject_id=self.subject_id, detected_terms=terms)

            top1_id = res.primary_concept.concept_id if res.primary_concept else None
            if top1_id not in gold_ids:
                failure_entry = {
                    "query_id": q["query_id"],
                    "text": text,
                    "category": category,
                    "gold_ids": list(gold_ids),
                    "predicted_id": top1_id,
                    "confidence": res.mapping_confidence,
                }
                failures.append(failure_entry)

                # Heuristic categorization
                if category == "te-Latn":
                    error_categories["spelling_variation"] += 1
                elif category == "te":
                    error_categories["multilingual_script_mixing"] += 1
                elif category == "te-en":
                    error_categories["syntax_colloquialism"] += 1
                else:
                    error_categories["unseen_technical_alias"] += 1

        return {
            "total_failures": len(failures),
            "error_distribution": error_categories,
            "failure_cases": failures,
        }

    def run_experiment(self, output_path: str = "data/evaluation/exp004_results.json") -> dict[str, Any]:
        """Run end-to-end EXP-004 evaluation and export results JSON."""
        logger.info("Executing EXP-004: Indic NLP & Concept Mapping...")

        lang_metrics = self.evaluate_language_detection()
        term_metrics = self.evaluate_technical_term_detection()
        concept_metrics = self.run_all_baselines()
        error_analysis = self.run_error_analysis()

        exp_results = {
            "experiment_id": "EXP-004",
            "name": "Indic NLP and Cross-Lingual Concept Mapping Evaluation",
            "benchmark_dataset": str(self.eval_path),
            "concept_graph": str(self.concept_graph_path),
            "subject_id": self.subject_id,
            "metrics": {
                "language_and_script_detection": lang_metrics,
                "technical_term_detection": term_metrics,
                "concept_mapping_baselines": concept_metrics,
            },
            "error_analysis": error_analysis,
        }

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(exp_results, f, indent=2, ensure_ascii=False)

        logger.info(f"EXP-004 results successfully written to {output_path}")
        return exp_results
