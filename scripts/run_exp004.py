#!/usr/bin/env python
"""Execution script for EXP-004: Indic NLP & Cross-Lingual Concept Mapping.

Runs evaluations on data/evaluation/concept_mapping_eval.json across:
1. Language/Script Identification
2. Technical Term Detection
3. Concept Mapping Baselines A-D
4. Error Analysis
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indiclearn.evaluation.exp004_runner import Exp004Runner


def main() -> int:
    print("=" * 80)
    print("INDICLEARN — EXP-004: INDIC NLP & CROSS-LINGUAL CONCEPT MAPPING")
    print("=" * 80)

    runner = Exp004Runner()
    results = runner.run_experiment("data/evaluation/exp004_results.json")

    metrics = results["metrics"]
    lang_res = metrics["language_and_script_detection"]
    term_res = metrics["technical_term_detection"]
    baselines = metrics["concept_mapping_baselines"]
    errors = results["error_analysis"]

    # Table 1: Language & Script Detection
    print("\n--- 1. LANGUAGE & SCRIPT IDENTIFICATION ---")
    print(f"Total Evaluated Queries: {lang_res['total_queries']}")
    print(f"Overall Language Accuracy: {lang_res['language_accuracy']:.2%}")
    print(f"Overall Script Accuracy:   {lang_res['script_accuracy']:.2%}\n")

    print(f"{'Category':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 60)
    for cat, data in lang_res["per_category"].items():
        print(
            f"{cat:<12} | {data['precision']:<10.4f} | {data['recall']:<10.4f} | "
            f"{data['f1']:<10.4f} | {data['support']:<8}"
        )

    print("\nConfusion Matrix (Rows: Gold, Columns: Pred):")
    cols = ["en", "te", "te-Latn", "te-en", "other"]
    title = "Gold \\ Pred"
    print(f"{title:<12} | " + " | ".join(f"{c:<7}" for c in cols))
    print("-" * 60)
    for gold_cat, row in lang_res["confusion_matrix"].items():
        row_str = " | ".join(f"{row.get(c, 0):<7}" for c in cols)
        print(f"{gold_cat:<12} | {row_str}")

    # Table 2: Technical Term Detection
    print("\n--- 2. TECHNICAL TERM DETECTION ---")
    print(f"Precision: {term_res['precision']:.4f}")
    print(f"Recall:    {term_res['recall']:.4f}")
    print(f"F1-Score:  {term_res['f1']:.4f}")
    print(f"Counts:    TP={term_res['true_positives']}, FP={term_res['false_positives']}, FN={term_res['false_negatives']}")

    # Table 3: Concept Mapping Baselines
    print("\n--- 3. CONCEPT MAPPING BASELINES COMPARISON ---")
    header = f"{'Baseline':<24} | {'Recall@1':<10} | {'Recall@3':<10} | {'MRR':<10} | {'EN R@1':<8} | {'TE R@1':<8} | {'RO R@1':<8} | {'CM R@1':<8}"
    print(header)
    print("-" * len(header))

    for b_data in baselines.values():
        cat_r1 = b_data["per_category_recall_at_1"]
        print(
            f"{b_data['baseline']:<24} | {b_data['recall_at_1']:<10.4f} | {b_data['recall_at_3']:<10.4f} | "
            f"{b_data['mrr']:<10.4f} | {cat_r1.get('en', 0.0):<8.4f} | {cat_r1.get('te', 0.0):<8.4f} | "
            f"{cat_r1.get('te-Latn', 0.0):<8.4f} | {cat_r1.get('te-en', 0.0):<8.4f}"
        )

    # Table 4: Error Analysis
    print("\n--- 4. ERROR ANALYSIS & AMBIGUITY ---")
    print(f"Total Mapping Failures: {errors['total_failures']}")
    print("Error Distribution:")
    for err_type, count in errors["error_distribution"].items():
        print(f"  - {err_type:<30}: {count}")

    print("\n" + "=" * 80)
    print("EXP-004 RUN COMPLETE. Results saved to data/evaluation/exp004_results.json")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
