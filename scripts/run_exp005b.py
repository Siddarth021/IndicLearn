"""Script to run EXP-005B: RAG Safety, Scope Control & Retrieval Hardening.

Usage:
    python scripts/run_exp005b.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from indiclearn.evaluation.exp005b_runner import Exp005bRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_exp005b")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EXP-005B: RAG Safety, Scope Control & Retrieval Hardening",
    )
    parser.add_argument(
        "--scope-path",
        default="data/evaluation/scope_eval.json",
        help="Path to scope evaluation benchmark dataset",
    )
    parser.add_argument(
        "--rag-path",
        default="data/evaluation/rag_eval_expanded.json",
        help="Path to expanded RAG evaluation benchmark dataset",
    )
    parser.add_argument(
        "--corpus-path",
        default="data/evaluation/signals_and_systems_eval.json",
        help="Path to corpus chunks JSON",
    )
    parser.add_argument(
        "--graph-path",
        default="data/evaluation/concept_graph_gold.json",
        help="Path to concept graph gold JSON",
    )
    parser.add_argument(
        "--output-path",
        default="data/evaluation/exp005b_results.json",
        help="Path to write evaluation results JSON",
    )

    args = parser.parse_args()

    logger.info("Starting EXP-005B evaluation...")
    runner = Exp005bRunner(
        scope_eval_path=args.scope_path,
        rag_eval_path=args.rag_path,
        corpus_path=args.corpus_path,
        concept_graph_path=args.graph_path,
    )

    results = runner.run_experiment(output_path=args.output_path)

    # Print summary of results
    print("\n" + "=" * 70)
    print("           EXP-005B EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Scope Benchmark:      {args.scope_path} ({results['scope_evaluation']['total_queries']} queries)")

    se = results["scope_evaluation"]
    print("\n--- SCOPE EVALUATION METRICS ---")
    print(f"  Scope Accuracy:            {se['accuracy']:.4f}")
    print(f"  Macro Precision:           {se['macro_precision']:.4f}")
    print(f"  Macro Recall:              {se['macro_recall']:.4f}")
    print(f"  Macro F1:                  {se['macro_f1']:.4f}")
    print(f"  False Acceptance Rate:     {se['false_acceptance_rate']:.4f} ({se['false_acceptance_count']} false acceptances)")

    print("\n--- PER-CLASS SCOPE METRICS ---")
    for cls_name, m in se["per_class_metrics"].items():
        print(f"  {cls_name:<15} (n={m['support']:2d}): P={m['precision']:.4f} | R={m['recall']:.4f} | F1={m['f1']:.4f} (TP={m['true_positives']}, FP={m['false_positives']}, FN={m['false_negatives']})")

    print("\n--- CONFUSION MATRIX (Actual \\ Predicted) ---")
    classes = ["in_scope", "out_of_scope", "ambiguous"]
    header = "Actual \\ Pred".ljust(16) + "".join(c.rjust(14) for c in classes)
    print(header)
    for actual in classes:
        row = actual.ljust(16) + "".join(str(se["confusion_matrix"][actual][pred]).rjust(14) for pred in classes)
        print(row)

    if se["false_acceptances"]:
        print("\n--- FALSE ACCEPTANCE INVESTIGATION ---")
        for fa in se["false_acceptances"]:
            print(f"  [{fa['query_id']}] '{fa['query']}'")
            print(f"     Reason: {fa['reason']}")
    else:
        print("\n--- FALSE ACCEPTANCE INVESTIGATION ---")
        print("  Zero false acceptances detected on out-of-scope/adversarial queries!")

    print("\n--- GRAPH ABLATION RERUN (Hardened Architecture) ---")
    ga = results["graph_ablation_rerun"]
    print(f"  System A (Hybrid only):       R@1={ga['system_a_hybrid_only']['recall_at_1']:.4f} | R@3={ga['system_a_hybrid_only']['recall_at_3']:.4f} | R@5={ga['system_a_hybrid_only']['recall_at_5']:.4f} | MRR={ga['system_a_hybrid_only']['mrr']:.4f} | GoldR@5={ga['system_a_hybrid_only']['gold_evidence_recall_at_5']:.4f}")
    print(f"  System B (+ Concept Mapping): R@1={ga['system_b_hybrid_plus_concept_mapping']['recall_at_1']:.4f} | R@3={ga['system_b_hybrid_plus_concept_mapping']['recall_at_3']:.4f} | R@5={ga['system_b_hybrid_plus_concept_mapping']['recall_at_5']:.4f} | MRR={ga['system_b_hybrid_plus_concept_mapping']['mrr']:.4f} | GoldR@5={ga['system_b_hybrid_plus_concept_mapping']['gold_evidence_recall_at_5']:.4f}")
    print(f"  System C (+ Post-Ret Graph):  R@1={ga['system_c_post_retrieval_graph_enrichment']['recall_at_1']:.4f} | R@3={ga['system_c_post_retrieval_graph_enrichment']['recall_at_3']:.4f} | R@5={ga['system_c_post_retrieval_graph_enrichment']['recall_at_5']:.4f} | MRR={ga['system_c_post_retrieval_graph_enrichment']['mrr']:.4f} | GoldR@5={ga['system_c_post_retrieval_graph_enrichment']['gold_evidence_recall_at_5']:.4f}")

    print(f"\nOverall Status: {results['status']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
