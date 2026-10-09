#!/usr/bin/env python
"""Execution script for EXP-005: RAG Pipeline & Grounded Answer Generation.

Evaluates:
1. Retrieval Metrics (Recall@1, Recall@3, Recall@5, MRR, Multi-Chunk Evidence Recall)
2. Insufficient Evidence Detection
3. Grounding & Citation Integrity (Citation Validity Rate, Coverage, Unsupported Rate)
4. Baselines Comparison (Baseline A, B, C)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indiclearn.evaluation.exp005_runner import Exp005Runner


def main() -> int:
    print("=" * 80)
    print("INDICLEARN — EXP-005: RAG PIPELINE & GROUNDED ANSWER GENERATION")
    print("=" * 80)

    runner = Exp005Runner()
    results = runner.run_experiment("data/evaluation/exp005_results.json")

    metrics = results["metrics"]
    ret_res = metrics["retrieval"]
    pipe_res = metrics["pipeline_and_grounding"]
    baselines = metrics["baselines_comparison"]

    # Table 1: Retrieval Metrics
    print("\n--- 1. RETRIEVAL METRICS (Answerable Queries) ---")
    print(f"Total Answerable Queries:        {ret_res['total_answerable_queries']}")
    print(f"Recall@1:                        {ret_res['recall_at_1']:.4f}")
    print(f"Recall@3:                        {ret_res['recall_at_3']:.4f}")
    print(f"Recall@5:                        {ret_res['recall_at_5']:.4f}")
    print(f"Mean Reciprocal Rank (MRR):      {ret_res['mrr']:.4f}")
    print(f"Gold Evidence Recall@5:          {ret_res['gold_evidence_recall_at_k']:.4f}")
    print(f"Multi-Chunk Evidence Recall:     {ret_res['multi_chunk_evidence_recall']:.4f}")

    print("\nPer-Category Recall@3:")
    for cat, val in ret_res["per_category_recall_at_3"].items():
        print(f"  - {cat:<26}: {val:.4f}")

    # Table 2: Answerability & Insufficiency Detection
    print("\n--- 2. ANSWERABILITY & INSUFFICIENT EVIDENCE AUDIT ---")
    ans = pipe_res["answerable_queries"]
    ins = pipe_res["insufficient_evidence_queries"]
    print(f"Answerable Queries Accuracy:     {ans['accuracy']:.2%} ({ans['correctly_answered']}/{ans['total']})")
    print(f"Insufficient Evidence Accuracy:  {ins['accuracy']:.2%} ({ins['correctly_identified']}/{ins['total']})")

    # Table 3: Grounding & Citation Integrity
    print("\n--- 3. GROUNDING & CITATION INTEGRITY ---")
    g_met = pipe_res["grounding_metrics"]
    print(f"Citation Validity Rate:          {g_met['citation_validity_rate']:.4f}")
    print(f"Citation Coverage Rate:          {g_met['citation_coverage']:.4f}")
    print(f"Unsupported Citation Rate:       {g_met['unsupported_citation_rate']:.4f}")

    # Table 4: Baselines Comparison
    print("\n--- 4. BASELINES COMPARISON ---")
    header = f"{'Baseline':<32} | {'Gold Evidence Recall':<22} | {'Gen Evaluated':<14}"
    print(header)
    print("-" * len(header))
    for b_data in baselines.values():
        print(
            f"{b_data['description'][:32]:<32} | "
            f"{b_data['gold_evidence_recall']:<22.4f} | "
            f"{b_data['generation_evaluated']!s:<14}"
        )

    # Human Evaluation note
    print("\n--- 5. HUMAN ANSWER QUALITY ---")
    print(f"Status: {results['human_answer_quality']}")

    print("\n" + "=" * 80)
    print("EXP-005 RUN COMPLETE. Results saved to data/evaluation/exp005_results.json")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
