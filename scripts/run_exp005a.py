"""Script to run EXP-005A: Research-Hardened RAG Evaluation & Benchmark Expansion.

Usage:
    python scripts/run_exp005a.py
    python scripts/run_exp005a.py --provider real
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

from indiclearn.evaluation.exp005a_runner import Exp005aRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_exp005a")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EXP-005A: Research-Hardened RAG Evaluation & Benchmark Expansion",
    )
    parser.add_argument(
        "--provider",
        choices=["mock", "real"],
        default="mock",
        help="LLM provider mode: 'mock' (default, offline deterministic) or 'real' (OpenAI-compatible endpoint)",
    )
    parser.add_argument(
        "--eval-path",
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
        default="data/evaluation/exp005a_results.json",
        help="Path to write evaluation results JSON",
    )

    args = parser.parse_args()

    logger.info(f"Starting EXP-005A evaluation with provider mode='{args.provider}'...")
    runner = Exp005aRunner(
        eval_path=args.eval_path,
        corpus_path=args.corpus_path,
        concept_graph_path=args.graph_path,
        provider_mode=args.provider,
    )

    results = runner.run_experiment(output_path=args.output_path)

    # Print summary of results
    print("\n" + "=" * 70)
    print("           EXP-005A EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Benchmark Dataset:    {args.eval_path} ({len(runner.eval_data.get('queries', []))} queries)")
    print(f"Leakage Audit:        {results['benchmark_leakage_audit']['status']}")

    agg = results["retrieval_metrics"]["aggregate"]
    print("\n--- RETRIEVAL METRICS (Aggregate, 50 answerable queries) ---")
    print(f"  Recall@1:           {agg['recall_at_1']:.4f}")
    print(f"  Recall@3:           {agg['recall_at_3']:.4f}")
    print(f"  Recall@5:           {agg['recall_at_5']:.4f}")
    print(f"  MRR:                {agg['mrr']:.4f}")
    print(f"  Gold Recall@5:      {agg['gold_evidence_recall_at_5']:.4f}")
    print(f"  Multi-Chunk Recall: {agg.get('multi_chunk_evidence_recall', 0.0):.4f}")

    print("\n--- RETRIEVAL BY LANGUAGE ---")
    for lang, m in results["retrieval_metrics"]["by_language"].items():
        print(f"  {lang:<10} (n={m['queries_count']:2d}): R@1={m['recall_at_1']:.4f} | R@3={m['recall_at_3']:.4f} | R@5={m['recall_at_5']:.4f} | MRR={m['mrr']:.4f} | GoldR@5={m['gold_evidence_recall_at_5']:.4f}")

    print("\n--- RETRIEVAL BY DIFFICULTY ---")
    for diff, m in results["retrieval_metrics"]["by_difficulty"].items():
        print(f"  {diff:<10} (n={m['queries_count']:2d}): R@1={m['recall_at_1']:.4f} | R@3={m['recall_at_3']:.4f} | R@5={m['recall_at_5']:.4f} | MRR={m['mrr']:.4f} | GoldR@5={m['gold_evidence_recall_at_5']:.4f}")

    print("\n--- KNOWLEDGE GRAPH ABLATION (Section 21) ---")
    kg = results["ablations"]["knowledge_graph_ablation"]
    print(f"  System A (Hybrid only):       R@5={kg['system_a_hybrid_only']['recall_at_5']:.4f} | MRR={kg['system_a_hybrid_only']['mrr']:.4f} | GoldR@5={kg['system_a_hybrid_only']['gold_evidence_recall_at_5']:.4f}")
    print(f"  System B (+ Concept Mapping): R@5={kg['system_b_hybrid_plus_concept_mapping']['recall_at_5']:.4f} | MRR={kg['system_b_hybrid_plus_concept_mapping']['mrr']:.4f} | GoldR@5={kg['system_b_hybrid_plus_concept_mapping']['gold_evidence_recall_at_5']:.4f}")
    print(f"  System C (+ Graph Context):   R@5={kg['system_c_hybrid_plus_concept_mapping_plus_graph']['recall_at_5']:.4f} | MRR={kg['system_c_hybrid_plus_concept_mapping_plus_graph']['mrr']:.4f} | GoldR@5={kg['system_c_hybrid_plus_concept_mapping_plus_graph']['gold_evidence_recall_at_5']:.4f}")

    print("\n--- CONCEPT MAPPING QUERY ABLATION (Section 22) ---")
    cm = results["ablations"]["concept_mapping_query_ablation"]
    print(f"  1. Raw Query:                 R@5={cm['setting_1_raw_query']['recall_at_5']:.4f} | MRR={cm['setting_1_raw_query']['mrr']:.4f} | GoldR@5={cm['setting_1_raw_query']['gold_evidence_recall_at_5']:.4f}")
    print(f"  2. Normalized Query:          R@5={cm['setting_2_normalized_query']['recall_at_5']:.4f} | MRR={cm['setting_2_normalized_query']['mrr']:.4f} | GoldR@5={cm['setting_2_normalized_query']['gold_evidence_recall_at_5']:.4f}")
    print(f"  3. Normalized + Canonical:    R@5={cm['setting_3_normalized_plus_canonical_expansion']['recall_at_5']:.4f} | MRR={cm['setting_3_normalized_plus_canonical_expansion']['mrr']:.4f} | GoldR@5={cm['setting_3_normalized_plus_canonical_expansion']['gold_evidence_recall_at_5']:.4f}")

    print("\n--- ANSWERABILITY & CITATION INTEGRITY ---")
    p_and_a = results["pipeline_and_answerability"]
    am = p_and_a["answerability_matrix"]
    print(f"  Answerable (Sufficient & Grounded):        {am['answerable']['evidence_sufficient_and_grounded']}")
    print(f"  Answerable (Sufficient, Grounding Failed): {am['answerable']['evidence_sufficient_but_grounding_failed']}")
    print(f"  Answerable (Insufficient Evidence):        {am['answerable']['evidence_insufficient_refused']}")
    print(f"  Out-of-Scope (Correctly Refused):          {am['out_of_scope_and_adversarial']['correctly_refused']}")
    print(f"  Out-of-Scope (Hallucinated/Answered):      {am['out_of_scope_and_adversarial']['incorrectly_hallucinated_or_answered']}")

    ci = p_and_a["citation_integrity_metrics"]
    print(f"  Citation Validity Rate:                    {ci['citation_validity_rate']:.4f}")
    print(f"  Citation Coverage Rate:                    {ci['citation_coverage_rate']:.4f}")
    print(f"  Unsupported Citation Rate:                 {ci['unsupported_citation_rate']:.4f}")

    print("\n--- REAL LLM EVALUATION ---")
    print(f"  Status: {results['real_llm_evaluation']['status']}")

    print("\n--- HUMAN EVALUATION ---")
    print(f"  Status: {results['human_evaluation']['status']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
