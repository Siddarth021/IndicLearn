"""Script to run EXP-007: Adaptive Assessment Selection Benchmark.

Usage:
    python scripts/run_exp007.py
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

from indiclearn.evaluation.exp007_runner import EXP007Runner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_exp007")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EXP-007: Adaptive Assessment Selection Benchmark",
    )
    parser.add_argument(
        "--data-path",
        default="data/evaluation/exp007_assessment_eval.json",
        help="Path to assessment evaluation benchmark dataset",
    )
    parser.add_argument(
        "--items-path",
        default="data/evaluation/signals_systems_items.json",
        help="Path to controlled assessment items question bank",
    )
    parser.add_argument(
        "--graph-path",
        default="data/evaluation/concept_graph_gold.json",
        help="Path to academic knowledge graph gold standard",
    )
    parser.add_argument(
        "--output-path",
        default="data/evaluation/exp007_results.json",
        help="Path to write evaluation results JSON",
    )

    args = parser.parse_args()

    logger.info("Starting EXP-007 evaluation...")
    runner = EXP007Runner(
        eval_data_path=args.data_path,
        items_data_path=args.items_path,
        gold_graph_path=args.graph_path,
        output_results_path=args.output_path,
    )
    results = runner.run_all()

    print("\n" + "=" * 70)
    print("           EXP-007 EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Benchmark Dataset:    {args.data_path}")
    print(f"Total Scenarios:      {results['total_scenarios']}")
    print(f"Scenarios Passed:     {results['scenarios_passed']} / {results['total_scenarios']} ({results['scenario_pass_rate'] * 100:.1f}%)")
    print(f"Invariants Checked:   {results['total_invariants_evaluated']}")
    print(f"Invariants Passed:    {results['invariants_passed']} / {results['total_invariants_evaluated']} ({results['invariant_pass_rate'] * 100:.1f}%)")
    print("\n--- SCENARIO BREAKDOWN ---")
    for sc in results["scenario_details"]:
        status_str = "PASS" if sc["all_invariants_passed"] else "FAIL"
        print(f"  [{status_str}] {sc['scenario_id']:<36} | {sc['name']}")

    print("\n--- SCIENTIFIC POLICY ---")
    print(f"  {results['scientific_disclaimer']}")

    print("\nOverall Status: " + results["status"])
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
