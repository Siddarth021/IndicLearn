"""Script to run EXP-006: Student Knowledge Modeling & Evidence-Based Personalization Benchmark.

Usage:
    python scripts/run_exp006.py
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

from indiclearn.evaluation.exp006_runner import EXP006Runner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_exp006")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EXP-006: Student Knowledge Modeling Benchmark",
    )
    parser.add_argument(
        "--data-path",
        default="data/evaluation/exp006_student_eval.json",
        help="Path to student evaluation benchmark dataset",
    )
    parser.add_argument(
        "--output-path",
        default="data/evaluation/exp006_results.json",
        help="Path to write evaluation results JSON",
    )

    args = parser.parse_args()

    logger.info("Starting EXP-006 evaluation...")
    runner = EXP006Runner(
        eval_data_path=args.data_path,
        output_results_path=args.output_path,
    )
    results = runner.run_all()

    print("\n" + "=" * 70)
    print("           EXP-006 EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Benchmark Dataset:    {args.data_path}")
    print(f"Total Scenarios:      {results['total_scenarios']}")
    print(f"Scenarios Passed:     {results['scenarios_passed']} / {results['total_scenarios']} ({results['scenario_pass_rate'] * 100:.1f}%)")
    print(f"Invariants Checked:   {results['total_invariants_evaluated']}")
    print(f"Invariants Passed:    {results['invariants_passed']} / {results['total_invariants_evaluated']} ({results['invariant_pass_rate'] * 100:.1f}%)")
    print("\n--- SCENARIO BREAKDOWN ---")
    for sc in results["scenario_details"]:
        status_str = "PASS" if sc["all_invariants_passed"] else "FAIL"
        print(f"  [{status_str}] {sc['scenario_id']:<32} | Score={sc['final_knowledge_score']:.4f} | Conf={sc['final_confidence']:.4f} | Status={sc['final_mastery_status']}")

    print("\n--- SCIENTIFIC POLICY ---")
    print(f"  {results['scientific_disclaimer']}")

    print("\nOverall Status: " + results["status"])
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
