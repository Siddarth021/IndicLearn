"""Script to run EXP-011: Controlled Adaptive Assessment Evaluation (Phase 9).

Usage:
    python scripts/run_exp011.py
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

from indiclearn.evaluation.exp011_runner import EXP011Runner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_exp011")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EXP-011: Controlled Adaptive Assessment Evaluation",
    )
    parser.add_argument(
        "--data-path",
        default="data/evaluation/exp011_assessment_eval.json",
        help="Path to EXP-011 benchmark dataset",
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
        default="data/evaluation/exp011_results.json",
        help="Path to write evaluation results JSON",
    )

    args = parser.parse_args()

    logger.info("Starting EXP-011 evaluation...")
    runner = EXP011Runner(
        eval_data_path=args.data_path,
        items_data_path=args.items_path,
        gold_graph_path=args.graph_path,
        output_results_path=args.output_path,
    )
    output = runner.run_all()
    meta = output["metadata"]
    scenarios = output["scenarios"]

    passed_scenarios = sum(1 for s in scenarios if s.get("passed", False))

    print("\n" + "=" * 70)
    print("           EXP-011 EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Benchmark Dataset:              {args.data_path}")
    print(f"Evaluation Mode:                {meta['evaluation_mode']}")
    print(f"Total Scenarios:                {meta['total_scenarios']}")
    print(f"Scenarios Passed:               {passed_scenarios} / {meta['total_scenarios']} ({meta['scenario_pass_rate'] * 100:.1f}%)")
    print("\n--- SCENARIO BREAKDOWN ---")
    for sc in scenarios:
        status_str = "PASS" if sc.get("passed", False) else "FAIL"
        print(f"  [{status_str}] {sc['scenario_id']}")

    print("\n--- SCIENTIFIC POLICY ---")
    print(f"  {meta['scientific_disclaimer']}")

    print(f"\nOverall Status: {meta['status']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
