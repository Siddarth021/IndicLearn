"""Script to run EXP-009: Student Dashboard & Knowledge Visualization Evaluation.

Usage:
    python scripts/run_exp009.py
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

from indiclearn.evaluation.exp009_runner import EXP009Runner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_exp009")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EXP-009: Student Dashboard & Knowledge Visualization Evaluation",
    )
    parser.add_argument(
        "--data-path",
        default="data/evaluation/exp009_dashboard_eval.json",
        help="Path to dashboard evaluation benchmark dataset",
    )
    parser.add_argument(
        "--graph-path",
        default="data/evaluation/concept_graph_gold.json",
        help="Path to academic knowledge graph gold standard",
    )
    parser.add_argument(
        "--items-path",
        default="data/evaluation/signals_systems_items.json",
        help="Path to controlled assessment items question bank",
    )
    parser.add_argument(
        "--chunks-path",
        default="data/evaluation/signals_and_systems_eval.json",
        help="Path to curriculum chunks corpus",
    )
    parser.add_argument(
        "--output-path",
        default="data/evaluation/exp009_results.json",
        help="Path to write evaluation results JSON",
    )

    args = parser.parse_args()

    logger.info("Starting EXP-009 evaluation...")
    runner = EXP009Runner(
        eval_data_path=args.data_path,
        gold_graph_path=args.graph_path,
        items_data_path=args.items_path,
        chunks_data_path=args.chunks_path,
        output_results_path=args.output_path,
    )
    results = runner.run()

    metadata = results["metadata"]
    logger.info("EXP-009 Evaluation Results Summary:")
    logger.info("  Total Scenarios: %d", metadata["total_scenarios"])
    logger.info("  Passed Scenarios: %d", metadata["passed_scenarios"])
    logger.info("  Pass Rate: %.1f%%", metadata["pass_rate"] * 100)
    logger.info("  System Invariants: %s", results.get("system_invariants", {}))
    logger.info("  Scientific Disclaimer: %s", metadata["scientific_disclaimer"])

    if metadata["passed_scenarios"] < metadata["total_scenarios"]:
        logger.error("EXP-009 FAILED: Not all scenarios passed!")
        sys.exit(1)
    else:
        logger.info("EXP-009 PASSED successfully!")


if __name__ == "__main__":
    main()
