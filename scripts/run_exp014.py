"""Script to run EXP-014: Multilingual Human Evaluation (Phase 9).

Usage:
    python scripts/run_exp014.py
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

from indiclearn.evaluation.exp014_runner import EXP014Runner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_exp014")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EXP-014: Multilingual Human Evaluation",
    )
    parser.add_argument(
        "--template-path",
        default="data/evaluation/exp014_multilingual_template.json",
        help="Path to multilingual evaluation template and response store",
    )
    parser.add_argument(
        "--output-path",
        default="data/evaluation/exp014_results.json",
        help="Path to write evaluation results JSON",
    )

    args = parser.parse_args()

    logger.info("Starting EXP-014 multilingual human evaluation runner...")
    runner = EXP014Runner(
        template_path=args.template_path,
        output_results_path=args.output_path,
    )
    output = runner.run_all()
    meta = output["metadata"]

    print("\n" + "=" * 70)
    print("           EXP-014 EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Evaluation Template:            {args.template_path}")
    print(f"Multilingual Evaluations:       {meta['sample_size']}")
    print(f"Status:                         {meta['status']}")
    print(f"Reason:                         {meta['reason']}")

    print("\n--- SCIENTIFIC POLICY ---")
    print(f"  {meta['scientific_disclaimer']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
