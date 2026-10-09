"""Script to run EXP-012: Dashboard Usability Evaluation Framework (Phase 9).

Usage:
    python scripts/run_exp012.py
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

from indiclearn.evaluation.exp012_runner import EXP012Runner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_exp012")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EXP-012: Dashboard Usability Evaluation Framework",
    )
    parser.add_argument(
        "--template-path",
        default="data/evaluation/exp012_usability_template.json",
        help="Path to usability evaluation template and response store",
    )
    parser.add_argument(
        "--output-path",
        default="data/evaluation/exp012_results.json",
        help="Path to write evaluation results JSON",
    )

    args = parser.parse_args()

    logger.info("Starting EXP-012 usability framework execution...")
    runner = EXP012Runner(
        template_path=args.template_path,
        output_results_path=args.output_path,
    )
    output = runner.run_all()
    meta = output["metadata"]
    val = meta["framework_validation"]

    print("\n" + "=" * 70)
    print("           EXP-012 EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Protocol Template:              {args.template_path}")
    print(f"Framework Valid:                {'YES' if val['framework_valid'] else 'NO'}")
    print(f"Tasks Specified:                {val['tasks_count']}")
    print(f"Likert Items:                   {val['likert_items_count']}")
    print(f"Human Participants Recorded:    {meta['sample_size']}")
    print(f"Status:                         {meta['status']}")
    print(f"Reason:                         {meta['reason']}")

    print("\n--- SCIENTIFIC POLICY ---")
    print(f"  {meta['scientific_disclaimer']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
