"""Script to run EXP-013: RAG Human Annotation Analysis (Phase 9).

Usage:
    python scripts/run_exp013.py
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

from indiclearn.evaluation.exp013_runner import EXP013Runner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_exp013")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EXP-013: RAG Human Annotation Analysis",
    )
    parser.add_argument(
        "--template-path",
        default="data/evaluation/exp013_annotation_template.json",
        help="Path to annotation template and response store",
    )
    parser.add_argument(
        "--output-path",
        default="data/evaluation/exp013_results.json",
        help="Path to write evaluation results JSON",
    )

    args = parser.parse_args()

    logger.info("Starting EXP-013 human annotation analysis runner...")
    runner = EXP013Runner(
        template_path=args.template_path,
        output_results_path=args.output_path,
    )
    output = runner.run_all()
    meta = output["metadata"]

    print("\n" + "=" * 70)
    print("           EXP-013 EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Annotation Store:               {args.template_path}")
    print(f"Human Annotations Recorded:     {meta['sample_size']}")
    print(f"Status:                         {meta['status']}")
    print(f"Reason:                         {meta['reason']}")

    print("\n--- SCIENTIFIC POLICY ---")
    print(f"  {meta['scientific_disclaimer']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
