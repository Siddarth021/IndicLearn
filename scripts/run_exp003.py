"""Executable script for EXP-003: Academic Concept and Relation Extraction.

Runs evaluation of baseline heuristic concept and relation extractor against
the Signals and Systems manual gold standard. Outputs metrics and error analysis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add src to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indiclearn.evaluation.exp003_runner import Exp003Runner


def main() -> None:
    corpus_path = ROOT / "data" / "evaluation" / "signals_and_systems_eval.json"
    gold_path = ROOT / "data" / "evaluation" / "concept_graph_gold.json"

    print("--- Running EXP-003: Academic Concept & Relation Extraction ---")
    runner = Exp003Runner(corpus_path=corpus_path, gold_path=gold_path)
    results = runner.run_baseline_evaluation()

    # Save output JSON
    out_file = ROOT / "data" / "evaluation" / "exp003_results.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved evaluation results to {out_file}\n")

    cm = results["concept_metrics"]
    rm = results["relation_metrics"]
    vr = results["validation_report"]

    print("=" * 80)
    print("EXP-003 BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Extractor: {results['extractor_name']}")
    print(f"Subject:   {results['subject_id']}")
    print(f"Chunks:    {results['chunks_processed']}")
    print(f"Graph Integrity Valid: {vr['is_valid']} (Errors: {len(vr['errors'])}, Warnings: {len(vr['warnings'])})")
    print("-" * 80)
    print(f"{'Category':<20} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'TP':<4} | {'FP':<4} | {'FN':<4}")
    print("-" * 80)
    print(f"{'Concept Extraction':<20} | {cm['precision']:<10.4f} | {cm['recall']:<10.4f} | {cm['f1']:<10.4f} | {cm['true_positives']:<4} | {cm['false_positives']:<4} | {cm['false_negatives']:<4}")
    print(f"{'Relation Extraction':<20} | {rm['precision']:<10.4f} | {rm['recall']:<10.4f} | {rm['f1']:<10.4f} | {rm['true_positives']:<4} | {rm['false_positives']:<4} | {rm['false_negatives']:<4}")
    print("=" * 80)

    ea = results["error_analysis"]
    print("\n--- Error Analysis ---")
    print(f"Missed Concepts ({len(ea['missed_concepts'])}): {ea['missed_concepts']}")
    print(f"Spurious/Extra Concepts ({len(ea['spurious_concepts'])}): {ea['spurious_concepts']}")
    print(f"Missed Relations ({len(ea['missed_relations'])}): {ea['missed_relations']}")
    print(f"Spurious Relations ({len(ea['spurious_relations'])}): {ea['spurious_relations']}")


if __name__ == "__main__":
    main()
