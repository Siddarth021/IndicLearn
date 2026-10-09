"""EXP-003: Academic Concept and Relation Extraction Experiment Runner (Phase 3).

Evaluates baseline concept and relation extraction against the manual gold standard.
Measures Concept Precision, Recall, F1 and Relation Precision, Recall, F1.
Produces detailed error analysis documenting missed and spurious extractions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from indiclearn.concepts.extractor import HeuristicConceptExtractor
from indiclearn.concepts.service import ConceptService
from indiclearn.domain.documents import DocumentChunk
from indiclearn.evaluation.graph_metrics import (
    GraphEvaluationReport,
    evaluate_graph_extraction,
)
from indiclearn.knowledge_graph.networkx_store import NetworkXGraphStore


class Exp003Runner:
    """Orchestrates EXP-003 comparative evaluation on concept graph extraction."""

    def __init__(
        self,
        corpus_path: Path | str,
        gold_path: Path | str,
        subject_id: str = "signals_and_systems",
    ) -> None:
        self.subject_id = subject_id
        self.corpus_path = Path(corpus_path)
        self.gold_path = Path(gold_path)

        with open(self.corpus_path, encoding="utf-8") as f:
            corpus_data = json.load(f)
        self.chunks = [DocumentChunk(**c) for c in corpus_data["corpus"]]

        with open(self.gold_path, encoding="utf-8") as f:
            self.gold_data = json.load(f)

    def run_baseline_evaluation(self) -> dict[str, Any]:
        """Execute heuristic baseline extraction and evaluate against gold standard."""
        store = NetworkXGraphStore(default_subject_id=self.subject_id)
        extractor = HeuristicConceptExtractor(subject_id=self.subject_id)
        service = ConceptService(extractor=extractor, graph_store=store, subject_id=self.subject_id)

        concepts, relations, validation = service.process_chunks(self.chunks)

        report: GraphEvaluationReport = evaluate_graph_extraction(
            extracted_concepts=concepts,
            extracted_relations=relations,
            gold_data=self.gold_data,
        )

        return {
            "experiment_id": "EXP-003",
            "subject_id": self.subject_id,
            "extractor_name": "HeuristicConceptExtractor (Baseline)",
            "chunks_processed": len(self.chunks),
            "concepts_extracted_count": len(concepts),
            "relations_extracted_count": len(relations),
            "validation_report": validation.model_dump(mode="json"),
            "concept_metrics": report.concept_metrics.model_dump(),
            "relation_metrics": report.relation_metrics.model_dump(),
            "error_analysis": {
                "missed_concepts": report.missed_concepts,
                "spurious_concepts": report.spurious_concepts,
                "missed_relations": report.missed_relations,
                "spurious_relations": report.spurious_relations,
            },
        }
