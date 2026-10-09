"""Integration tests for the complete Phase 3 Concept Graph pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from indiclearn.concepts.extractor import HeuristicConceptExtractor
from indiclearn.concepts.service import ConceptService
from indiclearn.domain.documents import DocumentChunk
from indiclearn.evaluation.exp003_runner import Exp003Runner
from indiclearn.knowledge_graph.networkx_store import NetworkXGraphStore


@pytest.fixture()
def eval_chunks() -> list[DocumentChunk]:
    corpus_path = Path("data/evaluation/signals_and_systems_eval.json")
    with open(corpus_path, encoding="utf-8") as f:
        data = json.load(f)
    return [DocumentChunk(**c) for c in data["corpus"]]


class TestGraphPipelineIntegration:
    """Integration test suite for extraction, graph population, validation, and serialization."""

    def test_full_pipeline_process_chunks(self, eval_chunks: list[DocumentChunk], tmp_path) -> None:
        store = NetworkXGraphStore(default_subject_id="signals_and_systems")
        extractor = HeuristicConceptExtractor(subject_id="signals_and_systems")
        service = ConceptService(extractor=extractor, graph_store=store, subject_id="signals_and_systems")

        concepts, relations, validation = service.process_chunks(eval_chunks)

        # 1. Extracted concepts and relations should be non-empty
        assert len(concepts) >= 15
        assert len(relations) >= 8

        # 2. Validation report should be valid
        assert validation.is_valid is True
        assert len(validation.errors) == 0

        # 3. Chunks must have concept_ids populated
        chunks_with_concepts = [c for c in eval_chunks if len(c.concept_ids) > 0]
        assert len(chunks_with_concepts) == len(eval_chunks)

        # 4. Graph queries work as expected
        conv = store.find_concept("Convolution")
        assert conv is not None
        assert "cpt_signals_and_systems_convolution" in conv.concept_id

        lti = store.find_concept("LTI System")
        assert lti is not None

        # 5. Graph persistence round-trip
        save_file = tmp_path / "saved_kg.json"
        store.save_to_json(save_file)
        assert save_file.exists()

        reloaded_store = NetworkXGraphStore(default_subject_id="signals_and_systems")
        reloaded_store.load_from_json(save_file)
        assert reloaded_store.concept_count() == store.concept_count()
        assert reloaded_store.relation_count() == store.relation_count()

    def test_exp003_runner_executes_successfully(self) -> None:
        runner = Exp003Runner(
            corpus_path="data/evaluation/signals_and_systems_eval.json",
            gold_path="data/evaluation/concept_graph_gold.json",
            subject_id="signals_and_systems",
        )
        results = runner.run_baseline_evaluation()

        assert results["experiment_id"] == "EXP-003"
        assert results["validation_report"]["is_valid"] is True
        assert results["concept_metrics"]["recall"] > 0.90
        assert results["relation_metrics"]["recall"] > 0.50
