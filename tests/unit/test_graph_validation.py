"""Unit tests for graph validation and integrity checking (Phase 3)."""

from __future__ import annotations

import pytest

from indiclearn.domain.concepts import (
    Concept,
    ConceptRelation,
    RelationType,
)
from indiclearn.knowledge_graph.networkx_store import NetworkXGraphStore
from indiclearn.knowledge_graph.validation import validate_graph


@pytest.fixture()
def valid_store() -> NetworkXGraphStore:
    store = NetworkXGraphStore(default_subject_id="signals_and_systems")
    c1 = Concept(
        concept_id="cpt_ss_lti",
        canonical_name="LTI System",
        subject_id="signals_and_systems",
        source_chunk_ids=["chunk_001"],
    )
    c2 = Concept(
        concept_id="cpt_ss_conv",
        canonical_name="Convolution",
        subject_id="signals_and_systems",
        source_chunk_ids=["chunk_001"],
    )
    rel = ConceptRelation(
        relation_id="rel_lti_conv",
        source_concept_id=c1.concept_id,
        target_concept_id=c2.concept_id,
        relation_type=RelationType.PREREQUISITE_OF,
        subject_id="signals_and_systems",
        source_chunk_ids=["chunk_001"],
    )
    store.add_concept(c1)
    store.add_concept(c2)
    store.add_relation(rel)
    return store


class TestGraphValidation:
    """Tests for referential integrity, self-loops, duplicates, and cycles."""

    def test_valid_graph_passes(self, valid_store: NetworkXGraphStore) -> None:
        report = validate_graph(valid_store, subject_id="signals_and_systems")
        assert report.is_valid is True
        assert len(report.errors) == 0
        assert len(report.prerequisite_cycles) == 0

    def test_missing_concept_reference_detected(self) -> None:
        store = NetworkXGraphStore(default_subject_id="signals_and_systems")
        c1 = Concept(
            concept_id="cpt_ss_c1",
            canonical_name="Concept 1",
            subject_id="signals_and_systems",
            source_chunk_ids=["chunk_001"],
        )
        store.add_concept(c1)

        # Relation references missing target cpt_ss_missing
        rel = ConceptRelation(
            relation_id="rel_bad",
            source_concept_id=c1.concept_id,
            target_concept_id="cpt_ss_missing",
            relation_type=RelationType.RELATED_TO,
            subject_id="signals_and_systems",
            source_chunk_ids=["chunk_001"],
        )
        # Directly insert into the underlying networkx graph to simulate corrupted data
        store.graph.add_edge(c1.concept_id, "cpt_ss_missing", key=rel.relation_id, relation=rel)

        report = validate_graph(store, subject_id="signals_and_systems")
        assert report.is_valid is False
        assert any("references non-existent target concept 'cpt_ss_missing'" in e for e in report.errors)

    def test_self_loop_detected(self) -> None:
        store = NetworkXGraphStore(default_subject_id="signals_and_systems")
        c1 = Concept(
            concept_id="cpt_ss_loop",
            canonical_name="Loop Concept",
            subject_id="signals_and_systems",
            source_chunk_ids=["chunk_001"],
        )
        store.add_concept(c1)

        rel = ConceptRelation(
            relation_id="rel_self",
            source_concept_id=c1.concept_id,
            target_concept_id=c1.concept_id,
            relation_type=RelationType.RELATED_TO,
            subject_id="signals_and_systems",
            source_chunk_ids=["chunk_001"],
        )
        store.graph.add_edge(c1.concept_id, c1.concept_id, key=rel.relation_id, relation=rel)

        report = validate_graph(store, subject_id="signals_and_systems")
        assert report.is_valid is False
        assert any("self-loop detected" in e.lower() for e in report.errors)

    def test_prerequisite_cycle_detected(self) -> None:
        store = NetworkXGraphStore(default_subject_id="signals_and_systems")
        c1 = Concept(
            concept_id="cpt_a",
            canonical_name="Concept A",
            subject_id="signals_and_systems",
            source_chunk_ids=["chunk_001"],
        )
        c2 = Concept(
            concept_id="cpt_b",
            canonical_name="Concept B",
            subject_id="signals_and_systems",
            source_chunk_ids=["chunk_001"],
        )
        store.add_concept(c1)
        store.add_concept(c2)

        # Cycle: A -> PREREQUISITE_OF -> B -> PREREQUISITE_OF -> A
        rel1 = ConceptRelation(
            relation_id="r1",
            source_concept_id=c1.concept_id,
            target_concept_id=c2.concept_id,
            relation_type=RelationType.PREREQUISITE_OF,
            subject_id="signals_and_systems",
            source_chunk_ids=["chunk_001"],
        )
        rel2 = ConceptRelation(
            relation_id="r2",
            source_concept_id=c2.concept_id,
            target_concept_id=c1.concept_id,
            relation_type=RelationType.PREREQUISITE_OF,
            subject_id="signals_and_systems",
            source_chunk_ids=["chunk_001"],
        )
        store.add_relation(rel1)
        store.add_relation(rel2)

        report = validate_graph(store, subject_id="signals_and_systems")
        assert len(report.prerequisite_cycles) > 0
        assert any("Prerequisite cycle detected" in w for w in report.warnings)

    def test_missing_provenance_produces_warning(self) -> None:
        store = NetworkXGraphStore(default_subject_id="signals_and_systems")
        c1 = Concept(
            concept_id="cpt_no_prov",
            canonical_name="No Provenance Concept",
            subject_id="signals_and_systems",
            source_chunk_ids=[],  # Missing!
        )
        store.add_concept(c1)

        report = validate_graph(store, subject_id="signals_and_systems")
        assert report.is_valid is True  # Warning does not invalidate graph
        assert any("has no source chunk provenance" in w for w in report.warnings)
