"""Unit tests for NetworkXGraphStore implementation (Phase 3)."""

from __future__ import annotations

import pytest

from indiclearn.domain.concepts import (
    Concept,
    ConceptRelation,
    ConceptType,
    PrerequisiteNature,
    RelationType,
)
from indiclearn.knowledge_graph.networkx_store import NetworkXGraphStore


@pytest.fixture()
def store() -> NetworkXGraphStore:
    return NetworkXGraphStore(default_subject_id="signals_and_systems")


@pytest.fixture()
def sample_concepts() -> list[Concept]:
    c1 = Concept(
        concept_id="cpt_signals_and_systems_lti_system",
        canonical_name="LTI System",
        concept_type=ConceptType.CONCEPT,
        aliases=["LTI", "linear time invariant system"],
        subject_id="signals_and_systems",
        source_chunk_ids=["chunk_001"],
    )
    c2 = Concept(
        concept_id="cpt_signals_and_systems_convolution",
        canonical_name="Convolution",
        concept_type=ConceptType.METHOD,
        aliases=["convolution integral", "కన్వల్యూషన్"],
        subject_id="signals_and_systems",
        source_chunk_ids=["chunk_001", "chunk_004"],
    )
    c3 = Concept(
        concept_id="cpt_signals_and_systems_linearity",
        canonical_name="Linearity",
        concept_type=ConceptType.PROPERTY,
        subject_id="signals_and_systems",
        source_chunk_ids=["chunk_002"],
    )
    return [c1, c2, c3]


class TestNetworkXGraphStore:
    """Tests covering GraphStore protocol operations."""

    def test_add_and_get_concept(self, store: NetworkXGraphStore, sample_concepts: list[Concept]) -> None:
        c1 = sample_concepts[0]
        store.add_concept(c1)
        retrieved = store.get_concept(c1.concept_id)
        assert retrieved is not None
        assert retrieved.canonical_name == "LTI System"
        assert retrieved.subject_id == "signals_and_systems"
        assert "LTI" in retrieved.aliases

    def test_get_nonexistent_concept_returns_none(self, store: NetworkXGraphStore) -> None:
        assert store.get_concept("cpt_nonexistent") is None

    def test_duplicate_concept_add_merges_provenance_and_aliases(
        self, store: NetworkXGraphStore
    ) -> None:
        c1 = Concept(
            concept_id="cpt_signals_and_systems_convolution",
            canonical_name="Convolution",
            aliases=["convolution operation"],
            subject_id="signals_and_systems",
            source_chunk_ids=["chunk_001"],
        )
        c2 = Concept(
            concept_id="cpt_signals_and_systems_convolution",
            canonical_name="Convolution",
            aliases=["కన్వల్యూషన్"],
            subject_id="signals_and_systems",
            source_chunk_ids=["chunk_004"],
        )
        store.add_concept(c1)
        store.add_concept(c2)

        merged = store.get_concept("cpt_signals_and_systems_convolution")
        assert merged is not None
        assert set(merged.source_chunk_ids) == {"chunk_001", "chunk_004"}
        assert "convolution operation" in merged.aliases
        assert "కన్వల్యూషన్" in merged.aliases
        assert store.concept_count() == 1

    def test_delete_concept_cascades_relations(
        self, store: NetworkXGraphStore, sample_concepts: list[Concept]
    ) -> None:
        c1, c2, _ = sample_concepts
        store.add_concept(c1)
        store.add_concept(c2)
        rel = ConceptRelation(
            relation_id="rel_lti_conv",
            source_concept_id=c1.concept_id,
            target_concept_id=c2.concept_id,
            relation_type=RelationType.PREREQUISITE_OF,
            subject_id="signals_and_systems",
        )
        store.add_relation(rel)
        assert store.relation_count() == 1

        deleted = store.delete_concept(c1.concept_id)
        assert deleted is True
        assert store.get_concept(c1.concept_id) is None
        assert store.relation_count() == 0

    def test_add_and_get_relations(
        self, store: NetworkXGraphStore, sample_concepts: list[Concept]
    ) -> None:
        c1, c2, c3 = sample_concepts
        for c in sample_concepts:
            store.add_concept(c)

        rel1 = ConceptRelation(
            relation_id="rel_1",
            source_concept_id=c3.concept_id,  # Linearity
            target_concept_id=c1.concept_id,  # LTI System
            relation_type=RelationType.PREREQUISITE_OF,
            subject_id="signals_and_systems",
        )
        rel2 = ConceptRelation(
            relation_id="rel_2",
            source_concept_id=c1.concept_id,  # LTI System
            target_concept_id=c2.concept_id,  # Convolution
            relation_type=RelationType.USED_IN,
            subject_id="signals_and_systems",
        )
        store.add_relation(rel1)
        store.add_relation(rel2)

        assert store.relation_count() == 2
        rels_c1 = store.get_relations(c1.concept_id)
        assert len(rels_c1) == 2

    def test_duplicate_relation_add_deduplicates(
        self, store: NetworkXGraphStore, sample_concepts: list[Concept]
    ) -> None:
        c1, c2, _ = sample_concepts
        store.add_concept(c1)
        store.add_concept(c2)

        rel1 = ConceptRelation(
            relation_id="rel_1",
            source_concept_id=c1.concept_id,
            target_concept_id=c2.concept_id,
            relation_type=RelationType.USED_IN,
            subject_id="signals_and_systems",
            confidence=0.8,
            source_chunk_ids=["chunk_001"],
        )
        rel2 = ConceptRelation(
            relation_id="rel_1",
            source_concept_id=c1.concept_id,
            target_concept_id=c2.concept_id,
            relation_type=RelationType.USED_IN,
            subject_id="signals_and_systems",
            confidence=0.95,
            source_chunk_ids=["chunk_002"],
        )
        store.add_relation(rel1)
        store.add_relation(rel2)

        assert store.relation_count() == 1
        stored_rel = store.get_relations(c1.concept_id)[0]
        assert stored_rel.confidence == 0.95
        assert set(stored_rel.source_chunk_ids) == {"chunk_001", "chunk_002"}

    def test_prerequisites_and_dependents(
        self, store: NetworkXGraphStore, sample_concepts: list[Concept]
    ) -> None:
        c1, c2, c3 = sample_concepts
        for c in sample_concepts:
            store.add_concept(c)

        # Linearity (c3) -> PREREQUISITE_OF -> LTI System (c1)
        # LTI System (c1) -> PREREQUISITE_OF -> Convolution (c2)
        store.add_relation(
            ConceptRelation(
                relation_id="r1",
                source_concept_id=c3.concept_id,
                target_concept_id=c1.concept_id,
                relation_type=RelationType.PREREQUISITE_OF,
                subject_id="signals_and_systems",
            )
        )
        store.add_relation(
            ConceptRelation(
                relation_id="r2",
                source_concept_id=c1.concept_id,
                target_concept_id=c2.concept_id,
                relation_type=RelationType.PREREQUISITE_OF,
                subject_id="signals_and_systems",
            )
        )

        # Direct prerequisites of Convolution (c2) is LTI System (c1)
        direct_prereqs = store.get_prerequisites(c2.concept_id, transitive=False)
        assert len(direct_prereqs) == 1
        assert direct_prereqs[0].concept_id == c1.concept_id

        # Transitive prerequisites of Convolution (c2) includes LTI System (c1) and Linearity (c3)
        trans_prereqs = store.get_prerequisites(c2.concept_id, transitive=True)
        prereq_ids = {c.concept_id for c in trans_prereqs}
        assert prereq_ids == {c1.concept_id, c3.concept_id}

        # Transitive dependents of Linearity (c3) includes LTI System and Convolution
        trans_deps = store.get_dependents(c3.concept_id, transitive=True)
        dep_ids = {c.concept_id for c in trans_deps}
        assert dep_ids == {c1.concept_id, c2.concept_id}

    def test_find_concept_by_name_and_alias(
        self, store: NetworkXGraphStore, sample_concepts: list[Concept]
    ) -> None:
        for c in sample_concepts:
            store.add_concept(c)

        # Direct canonical name
        found = store.find_concept("lti system")
        assert found is not None
        assert found.canonical_name == "LTI System"

        # English alias
        found_alias = store.find_concept("linear time invariant system")
        assert found_alias is not None
        assert found_alias.canonical_name == "LTI System"

        # Telugu script alias
        found_telugu = store.find_concept("కన్వల్యూషన్")
        assert found_telugu is not None
        assert found_telugu.canonical_name == "Convolution"

    def test_search_concepts(
        self, store: NetworkXGraphStore, sample_concepts: list[Concept]
    ) -> None:
        for c in sample_concepts:
            store.add_concept(c)

        results = store.search_concepts("system")
        assert len(results) >= 1
        assert any(c.canonical_name == "LTI System" for c in results)

    def test_find_path_between_concepts(
        self, store: NetworkXGraphStore, sample_concepts: list[Concept]
    ) -> None:
        c1, c2, c3 = sample_concepts
        for c in sample_concepts:
            store.add_concept(c)

        store.add_relation(
            ConceptRelation(
                relation_id="r1",
                source_concept_id=c3.concept_id,
                target_concept_id=c1.concept_id,
                relation_type=RelationType.PREREQUISITE_OF,
                subject_id="signals_and_systems",
            )
        )
        store.add_relation(
            ConceptRelation(
                relation_id="r2",
                source_concept_id=c1.concept_id,
                target_concept_id=c2.concept_id,
                relation_type=RelationType.USED_IN,
                subject_id="signals_and_systems",
            )
        )

        path = store.find_path(c3.concept_id, c2.concept_id)
        assert path == [c3.concept_id, c1.concept_id, c2.concept_id]

    def test_subject_isolation(self, store: NetworkXGraphStore) -> None:
        c_ss = Concept(
            concept_id="cpt_ss_filter",
            canonical_name="Filter",
            subject_id="signals_and_systems",
        )
        c_dl = Concept(
            concept_id="cpt_dl_filter",
            canonical_name="Filter",
            subject_id="digital_logic",
        )
        store.add_concept(c_ss)
        store.add_concept(c_dl)

        ss_concepts = store.get_all_concepts(subject_id="signals_and_systems")
        dl_concepts = store.get_all_concepts(subject_id="digital_logic")

        assert len(ss_concepts) == 1
        assert ss_concepts[0].concept_id == "cpt_ss_filter"
        assert len(dl_concepts) == 1
        assert dl_concepts[0].concept_id == "cpt_dl_filter"

    def test_persistence_json_roundtrip(
        self, store: NetworkXGraphStore, sample_concepts: list[Concept], tmp_path
    ) -> None:
        for c in sample_concepts:
            store.add_concept(c)

        store.add_relation(
            ConceptRelation(
                relation_id="r1",
                source_concept_id=sample_concepts[0].concept_id,
                target_concept_id=sample_concepts[1].concept_id,
                relation_type=RelationType.PREREQUISITE_OF,
                subject_id="signals_and_systems",
                confidence=0.88,
                prerequisite_nature=PrerequisiteNature.EXPLICIT_PREREQUISITE,
                source_chunk_ids=["chunk_001"],
            )
        )

        save_path = tmp_path / "academic_kg.json"
        store.save_to_json(save_path)

        new_store = NetworkXGraphStore(default_subject_id="signals_and_systems")
        new_store.load_from_json(save_path)

        assert new_store.concept_count() == 3
        assert new_store.relation_count() == 1

        loaded_c = new_store.get_concept(sample_concepts[0].concept_id)
        assert loaded_c is not None
        assert loaded_c.canonical_name == "LTI System"

        loaded_r = new_store.get_relations(sample_concepts[0].concept_id)[0]
        assert loaded_r.relation_type == RelationType.PREREQUISITE_OF
        assert loaded_r.confidence == 0.88
        assert loaded_r.prerequisite_nature == PrerequisiteNature.EXPLICIT_PREREQUISITE
