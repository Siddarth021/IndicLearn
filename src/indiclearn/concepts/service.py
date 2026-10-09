"""Concept service layer orchestrating extraction, normalization, and graph storage (Phase 3)."""

from __future__ import annotations

from pathlib import Path

from indiclearn.concepts.extractor import ConceptExtractor, HeuristicConceptExtractor
from indiclearn.domain.concepts import Concept, ConceptRelation
from indiclearn.domain.documents import DocumentChunk
from indiclearn.knowledge_graph.base import GraphStore
from indiclearn.knowledge_graph.networkx_store import NetworkXGraphStore
from indiclearn.knowledge_graph.validation import ValidationReport, validate_graph
from indiclearn.utils.logging import get_logger, log_event

logger = get_logger("concepts.service")


class ConceptService:
    """Orchestrates concept extraction, normalization, deduplication, and graph storage."""

    def __init__(
        self,
        extractor: ConceptExtractor | None = None,
        graph_store: GraphStore | None = None,
        subject_id: str = "signals_and_systems",
    ) -> None:
        self.subject_id = subject_id
        self.extractor = extractor or HeuristicConceptExtractor(subject_id=subject_id)
        self.graph_store = graph_store or NetworkXGraphStore(default_subject_id=subject_id)

    def process_chunks(
        self, chunks: list[DocumentChunk]
    ) -> tuple[list[Concept], list[ConceptRelation], ValidationReport]:
        """Extract concepts and relations from chunks, populate graph, and run validation."""
        log_event(logger, "concept_extraction_started", chunk_count=len(chunks), subject_id=self.subject_id)

        concepts, relations = self.extractor.extract(chunks)

        # Store in graph
        for cpt in concepts:
            self.graph_store.add_concept(cpt)

        for rel in relations:
            self.graph_store.add_relation(rel)

        # Validate graph
        validation = validate_graph(self.graph_store, subject_id=self.subject_id)

        log_event(
            logger,
            "concept_extraction_completed",
            concepts_extracted=len(concepts),
            relations_extracted=len(relations),
            is_valid=validation.is_valid,
            errors_count=len(validation.errors),
        )

        return concepts, relations, validation

    def link_chunks_to_concepts(
        self, chunks: list[DocumentChunk]
    ) -> list[DocumentChunk]:
        """Update each chunk's concept_ids field with matching concepts from the graph store."""
        # Build mapping: chunk_id -> list[concept_id]
        chunk_to_cpts: dict[str, list[str]] = {}
        for cpt in self.graph_store.list_concepts(subject_id=self.subject_id):
            for chk_id in cpt.source_chunk_ids:
                chunk_to_cpts.setdefault(chk_id, []).append(cpt.concept_id)

        updated_chunks: list[DocumentChunk] = []
        for chk in chunks:
            mapped_ids = sorted(set(chk.concept_ids + chunk_to_cpts.get(chk.chunk_id, [])))
            updated_chunks.append(chk.model_copy(update={"concept_ids": mapped_ids}))

        return updated_chunks

    def get_concept(self, concept_id: str) -> Concept | None:
        return self.graph_store.get_concept(concept_id)

    def find_concept(self, query: str) -> Concept | None:
        return self.graph_store.find_concept_by_alias(query, subject_id=self.subject_id)

    def get_prerequisites(self, concept_id: str, transitive: bool = False) -> list[Concept]:
        return self.graph_store.get_prerequisites(concept_id, transitive=transitive)

    def get_dependents(self, concept_id: str, transitive: bool = False) -> list[Concept]:
        return self.graph_store.get_dependents(concept_id, transitive=transitive)

    def save_graph(self, path: Path | str) -> None:
        self.graph_store.save(path)

    def load_graph(self, path: Path | str) -> None:
        self.graph_store.load(path)
