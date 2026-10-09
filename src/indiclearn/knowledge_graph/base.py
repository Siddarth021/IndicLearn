"""Base protocol and interfaces for Knowledge Graph storage (Phase 3).

Decouples concept and relationship storage from any specific graph backend
(NetworkX, Neo4j, RDF, SQLite).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from indiclearn.domain.concepts import Concept, ConceptRelation, RelationType


@runtime_checkable
class GraphStore(Protocol):
    """Protocol for academic knowledge graph operations."""

    def add_concept(self, concept: Concept) -> None:
        """Add or update a Concept node in the graph."""
        ...

    def get_concept(self, concept_id: str) -> Concept | None:
        """Retrieve a Concept node by its unique ID."""
        ...

    def has_concept(self, concept_id: str) -> bool:
        """Check if concept exists in graph."""
        ...

    def delete_concept(self, concept_id: str) -> bool:
        """Delete a concept and all connected incident relations."""
        ...

    def list_concepts(self, subject_id: str | None = None) -> list[Concept]:
        """List all concepts, optionally filtered by subject_id."""
        ...

    def add_relation(self, relation: ConceptRelation) -> None:
        """Add or update a directed ConceptRelation edge in the graph."""
        ...

    def get_relation(self, relation_id: str) -> ConceptRelation | None:
        """Retrieve a specific relation by ID."""
        ...

    def get_relations(
        self,
        subject_id: str | None = None,
        source_id: str | None = None,
        target_id: str | None = None,
        relation_type: RelationType | None = None,
    ) -> list[ConceptRelation]:
        """Query relations matching optional filter criteria."""
        ...

    def delete_relation(self, relation_id: str) -> bool:
        """Delete a specific relation edge."""
        ...

    def get_neighbors(
        self,
        concept_id: str,
        direction: Literal["in", "out", "both"] = "both",
    ) -> list[Concept]:
        """Get adjacent concept nodes."""
        ...

    def get_prerequisites(
        self,
        concept_id: str,
        transitive: bool = False,
    ) -> list[Concept]:
        """Get concepts that are prerequisites of concept_id (Source -> PREREQUISITE_OF -> concept_id)."""
        ...

    def get_dependents(
        self,
        concept_id: str,
        transitive: bool = False,
    ) -> list[Concept]:
        """Get concepts that depend on concept_id (concept_id -> PREREQUISITE_OF -> Target)."""
        ...

    def find_concept_by_alias(
        self,
        alias_query: str,
        subject_id: str | None = None,
    ) -> Concept | None:
        """Find a concept matching canonical name or alias (case-insensitive)."""
        ...

    def search_concepts(
        self,
        query: str,
        subject_id: str | None = None,
        limit: int = 5,
    ) -> list[Concept]:
        """Search concepts by substring match against name and aliases."""
        ...

    def save(self, path: Path | str) -> None:
        """Serialize knowledge graph to disk in a deterministic format."""
        ...

    def load(self, path: Path | str) -> None:
        """Load knowledge graph from disk."""
        ...

    def node_count(self, subject_id: str | None = None) -> int:
        """Count concept nodes, optionally filtered by subject."""
        ...

    def edge_count(self, subject_id: str | None = None) -> int:
        """Count relationship edges, optionally filtered by subject."""
        ...
