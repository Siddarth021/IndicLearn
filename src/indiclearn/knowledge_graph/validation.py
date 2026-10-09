"""Knowledge Graph validation routines (Phase 3).

Validates node integrity, edge referential integrity, prohibited self-loops,
relation types, provenance presence, and detects prerequisite cycles.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
from pydantic import BaseModel, Field

from indiclearn.domain.concepts import RelationType
from indiclearn.knowledge_graph.base import GraphStore


class ValidationReport(BaseModel):
    """Validation report detailing integrity and structure of a knowledge graph."""

    is_valid: bool = Field(description="True if no fatal integrity errors exist")
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    errors: list[str] = Field(default_factory=list, description="Fatal integrity violations")
    warnings: list[str] = Field(
        default_factory=list, description="Quality concerns (e.g. missing provenance)"
    )
    prerequisite_cycles: list[list[str]] = Field(
        default_factory=list, description="Cycles detected in the PREREQUISITE_OF subgraph"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


def validate_graph(
    store: GraphStore,
    subject_id: str | None = None,
    strict_provenance: bool = False,
) -> ValidationReport:
    """Validate graph referential integrity, ontology rules, and prerequisite DAG property."""
    errors: list[str] = []
    warnings: list[str] = []

    concepts = store.list_concepts(subject_id=subject_id)
    relations = store.get_relations(subject_id=subject_id)

    concept_ids = {c.concept_id for c in concepts}
    valid_relation_types = {rt.value for rt in RelationType}

    # 1. Concept checks
    seen_ids: set[str] = set()
    for c in concepts:
        if c.concept_id in seen_ids:
            errors.append(f"Duplicate concept ID detected: '{c.concept_id}'")
        seen_ids.add(c.concept_id)

        if not c.source_chunk_ids:
            msg = f"Concept '{c.concept_id}' ({c.canonical_name}) has no source chunk provenance."
            if strict_provenance:
                errors.append(msg)
            else:
                warnings.append(msg)

    # 2. Relation checks
    seen_edge_signatures: set[tuple[str, str, str]] = set()
    prereq_graph = nx.DiGraph()

    for rel in relations:
        # Check source and target exist
        if rel.source_concept_id not in concept_ids:
            errors.append(
                f"Relation '{rel.relation_id}' references non-existent source concept '{rel.source_concept_id}'"
            )
        if rel.target_concept_id not in concept_ids:
            errors.append(
                f"Relation '{rel.relation_id}' references non-existent target concept '{rel.target_concept_id}'"
            )

        # Prohibit self-loops
        if rel.source_concept_id == rel.target_concept_id:
            errors.append(
                f"Prohibited self-loop detected in relation '{rel.relation_id}' on concept '{rel.source_concept_id}'"
            )

        # Check relation type
        if rel.relation_type.value not in valid_relation_types:
            errors.append(
                f"Unknown relation type '{rel.relation_type}' in relation '{rel.relation_id}'"
            )

        # Check duplicate edges (same source, target, relation_type)
        sig = (rel.source_concept_id, rel.target_concept_id, rel.relation_type.value)
        if sig in seen_edge_signatures:
            warnings.append(
                f"Duplicate relationship signature between '{rel.source_concept_id}' and '{rel.target_concept_id}' with type '{rel.relation_type.value}'"
            )
        seen_edge_signatures.add(sig)

        # Check provenance
        if not rel.source_chunk_ids:
            msg = f"Relation '{rel.relation_id}' lacks source chunk provenance."
            if strict_provenance:
                errors.append(msg)
            else:
                warnings.append(msg)

        # Build prerequisite DAG
        if rel.relation_type == RelationType.PREREQUISITE_OF:
            prereq_graph.add_edge(rel.source_concept_id, rel.target_concept_id)

    # 3. Prerequisite cycle detection
    prereq_cycles: list[list[str]] = []
    if prereq_graph.number_of_nodes() > 0:
        try:
            raw_cycles = list(nx.simple_cycles(prereq_graph))
            for cycle in raw_cycles:
                prereq_cycles.append(cycle)
                warnings.append(
                    f"Prerequisite cycle detected: {' -> '.join(cycle)} -> {cycle[0]}. Prerequisite relationships should ideally form a DAG."
                )
        except nx.NetworkXError as e:
            warnings.append(f"Cycle detection encountered error: {e}")

    is_valid = len(errors) == 0

    return ValidationReport(
        is_valid=is_valid,
        node_count=len(concepts),
        edge_count=len(relations),
        errors=errors,
        warnings=warnings,
        prerequisite_cycles=prereq_cycles,
        metadata={
            "subject_id": subject_id,
            "prerequisite_edge_count": prereq_graph.number_of_edges(),
        },
    )
