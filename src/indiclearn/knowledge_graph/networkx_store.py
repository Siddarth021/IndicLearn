"""NetworkX in-memory GraphStore adapter (Phase 3).

Implements GraphStore using networkx.MultiDiGraph with subject isolation
and deterministic JSON persistence.
"""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import networkx as nx

from indiclearn.domain.concepts import Concept, ConceptRelation, RelationType
from indiclearn.knowledge_graph.base import GraphStore


class NetworkXGraphStore(GraphStore):
    """Knowledge graph store implemented via networkx.MultiDiGraph."""

    def __init__(self, default_subject_id: str = "signals_and_systems") -> None:
        self.default_subject_id = default_subject_id
        self._graph = nx.MultiDiGraph()
        self._relations_by_id: dict[str, ConceptRelation] = {}

    @property
    def graph(self) -> nx.MultiDiGraph:
        """Direct access to the underlying networkx MultiDiGraph."""
        return self._graph

    def add_concept(self, concept: Concept) -> None:
        """Add or merge a concept node in the graph."""
        cid = concept.concept_id
        if self._graph.has_node(cid):
            # Merge provenance and aliases
            existing_data: dict[str, Any] = self._graph.nodes[cid]
            existing_cpt: Concept = existing_data["concept"]

            merged_aliases = sorted(set(existing_cpt.aliases + concept.aliases))
            merged_docs = sorted(set(existing_cpt.source_document_ids + concept.source_document_ids))
            merged_chunks = sorted(set(existing_cpt.source_chunk_ids + concept.source_chunk_ids))
            merged_pages = sorted(set(existing_cpt.page_numbers + concept.page_numbers))
            merged_sections = sorted(set(existing_cpt.sections + concept.sections))

            updated = existing_cpt.model_copy(
                update={
                    "aliases": merged_aliases,
                    "source_document_ids": merged_docs,
                    "source_chunk_ids": merged_chunks,
                    "page_numbers": merged_pages,
                    "sections": merged_sections,
                    "description": concept.description or existing_cpt.description,
                    "confidence": max(existing_cpt.confidence, concept.confidence),
                }
            )
            self._graph.nodes[cid]["concept"] = updated
        else:
            self._graph.add_node(cid, concept=concept, subject_id=concept.subject_id)

    def get_concept(self, concept_id: str) -> Concept | None:
        """Get concept by ID."""
        if not self._graph.has_node(concept_id):
            return None
        return self._graph.nodes[concept_id].get("concept")

    def has_concept(self, concept_id: str) -> bool:
        return self._graph.has_node(concept_id) and "concept" in self._graph.nodes[concept_id]

    def delete_concept(self, concept_id: str) -> bool:
        """Delete concept and its connected relations."""
        if not self._graph.has_node(concept_id):
            return False

        # Remove from relation map
        for u, v, k in list(self._graph.edges(concept_id, keys=True)):
            rel_id = self._graph.edges[u, v, k].get("relation_id")
            if rel_id:
                self._relations_by_id.pop(rel_id, None)

        for u, v, k in list(self._graph.in_edges(concept_id, keys=True)):
            rel_id = self._graph.edges[u, v, k].get("relation_id")
            if rel_id:
                self._relations_by_id.pop(rel_id, None)

        self._graph.remove_node(concept_id)
        return True

    def list_concepts(self, subject_id: str | None = None) -> list[Concept]:
        """List all concepts, optionally filtered by subject."""
        concepts: list[Concept] = []
        for _, data in self._graph.nodes(data=True):
            cpt: Concept | None = data.get("concept")
            if cpt is None:
                continue
            if subject_id is None or cpt.subject_id == subject_id:
                concepts.append(cpt)
        return sorted(concepts, key=lambda c: c.concept_id)

    def add_relation(self, relation: ConceptRelation) -> None:
        """Add a directed relation edge."""
        u = relation.source_concept_id
        v = relation.target_concept_id
        rid = relation.relation_id

        # Ensure node presence
        if not self._graph.has_node(u) or not self._graph.has_node(v):
            raise KeyError(
                f"Cannot add relation {rid}: source '{u}' or target '{v}' node does not exist in graph."
            )

        if rid in self._relations_by_id:
            existing = self._relations_by_id[rid]
            merged_chunks = sorted(set(existing.source_chunk_ids + relation.source_chunk_ids))
            merged_docs = sorted(set(existing.source_document_ids + relation.source_document_ids))
            updated = existing.model_copy(
                update={
                    "confidence": max(existing.confidence, relation.confidence),
                    "source_chunk_ids": merged_chunks,
                    "source_document_ids": merged_docs,
                }
            )
            self._relations_by_id[rid] = updated
            if self._graph.has_edge(u, v, key=rid):
                self._graph[u][v][rid]["confidence"] = updated.confidence
                self._graph[u][v][rid]["relation"] = updated
            return

        self._relations_by_id[rid] = relation
        self._graph.add_edge(
            u,
            v,
            key=rid,
            relation_id=rid,
            relation_type=relation.relation_type.value,
            subject_id=relation.subject_id,
            confidence=relation.confidence,
            relation=relation,
        )

    def get_relation(self, relation_id: str) -> ConceptRelation | None:
        return self._relations_by_id.get(relation_id)

    def get_relations(
        self,
        subject_id: str | None = None,
        source_id: str | None = None,
        target_id: str | None = None,
        relation_type: RelationType | None = None,
        concept_id: str | None = None,
    ) -> list[ConceptRelation]:
        """Query relations matching filters."""
        # If subject_id starts with "cpt_", treat it as concept_id passed positionally
        if subject_id and subject_id.startswith("cpt_") and concept_id is None:
            concept_id = subject_id
            subject_id = None

        results: list[ConceptRelation] = []
        for _u, _v, _k, edge_data in self._graph.edges(keys=True, data=True):
            rel: ConceptRelation | None = edge_data.get("relation") or edge_data.get("data")
            if rel is None:
                continue
            if subject_id is not None and rel.subject_id != subject_id:
                continue
            if concept_id is not None and rel.source_concept_id != concept_id and rel.target_concept_id != concept_id:
                continue
            if source_id is not None and rel.source_concept_id != source_id:
                continue
            if target_id is not None and rel.target_concept_id != target_id:
                continue
            if relation_type is not None and rel.relation_type != relation_type:
                continue
            results.append(rel)
        return sorted(results, key=lambda r: r.relation_id)

    def delete_relation(self, relation_id: str) -> bool:
        """Delete relation edge by ID."""
        rel = self._relations_by_id.pop(relation_id, None)
        if rel is None:
            return False
        if self._graph.has_edge(rel.source_concept_id, rel.target_concept_id, key=relation_id):
            self._graph.remove_edge(
                rel.source_concept_id, rel.target_concept_id, key=relation_id
            )
            return True
        return False

    def get_neighbors(
        self,
        concept_id: str,
        direction: Literal["in", "out", "both"] = "both",
    ) -> list[Concept]:
        """Get adjacent concept nodes."""
        if not self._graph.has_node(concept_id):
            return []

        neighbor_ids: set[str] = set()
        if direction in ("out", "both"):
            neighbor_ids.update(self._graph.successors(concept_id))
        if direction in ("in", "both"):
            neighbor_ids.update(self._graph.predecessors(concept_id))

        return [
            self._graph.nodes[nid]["concept"]
            for nid in sorted(neighbor_ids)
            if self._graph.has_node(nid)
        ]

    def get_prerequisites(
        self,
        concept_id: str,
        transitive: bool = False,
    ) -> list[Concept]:
        """Get concepts that are prerequisites of concept_id.

        Edge direction: Prerequisite -> PREREQUISITE_OF -> TargetConcept.
        Prerequisites of concept_id are incoming predecessor nodes with relation_type == PREREQUISITE_OF.
        """
        if not self._graph.has_node(concept_id):
            return []

        if not transitive:
            prereqs: set[str] = set()
            for pred in self._graph.predecessors(concept_id):
                for k in self._graph[pred][concept_id]:
                    if (
                        self._graph[pred][concept_id][k].get("relation_type")
                        == RelationType.PREREQUISITE_OF.value
                    ):
                        prereqs.add(pred)
            return [self._graph.nodes[pid]["concept"] for pid in sorted(prereqs)]

        # Transitive BFS traversal backwards along PREREQUISITE_OF edges
        visited: set[str] = set()
        queue: deque[str] = deque([concept_id])

        while queue:
            curr = queue.popleft()
            for pred in self._graph.predecessors(curr):
                has_prereq_edge = any(
                    self._graph[pred][curr][k].get("relation_type")
                    == RelationType.PREREQUISITE_OF.value
                    for k in self._graph[pred][curr]
                )
                if has_prereq_edge and pred not in visited and pred != concept_id:
                    visited.add(pred)
                    queue.append(pred)

        return [self._graph.nodes[pid]["concept"] for pid in sorted(visited)]

    def get_dependents(
        self,
        concept_id: str,
        transitive: bool = False,
    ) -> list[Concept]:
        """Get concepts that depend on concept_id.

        Edge direction: Concept -> PREREQUISITE_OF -> DependentConcept.
        Dependents are outgoing successor nodes with relation_type == PREREQUISITE_OF.
        """
        if not self._graph.has_node(concept_id):
            return []

        if not transitive:
            dependents: set[str] = set()
            for succ in self._graph.successors(concept_id):
                for k in self._graph[concept_id][succ]:
                    if (
                        self._graph[concept_id][succ][k].get("relation_type")
                        == RelationType.PREREQUISITE_OF.value
                    ):
                        dependents.add(succ)
            return [self._graph.nodes[did]["concept"] for did in sorted(dependents)]

        visited: set[str] = set()
        queue: deque[str] = deque([concept_id])

        while queue:
            curr = queue.popleft()
            for succ in self._graph.successors(curr):
                has_prereq_edge = any(
                    self._graph[curr][succ][k].get("relation_type")
                    == RelationType.PREREQUISITE_OF.value
                    for k in self._graph[curr][succ]
                )
                if has_prereq_edge and succ not in visited and succ != concept_id:
                    visited.add(succ)
                    queue.append(succ)

        return [self._graph.nodes[did]["concept"] for did in sorted(visited)]

    def find_concept_by_alias(
        self,
        alias_query: str,
        subject_id: str | None = None,
    ) -> Concept | None:
        """Lookup concept by canonical name or alias match."""
        clean_q = alias_query.strip().lower()
        if not clean_q:
            return None

        for _, data in self._graph.nodes(data=True):
            cpt: Concept = data["concept"]
            if subject_id is not None and cpt.subject_id != subject_id:
                continue
            if cpt.has_alias(clean_q):
                return cpt
        return None

    def search_concepts(
        self,
        query: str,
        subject_id: str | None = None,
        limit: int = 5,
    ) -> list[Concept]:
        """Search concepts by substring match."""
        clean_q = query.strip().lower()
        if not clean_q:
            return []

        scored: list[tuple[int, Concept]] = []
        for _, data in self._graph.nodes(data=True):
            cpt: Concept = data["concept"]
            if subject_id is not None and cpt.subject_id != subject_id:
                continue

            score = 0
            if clean_q == cpt.canonical_name.lower():
                score = 3
            elif clean_q in cpt.canonical_name.lower():
                score = 2
            elif any(clean_q in a.lower() for a in cpt.aliases):
                score = 1

            if score > 0:
                scored.append((score, cpt))

        # Sort descending by score, ascending by concept_id
        scored.sort(key=lambda item: (-item[0], item[1].concept_id))
        return [c for _, c in scored[:limit]]

    def save(self, path: Path | str) -> None:
        """Serialize graph to disk in deterministic JSON format."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        nodes_data = [
            data["concept"].model_dump(mode="json")
            for _, data in sorted(self._graph.nodes(data=True), key=lambda x: x[0])
        ]
        edges_data = [
            rel.model_dump(mode="json")
            for _, rel in sorted(self._relations_by_id.items(), key=lambda x: x[0])
        ]

        payload = {
            "format_version": 1,
            "subject_id": self.default_subject_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "node_count": len(nodes_data),
            "edge_count": len(edges_data),
            "nodes": nodes_data,
            "edges": edges_data,
        }

        save_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self, path: Path | str) -> None:
        """Load graph from JSON file."""
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Knowledge graph file not found: {load_path}")

        raw = json.loads(load_path.read_text(encoding="utf-8"))
        self._graph.clear()
        self._relations_by_id.clear()
        self.default_subject_id = raw.get("subject_id", self.default_subject_id)

        for n_dict in raw.get("nodes", []):
            cpt = Concept(**n_dict)
            self.add_concept(cpt)

        for e_dict in raw.get("edges", []):
            rel = ConceptRelation(**e_dict)
            self.add_relation(rel)

    def node_count(self, subject_id: str | None = None) -> int:
        if subject_id is None:
            return self._graph.number_of_nodes()
        return sum(1 for _, d in self._graph.nodes(data=True) if d["concept"].subject_id == subject_id)

    def edge_count(self, subject_id: str | None = None) -> int:
        if subject_id is None:
            return len(self._relations_by_id)
        return sum(1 for rel in self._relations_by_id.values() if rel.subject_id == subject_id)

    def concept_count(self, subject_id: str | None = None) -> int:
        """Alias for node_count."""
        return self.node_count(subject_id)

    def relation_count(self, subject_id: str | None = None) -> int:
        """Alias for edge_count."""
        return self.edge_count(subject_id)

    def find_concept(self, query: str, subject_id: str | None = None) -> Concept | None:
        """Convenience alias for find_concept_by_alias."""
        return self.find_concept_by_alias(query, subject_id)

    def get_all_concepts(self, subject_id: str | None = None) -> list[Concept]:
        """Convenience alias for list_concepts."""
        return self.list_concepts(subject_id)

    def find_path(self, source_concept_id: str, target_concept_id: str) -> list[str] | None:
        """Find the shortest directed path between two concepts."""
        if not self._graph.has_node(source_concept_id) or not self._graph.has_node(target_concept_id):
            return None
        try:
            return list(nx.shortest_path(self._graph, source_concept_id, target_concept_id))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    save_to_json = save
    load_from_json = load
