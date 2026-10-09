"""Context Builder for assembling deterministic, provenance-preserving prompts (Phase 5).

Formats EvidenceSets and optional academic Knowledge Graph context into structured,
source-delimited prompt blocks for LLM consumption.
"""

from __future__ import annotations

from indiclearn.domain.concepts import Concept
from indiclearn.rag.models import ContextDocument, EvidenceItem, EvidenceSet


class ContextBuilder:
    """Builds deterministic, structured context blocks from selected academic evidence."""

    def __init__(self, include_metadata_header: bool = True) -> None:
        self.include_metadata_header = include_metadata_header

    def format_evidence_item(self, item_index: int, item: EvidenceItem) -> ContextDocument:
        """Format an individual evidence item into a ContextDocument block."""
        source_id = item.source_id or f"S{item_index}"
        doc_name = item.document_id
        page_str = f"Page {item.page_number}" if item.page_number else "Page N/A"
        section_str = f"Section: {item.section}" if item.section else "Section: General"

        header_lines = [
            f"[SOURCE {item_index}] (ID: {source_id})",
            f"Document: {doc_name} | {page_str} | {section_str} | Chunk: {item.chunk_id}",
        ]
        if item.concept_ids:
            header_lines.append(f"Related Concepts: {', '.join(item.concept_ids)}")

        header = "\n".join(header_lines)
        body = item.text.strip()
        full_text = f"{header}\n{body}\n"

        return ContextDocument(
            source_id=source_id,
            header=header,
            body=body,
            full_text=full_text,
        )

    def build_context(
        self,
        evidence_set: EvidenceSet,
        graph_concepts: list[Concept] | None = None,
        prerequisite_map: dict[str, list[str]] | None = None,
    ) -> str:
        """Build full deterministic context string for inclusion in LLM prompt.

        Args:
            evidence_set: Selected academic evidence items.
            graph_concepts: Optional list of related canonical concepts from GraphStore.
            prerequisite_map: Optional mapping of concept_id -> prerequisite concept names.

        Returns:
            Formatted context string.
        """
        if not evidence_set.items:
            return "No academic sources retrieved."

        blocks: list[str] = []

        # 1. Format academic knowledge graph curriculum context if provided
        if graph_concepts:
            graph_lines = [
                "=== ACADEMIC CURRICULUM CONTEXT ===",
                f"Subject: {evidence_set.subject_id}",
            ]
            for cpt in graph_concepts:
                cpt_type_str = cpt.concept_type.value if hasattr(cpt.concept_type, "value") else str(cpt.concept_type)
                line = f"- Concept: {cpt.canonical_name} ({cpt_type_str})"
                if cpt.aliases:
                    line += f" [Aliases: {', '.join(cpt.aliases[:3])}]"
                if prerequisite_map and cpt.concept_id in prerequisite_map:
                    prereqs = prerequisite_map[cpt.concept_id]
                    if prereqs:
                        line += f" | Prerequisites: {', '.join(prereqs)}"
                graph_lines.append(line)
            graph_lines.append("===================================")
            blocks.append("\n".join(graph_lines))

        # 2. Format individual source evidence blocks
        blocks.append("=== RETRIEVED ACADEMIC EVIDENCE ===")
        for idx, item in enumerate(evidence_set.items, start=1):
            doc = self.format_evidence_item(idx, item)
            blocks.append(doc.full_text)
        blocks.append("===================================")

        return "\n\n".join(blocks)
