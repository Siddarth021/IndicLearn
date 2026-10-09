"""Citation parsing, normalization, and validation for RAG answers (Phase 5).

Ensures that every citation tag in the LLM completion (e.g. [S1], [S2]) strictly
resolves to an actual evidence chunk provided in the prompt context.
"""

from __future__ import annotations

import re

from indiclearn.rag.models import Citation, CitationSpan, EvidenceSet


class CitationParser:
    """Parses citation markers and offsets from generated answer text."""

    # Matches [S1], [s2], [S1, S2], [S1,S3]
    CITATION_PATTERN = re.compile(r"\[([Ss]\d+(?:\s*,\s*[Ss]\d+)*)\]")

    def parse_citation_spans(self, text: str) -> list[CitationSpan]:
        """Extract all citation spans with character offsets and referenced source IDs."""
        spans: list[CitationSpan] = []
        if not text:
            return spans

        for match in self.CITATION_PATTERN.finditer(text):
            raw_block = match.group(1)
            # Normalize to uppercase IDs (e.g. ['S1', 'S2'])
            raw_ids = [s.strip().upper() for s in raw_block.split(",") if s.strip()]
            start, end = match.span()

            spans.append(
                CitationSpan(
                    start=start,
                    end=end,
                    raw_text=match.group(0),
                    cited_source_ids=raw_ids,
                )
            )
        return spans

    def extract_unique_source_ids(self, text: str) -> list[str]:
        """Extract a deduplicated, ordered list of referenced source IDs."""
        spans = self.parse_citation_spans(text)
        seen: set[str] = set()
        ordered: list[str] = []
        for span in spans:
            for sid in span.cited_source_ids:
                if sid not in seen:
                    seen.add(sid)
                    ordered.append(sid)
        return ordered


class CitationValidator:
    """Validates cited source IDs against supplied EvidenceSet."""

    def __init__(self, parser: CitationParser | None = None) -> None:
        self.parser = parser or CitationParser()

    def validate_citations(
        self,
        answer_text: str,
        evidence_set: EvidenceSet,
    ) -> tuple[list[Citation], list[Citation], list[str]]:
        """Validate citations in answer text against provided evidence.

        Args:
            answer_text: LLM generated answer string.
            evidence_set: Academic evidence supplied in prompt context.

        Returns:
            Tuple of:
            - all_citations: list of parsed Citation objects with validation flags
            - valid_citations: subset of citations that correctly mapped to evidence
            - invalid_source_ids: list of source IDs that do not exist in evidence
        """
        source_ids = self.parser.extract_unique_source_ids(answer_text)
        all_citations: list[Citation] = []
        valid_citations: list[Citation] = []
        invalid_ids: list[str] = []

        for sid in source_ids:
            evidence_item = evidence_set.get_by_source_id(sid)

            if evidence_item is not None:
                snippet = (
                    evidence_item.text[:120] + "..."
                    if len(evidence_item.text) > 120
                    else evidence_item.text
                )
                citation = Citation(
                    source_id=evidence_item.source_id,
                    document_id=evidence_item.document_id,
                    chunk_id=evidence_item.chunk_id,
                    page_number=evidence_item.page_number,
                    section=evidence_item.section,
                    text_snippet=snippet,
                    is_valid=True,
                    validation_error=None,
                )
                all_citations.append(citation)
                valid_citations.append(citation)
            else:
                citation = Citation(
                    source_id=sid,
                    document_id="UNKNOWN",
                    chunk_id="UNKNOWN",
                    is_valid=False,
                    validation_error=f"Source ID '{sid}' was not supplied in prompt evidence.",
                )
                all_citations.append(citation)
                invalid_ids.append(sid)

        return all_citations, valid_citations, invalid_ids
