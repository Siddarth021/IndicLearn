"""Strongly typed domain models for RAG and Grounded Generation (Phase 5).

Preserves complete end-to-end provenance:
Query -> NLP Concept -> Retrieved Evidence -> Selected Context -> Citations -> Grounding Report.
Strictly decoupled from student knowledge / mastery state.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    """A single retrieved and selected piece of academic evidence with complete provenance."""

    source_id: str = Field(description="Deterministic label in prompt, e.g. 'S1', 'S2'")
    document_id: str
    chunk_id: str
    text: str
    page_number: int | None = None
    section: str | None = None
    score: float = Field(ge=0.0, description="Retrieval similarity score or fusion weight")
    retrieval_method: str = Field(default="hybrid", description="e.g. 'bm25', 'dense', 'hybrid'")
    concept_ids: list[str] = Field(default_factory=list)
    rank: int = Field(ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceSet(BaseModel):
    """Container holding selected evidence items and sufficiency status."""

    items: list[EvidenceItem] = Field(default_factory=list)
    query: str
    subject_id: str
    total_candidates: int = 0
    insufficient_evidence: bool = False
    insufficiency_reason: str | None = None

    def get_by_source_id(self, source_id: str) -> EvidenceItem | None:
        """Find evidence item by assigned prompt source identifier."""
        norm_id = source_id.upper().strip()
        for item in self.items:
            if item.source_id.upper().strip() == norm_id:
                return item
        return None

    def get_by_chunk_id(self, chunk_id: str) -> EvidenceItem | None:
        """Find evidence item by exact chunk ID."""
        for item in self.items:
            if item.chunk_id == chunk_id:
                return item
        return None


class ContextDocument(BaseModel):
    """Formatted representation of an evidence item prepared for LLM ingestion."""

    source_id: str
    header: str
    body: str
    full_text: str


class Citation(BaseModel):
    """Extracted and validated citation linking a claim to an academic source."""

    source_id: str
    document_id: str
    chunk_id: str
    page_number: int | None = None
    section: str | None = None
    text_snippet: str | None = None
    is_valid: bool = True
    validation_error: str | None = None


class CitationSpan(BaseModel):
    """Span in the generated answer text referencing one or more citations."""

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    raw_text: str
    cited_source_ids: list[str] = Field(default_factory=list)


class GroundingCheck(BaseModel):
    """Quantitative evaluation of factual citation validity and claim grounding."""

    is_grounded: bool
    citation_validity_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of citations that map to supplied evidence",
    )
    citation_coverage: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of content sentences that cite valid evidence",
    )
    unsupported_citation_count: int = Field(
        ge=0,
        description="Count of citations referencing unprovided or out-of-context sources",
    )
    unsupported_citation_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of citations that are invalid or unsupported",
    )
    evidence_sufficiency: bool
    invalid_citations: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    explanation: str = Field(default="")


class GroundingReport(BaseModel):
    """Summary report detailing citation correctness and grounding integrity."""

    check: GroundingCheck
    status: str = Field(
        description="'GROUNDED', 'PARTIALLY_GROUNDED', 'UNSUPPORTED', or 'INSUFFICIENT_EVIDENCE'",
    )
    citations_parsed: list[Citation] = Field(default_factory=list)
    valid_citations: list[Citation] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)


class GenerationMetadata(BaseModel):
    """Telemetry and operational metadata for a RAG completion."""

    model_name: str
    requested_language: str
    generated_language: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None
    graph_concepts_used: list[str] = Field(default_factory=list)
    retrieval_time_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGRequest(BaseModel):
    """High-level request to generate a grounded answer from academic course material."""

    query: str
    subject_id: str = "signals_and_systems"
    requested_language: str | None = None
    top_k: int = Field(default=5, ge=1)
    max_graph_hops: int = Field(default=1, ge=0)
    max_graph_concepts: int = Field(default=3, ge=0)
    min_score_threshold: float = Field(default=0.005, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGResponse(BaseModel):
    """Complete grounded answer response with full provenance and grounding audit."""

    query: str
    subject_id: str
    answer: str
    is_answerable: bool = True
    status: str = Field(default="COMPLETED")
    citations: list[Citation] = Field(default_factory=list)
    evidence: EvidenceSet
    concept_ids: list[str] = Field(default_factory=list)
    grounding_report: GroundingReport
    metadata: GenerationMetadata
