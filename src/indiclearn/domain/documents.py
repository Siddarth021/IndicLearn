"""Document-side domain entities (Phase 1).

Purpose: typed, storage-agnostic entities for ingested material. A ``Document`` is a source file,
a ``DocumentChunk`` is a retrievable slice of it. Neither holds student state or concept-graph
state; those are separate entities (see docs/DATA_MODEL.md).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    EXTRACTED = "extracted"
    FAILED = "failed"
    DUPLICATE = "duplicate"


class ExtractedPage(BaseModel):
    """One page/slide/section-block of extracted text. ``page_number`` is 1-based."""

    page_number: int = Field(ge=1)
    text: str
    section: str | None = None


class Document(BaseModel):
    document_id: str
    subject_id: str
    filename: str
    file_type: str
    sha256: str
    size_bytes: int
    page_count: int = 0
    status: DocumentStatus
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    subject_id: str
    page_number: int = Field(ge=1)
    section: str | None = None
    text: str
    char_start: int = Field(ge=0)  # offsets into the cleaned page text
    char_end: int = Field(ge=0)
    language: str | None = None  # filled by Phase 4
    concept_ids: list[str] = Field(default_factory=list)  # filled by Phase 3
