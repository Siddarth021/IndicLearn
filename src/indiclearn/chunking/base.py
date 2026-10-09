"""Chunker interface."""

from __future__ import annotations

from typing import Protocol

from indiclearn.domain.documents import DocumentChunk, ExtractedPage


class Chunker(Protocol):
    def chunk_pages(
        self, pages: list[ExtractedPage], document_id: str, subject_id: str
    ) -> list[DocumentChunk]: ...
