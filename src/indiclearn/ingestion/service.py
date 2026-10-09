"""Ingestion service: file -> Document + cleaned pages (+ chunks).

Purpose: orchestrate loader selection, size/extension checks, hashing, duplicate detection,
cleaning, failure recording and chunking. Pure/in-memory for now; persistence is a Phase 2+
repository concern (seen hashes are injected so a DB-backed set can replace the default).
Failure policy: never raises for bad input. Returns ``status=FAILED`` with an error message
identifying the file (unsupported type, too large, corrupt, no extractable text).
Duplicate policy: same sha256 within the same subject -> ``status=DUPLICATE``, no re-extraction.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from indiclearn.chunking.base import Chunker
from indiclearn.config.settings import Settings
from indiclearn.domain.documents import Document, DocumentChunk, DocumentStatus, ExtractedPage
from indiclearn.ingestion.base import DocumentLoader, ExtractionError
from indiclearn.ingestion.loaders import PdfLoader, PptxLoader, TextLoader
from indiclearn.preprocessing.cleaning import clean_text
from indiclearn.utils.logging import get_logger, log_event

logger = get_logger("ingestion")


@dataclass
class IngestionResult:
    document: Document
    pages: list[ExtractedPage] = field(default_factory=list)
    chunks: list[DocumentChunk] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.document.status == DocumentStatus.EXTRACTED


def default_loaders() -> list[DocumentLoader]:
    return [TextLoader(), PdfLoader(), PptxLoader()]


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        chunker: Chunker,
        loaders: list[DocumentLoader] | None = None,
        seen: set[tuple[str, str]] | None = None,
    ) -> None:
        self.settings = settings
        self.chunker = chunker
        self._loaders: dict[str, DocumentLoader] = {}
        for loader in loaders or default_loaders():
            for ext in loader.extensions:
                if ext in settings.ingestion.supported_extensions:
                    self._loaders[ext] = loader
        self._seen = seen if seen is not None else set()  # {(subject_id, sha256)}

    def ingest(self, path: str | Path, subject_id: str) -> IngestionResult:
        p = Path(path)
        ext = p.suffix.lower()
        base = {"subject_id": subject_id, "filename": p.name, "file_type": ext.lstrip(".")}

        def fail(msg: str, sha: str = "", size: int = 0) -> IngestionResult:
            doc = Document(
                document_id=f"doc_failed_{hashlib.sha1(p.name.encode()).hexdigest()[:10]}",
                sha256=sha, size_bytes=size, status=DocumentStatus.FAILED, error=msg, **base,
            )
            log_event(logger, "document_ingest_failed", subject_id=subject_id, filename=p.name, error=msg)
            return IngestionResult(document=doc)

        if ext not in self._loaders:
            return fail(f"unsupported file type '{ext}'")
        try:
            data = p.read_bytes()
        except OSError as exc:
            return fail(f"cannot read file: {type(exc).__name__}")
        size = len(data)
        sha = hashlib.sha256(data).hexdigest()
        if size > self.settings.ingestion.max_file_mb * 1024 * 1024:
            return fail(f"file exceeds {self.settings.ingestion.max_file_mb} MB limit", sha, size)
        if (subject_id, sha) in self._seen:
            doc = Document(document_id=f"doc_{sha[:16]}", sha256=sha, size_bytes=size,
                           status=DocumentStatus.DUPLICATE, **base)
            log_event(logger, "document_duplicate", subject_id=subject_id, document_id=doc.document_id)
            return IngestionResult(document=doc)

        try:
            raw_pages = self._loaders[ext].load(p)
        except ExtractionError as exc:
            return fail(str(exc), sha, size)

        pp = self.settings.preprocessing
        pages = [
            ExtractedPage(page_number=pg.page_number, section=pg.section,
                          text=clean_text(pg.text, pp.unicode_form, pp.join_hyphenated_lines))
            for pg in raw_pages
        ]
        if not any(pg.text for pg in pages):
            return fail("no extractable text (scanned/image-only or empty document?)", sha, size)

        doc = Document(document_id=f"doc_{sha[:16]}", sha256=sha, size_bytes=size,
                       page_count=len(pages), status=DocumentStatus.EXTRACTED, **base)
        chunks = self.chunker.chunk_pages(pages, doc.document_id, subject_id)
        self._seen.add((subject_id, sha))
        log_event(logger, "document_ingested", subject_id=subject_id, document_id=doc.document_id,
                  pages=len(pages), chunks=len(chunks))
        return IngestionResult(document=doc, pages=pages, chunks=chunks)
