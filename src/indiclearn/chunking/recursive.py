"""Recursive character chunker (BASELINE; chunk_size/overlap are PROPOSED INITIAL HEURISTICS).

Purpose: split page text into retrievable chunks that keep page/section metadata.
Input: pages (cleaned text). Output: DocumentChunk list with exact char offsets into page text.
Algorithm: recursively split text by separator hierarchy (paragraph, line, sentence, word,
hard cut) into units <= chunk_size whose concatenation equals the page text; greedily pack
contiguous units up to chunk_size; the next chunk starts by re-including trailing whole units
totalling <= chunk_overlap characters (always advancing by at least one unit).
Complexity: O(n) per page.
Assumptions: size measured in characters, not tokens (token-aware chunking is future work).
Limitations: overlap is unit-granular so actual overlap may be less than requested; chunks never
cross page boundaries; no semantic boundary detection.
Failure modes: pathological text without separators falls back to hard cuts (may split a word).
"""

from __future__ import annotations

import hashlib

from indiclearn.domain.documents import DocumentChunk, ExtractedPage

DEFAULT_SEPARATORS = ("\n\n", "\n", "। ", ". ", "? ", "! ", " ")


def _split_units(text: str, separators: tuple[str, ...], size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    if not separators:
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep, rest = separators[0], separators[1:]
    parts = text.split(sep)
    units: list[str] = []
    for idx, part in enumerate(parts):
        piece = part + (sep if idx < len(parts) - 1 else "")
        if not piece:
            continue
        units.extend([piece] if len(piece) <= size else _split_units(piece, rest, size))
    return units


def make_chunk_id(document_id: str, page_number: int, start: int, end: int) -> str:
    raw = f"{document_id}|{page_number}|{start}|{end}".encode()
    return "chk_" + hashlib.sha1(raw).hexdigest()[:16]


class RecursiveCharChunker:
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        separators: tuple[str, ...] = DEFAULT_SEPARATORS,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def chunk_pages(
        self, pages: list[ExtractedPage], document_id: str, subject_id: str
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for page in pages:
            chunks.extend(self._chunk_page(page, document_id, subject_id))
        return chunks

    def _chunk_page(
        self, page: ExtractedPage, document_id: str, subject_id: str
    ) -> list[DocumentChunk]:
        text = page.text
        if not text.strip():
            return []
        units = _split_units(text, self.separators, self.chunk_size)
        offsets = [0]
        for u in units:
            offsets.append(offsets[-1] + len(u))

        out: list[DocumentChunk] = []
        i, n = 0, len(units)
        while i < n:
            j, total = i, 0
            while j < n and total + len(units[j]) <= self.chunk_size:
                total += len(units[j])
                j += 1
            start, end = offsets[i], offsets[j]
            body = text[start:end]
            if body.strip():
                out.append(
                    DocumentChunk(
                        chunk_id=make_chunk_id(document_id, page.page_number, start, end),
                        document_id=document_id,
                        subject_id=subject_id,
                        page_number=page.page_number,
                        section=page.section,
                        text=body.strip(),
                        char_start=start,
                        char_end=end,
                    )
                )
            if j >= n:
                break
            k, overlap = j, 0
            while k > i + 1 and overlap + len(units[k - 1]) <= self.chunk_overlap:
                k -= 1
                overlap += len(units[k])
            i = k
        return out
