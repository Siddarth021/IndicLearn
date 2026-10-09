"""BM25 (BM25Okapi) lexical retrieval baseline (Phase 2).

Provides a self-contained, dependency-light lexical baseline. Supports Unicode
tokenization for English, Telugu script, and Romanized code-mixed text.
Satisfies the Retriever protocol.
"""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from typing import Any

from indiclearn.domain.documents import DocumentChunk
from indiclearn.retrieval.base import RetrievalResult, Retriever


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase Unicode tokens, preserving Indic, Latin, and code-mixed words."""
    if not text:
        return []
    tokens: list[str] = []
    current: list[str] = []
    for char in text:
        cat = unicodedata.category(char)
        # Match letters (L), marks/vowels/viramas (M), numbers (N), connectors, ZWJ/ZWNJ
        if cat[0] in ("L", "N", "M") or cat == "Pc" or char in ("\u200c", "\u200d"):
            current.append(char)
        else:
            if current:
                tokens.append("".join(current).lower())
                current = []
    if current:
        tokens.append("".join(current).lower())
    return tokens


class BM25Retriever(Retriever):
    """BM25Okapi retriever implementation for DocumentChunk collections."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.chunks: list[DocumentChunk] = []
        self.doc_lengths: list[int] = []
        self.avg_doc_length: float = 0.0
        self.doc_count: int = 0
        self.doc_term_frequencies: list[Counter[str]] = []
        self.idf: dict[str, float] = {}

    def index(self, chunks: list[DocumentChunk]) -> None:
        """Index a list of DocumentChunks."""
        self.chunks = list(chunks)
        self.doc_count = len(chunks)
        if self.doc_count == 0:
            self.doc_lengths = []
            self.avg_doc_length = 0.0
            self.doc_term_frequencies = []
            self.idf = {}
            return

        self.doc_term_frequencies = []
        self.doc_lengths = []
        df: Counter[str] = Counter()

        for chunk in chunks:
            tokens = tokenize(chunk.text)
            self.doc_lengths.append(len(tokens))
            term_freq = Counter(tokens)
            self.doc_term_frequencies.append(term_freq)
            for term in term_freq:
                df[term] += 1

        self.avg_doc_length = sum(self.doc_lengths) / self.doc_count if self.doc_count > 0 else 0.0

        # Calculate Okapi IDF with floor for rare/frequent terms
        idf_values: dict[str, float] = {}
        for term, freq in df.items():
            # Standard BM25 IDF: ln((N - n + 0.5) / (n + 0.5) + 1)
            val = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1.0)
            idf_values[term] = val

        # Handle negative IDFs by setting a small positive epsilon * average positive IDF
        positive_idfs = [v for v in idf_values.values() if v > 0]
        avg_pos_idf = (sum(positive_idfs) / len(positive_idfs)) if positive_idfs else 0.1
        eps_floor = self.epsilon * avg_pos_idf

        for term, val in idf_values.items():
            if val <= 0:
                idf_values[term] = eps_floor

        self.idf = idf_values

    def _matches_filters(self, chunk: DocumentChunk, filters: dict[str, Any] | None) -> bool:
        """Check if chunk satisfies metadata filters."""
        if not filters:
            return True
        for key, expected in filters.items():
            if key == "document_id" and chunk.document_id != expected:
                return False
            if key == "subject_id" and chunk.subject_id != expected:
                return False
            if key == "language" and chunk.language != expected:
                return False
            if key == "concept_id":
                if isinstance(expected, list):
                    if not any(cid in chunk.concept_ids for cid in expected):
                        return False
                elif expected not in chunk.concept_ids:
                    return False
        return True

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve top_k chunks matching query via BM25 scoring."""
        if self.doc_count == 0 or not query.strip():
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores: list[tuple[float, int, DocumentChunk]] = []

        for idx, chunk in enumerate(self.chunks):
            if not self._matches_filters(chunk, filters):
                continue

            doc_len = self.doc_lengths[idx]
            tf_map = self.doc_term_frequencies[idx]
            score = 0.0

            for q_term in query_tokens:
                if q_term not in self.idf:
                    continue
                tf = tf_map.get(q_term, 0)
                if tf == 0:
                    continue

                idf_val = self.idf[q_term]
                # BM25 term saturation formula
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_length))
                term_score = idf_val * (tf * (self.k1 + 1.0)) / (denom if denom > 0 else 1.0)
                score += term_score

            if score > 0:
                scores.append((score, idx, chunk))

        # Sort by score descending, then by chunk_id ascending for deterministic tie-breaking
        scores.sort(key=lambda item: (-item[0], item[2].chunk_id))

        results: list[RetrievalResult] = []
        for rank, (score, _, chunk) in enumerate(scores[:top_k], start=1):
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    score=float(score),
                    rank=rank,
                    text=chunk.text,
                    metadata={
                        "subject_id": chunk.subject_id,
                        "page_number": chunk.page_number,
                        "section": chunk.section,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                        "language": chunk.language,
                        "concept_ids": chunk.concept_ids,
                    },
                )
            )
        return results
