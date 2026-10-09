"""Academic technical term detector grounded directly in the Academic Knowledge Graph (Phase 4).

Uses the canonical concept graph as the authoritative source of truth.
Extracts canonical terms, English aliases, Telugu script aliases, and Romanized variations.
"""

from __future__ import annotations

import re
import unicodedata

from indiclearn.concepts.normalizer import ConceptNormalizer
from indiclearn.domain.concepts import Concept
from indiclearn.knowledge_graph.base import GraphStore
from indiclearn.nlp.models import TechnicalTerm
from indiclearn.nlp.terminology.base import TechnicalTermDetector


class GraphGroundedTermDetector(TechnicalTermDetector):
    """Detects engineering terminology using canonical concepts and aliases from GraphStore."""

    def __init__(
        self,
        graph_store: GraphStore | None = None,
        default_subject_id: str = "signals_and_systems",
        concepts: list[Concept] | None = None,
    ) -> None:
        self.graph_store = graph_store
        self.default_subject_id = default_subject_id
        self.normalizer = ConceptNormalizer(subject_id=default_subject_id)
        self._preloaded_concepts = concepts or []

    @classmethod
    def from_graph_store(
        cls,
        graph_store: GraphStore,
        default_subject_id: str = "signals_and_systems",
    ) -> GraphGroundedTermDetector:
        """Instantiate term detector directly grounded in a GraphStore."""
        return cls(graph_store=graph_store, default_subject_id=default_subject_id)

    def _get_concepts(self, subject_id: str) -> list[Concept]:
        if self._preloaded_concepts:
            return [c for c in self._preloaded_concepts if c.subject_id == subject_id]
        if self.graph_store is not None:
            return self.graph_store.list_concepts(subject_id=subject_id)
        return []

    def detect_terms(
        self,
        text: str,
        subject_id: str = "signals_and_systems",
    ) -> list[TechnicalTerm]:
        """Identify technical terms in query string, linking each to its canonical concept."""
        norm_text = unicodedata.normalize("NFC", text or "")
        if not norm_text.strip():
            return []

        concepts = self._get_concepts(subject_id or self.default_subject_id)

        # Build vocabulary entry list: (pattern_str, canonical_name, concept_id, term_type)
        entries: list[tuple[str, str, str, str]] = []

        for cpt in concepts:
            cid = cpt.concept_id
            cname = cpt.canonical_name

            # Canonical name
            entries.append((cname, cname, cid, "canonical"))

            # Aliases
            for alias in cpt.aliases:
                clean_alias = alias.strip()
                if not clean_alias:
                    continue
                has_te = any(0x0C00 <= ord(c) <= 0x0C7F for c in clean_alias)
                is_acronym = clean_alias.isupper() and len(clean_alias) <= 6
                term_type = "telugu_alias" if has_te else ("acronym" if is_acronym else "alias")
                entries.append((clean_alias, cname, cid, term_type))

        # Add known acronyms from normalizer
        for acr, canon in self.normalizer._acronym_map.items():
            matched_cpt = next((c for c in concepts if c.canonical_name.lower() == canon.lower()), None)
            cid = matched_cpt.concept_id if matched_cpt else None
            entries.append((acr, canon, cid or "", "acronym"))

        # Sort entries descending by length to prioritize longest matching phrase
        # e.g. "Continuous-Time Fourier Transform" before "Fourier Transform", "LTI System" before "System"
        entries.sort(key=lambda e: len(e[0]), reverse=True)

        # Track matched character spans to avoid overlapping sub-matches
        occupied_spans: list[tuple[int, int]] = []
        found_terms: list[TechnicalTerm] = []

        def is_overlapping(start: int, end: int) -> bool:
            return any(s < end and start < e for s, e in occupied_spans)

        for term_str, canon_name, cpt_id, term_type in entries:
            clean_term = term_str.strip()
            if len(clean_term) < 2:
                continue

            escaped = re.escape(clean_term)
            has_te = any(0x0C00 <= ord(c) <= 0x0C7F for c in clean_term)

            # For Latin words, require word boundaries; for Indic script or punctuation-laden terms, relaxed boundaries
            if has_te or not clean_term[0].isalnum() or not clean_term[-1].isalnum():
                pattern = re.compile(rf"(?i){escaped}")
            else:
                pattern = re.compile(rf"(?i)\b{escaped}\b")

            for m in pattern.finditer(norm_text):
                start, end = m.span()
                if is_overlapping(start, end):
                    continue

                matched_str = norm_text[start:end]
                term_obj = TechnicalTerm(
                    start=start,
                    end=end,
                    matched_text=matched_str,
                    canonical_name=canon_name,
                    concept_id=cpt_id if cpt_id else None,
                    term_type=term_type,
                    confidence=0.95 if term_type in ("canonical", "acronym") else 0.85,
                )
                found_terms.append(term_obj)
                occupied_spans.append((start, end))

        # Sort extracted terms chronologically by text occurrence
        found_terms.sort(key=lambda t: t.start)
        return found_terms
