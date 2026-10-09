"""Claim-level extraction, entailment protocol, and citation support assessment (Phase 5.1).

Implements Section 12 (Claim-Level Analysis):
Generated Answer
       ↓
Claim Extraction (Sentence / Propositional baseline)
       ↓
Claim → Citation mapping
       ↓
Evidence resolution
       ↓
Support Assessment (EntailmentVerifier protocol with heuristic lexical baseline)

Scientific Note:
The HeuristicLexicalOverlapAssessor provides a term-overlap baseline.
It does NOT constitute an NLI / natural language inference model. True entailment
requires a neural cross-encoder conforming to the EntailmentVerifier protocol.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import ClassVar, Protocol, runtime_checkable

from indiclearn.rag.citation import CitationParser
from indiclearn.rag.models import EvidenceItem, EvidenceSet


@dataclass
class Claim:
    """A granular propositional claim extracted from generated answer text."""

    claim_id: str
    text: str
    cited_source_ids: list[str] = field(default_factory=list)
    is_cited: bool = False
    support_status: str = "PENDING"  # 'SUPPORTED', 'UNSUPPORTED_CITATION', 'UNRELATED_EVIDENCE', 'UNCITED'
    support_score: float = 0.0
    contradictions: list[str] = field(default_factory=list)
    explanation: str = ""


@runtime_checkable
class ClaimExtractor(Protocol):
    """Protocol for extracting granular factual claims from generated text."""

    def extract_claims(self, text: str) -> list[Claim]:
        """Extract a list of propositional claims from text."""
        ...


class SentenceClaimExtractor:
    """Heuristic sentence-level claim extractor.

    Splits answer text into substantive propositional sentences while parsing
    inline citation references.
    """

    def __init__(self, parser: CitationParser | None = None) -> None:
        self.parser = parser or CitationParser()

    def _split_sentences(self, text: str) -> list[str]:
        raw_sentences = re.split(r"(?<=[.!?\n])\s+", text)
        return [s.strip() for s in raw_sentences if len(s.strip()) > 10]

    def extract_claims(self, text: str) -> list[Claim]:
        sentences = self._split_sentences(text)
        claims: list[Claim] = []

        for idx, sent in enumerate(sentences, start=1):
            cited_ids = self.parser.extract_unique_source_ids(sent)
            # Remove inline citation brackets to obtain clean propositional text
            clean_text = self.parser.CITATION_PATTERN.sub("", sent).strip()
            # Also clean up trailing punctuation artifacts if needed
            clean_text = re.sub(r"\s+", " ", clean_text)

            claims.append(
                Claim(
                    claim_id=f"C{idx}",
                    text=clean_text or sent,
                    cited_source_ids=cited_ids,
                    is_cited=len(cited_ids) > 0,
                    support_status="PENDING",
                )
            )

        return claims


@runtime_checkable
class EntailmentVerifier(Protocol):
    """Protocol for evaluating whether an evidence snippet entails or supports a claim."""

    def verify_support(
        self, claim_text: str, evidence_text: str
    ) -> tuple[bool, float, str]:
        """Verify whether evidence supports the claim.

        Returns:
            Tuple of (is_supported, confidence_or_overlap_score, explanation)
        """
        ...


class HeuristicLexicalOverlapAssessor:
    """Term-overlap baseline for assessing support between a claim and cited evidence.

    CAUTION: This is NOT a neural NLI model. It assesses lexical overlap
    of substantive content words (stemmed/lowercased non-stopwords).
    """

    STOPWORDS: ClassVar[set[str]] = {
        "a", "an", "the", "and", "or", "but", "if", "is", "are", "was", "were",
        "be", "been", "being", "in", "on", "at", "to", "for", "with", "by",
        "about", "against", "between", "into", "through", "during", "before",
        "after", "above", "below", "from", "up", "down", "of", "off", "over",
        "under", "again", "further", "then", "once", "here", "there", "when",
        "where", "why", "how", "all", "any", "both", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
        "don", "should", "now", "it", "its", "this", "that", "these", "those",
    }

    def __init__(self, min_token_overlap_threshold: float = 0.15) -> None:
        self.min_token_overlap_threshold = min_token_overlap_threshold

    def _tokenize_content_words(self, text: str) -> set[str]:
        words = re.findall(r"\b[a-zA-Z0-9_\u0C00-\u0C7F]{3,}\b", text.lower())
        return {w for w in words if w not in self.STOPWORDS}

    def verify_support(
        self, claim_text: str, evidence_text: str
    ) -> tuple[bool, float, str]:
        claim_words = self._tokenize_content_words(claim_text)
        evidence_words = self._tokenize_content_words(evidence_text)

        if not claim_words:
            # Trivial or symbolic sentence
            return True, 1.0, "Claim contains no substantive content words to verify."

        intersection = claim_words & evidence_words
        overlap_score = len(intersection) / len(claim_words)

        if overlap_score >= self.min_token_overlap_threshold:
            return (
                True,
                round(overlap_score, 4),
                f"Supported by lexical overlap ({len(intersection)} shared terms: {', '.join(sorted(intersection)[:5])}).",
            )
        else:
            return (
                False,
                round(overlap_score, 4),
                f"Unrelated evidence: only {len(intersection)} shared content words (overlap {overlap_score:.1%} < threshold {self.min_token_overlap_threshold:.1%}).",
            )


@dataclass
class ClaimAuditResult:
    """Results of claim-level audit for a generated answer."""

    claims: list[Claim]
    total_claims: int
    cited_claims: int
    uncited_claims: int
    supported_claims: int
    unsupported_citation_claims: int
    unrelated_evidence_claims: int
    metadata_contradictions: list[str]


class ClaimLevelAuditor:
    """Audits generated answers at the claim level against retrieved evidence.

    Detects:
    1. Invalid citation tags ([S999], [S998])
    2. Citation omitted on substantive factual claims
    3. Citation to unrelated evidence (zero/low topical overlap with cited chunk)
    4. Explicit document/page metadata mismatches (e.g. claiming wrong doc or page)
    """

    def __init__(
        self,
        extractor: ClaimExtractor | None = None,
        assessor: EntailmentVerifier | None = None,
    ) -> None:
        self.extractor = extractor or SentenceClaimExtractor()
        self.assessor = assessor or HeuristicLexicalOverlapAssessor()

    def audit_claims(self, answer_text: str, evidence_set: EvidenceSet) -> ClaimAuditResult:
        claims = self.extractor.extract_claims(answer_text)
        total_claims = len(claims)
        cited_count = 0
        uncited_count = 0
        supported_count = 0
        unsupported_citation_count = 0
        unrelated_count = 0
        metadata_contradictions: list[str] = []

        for claim in claims:
            if not claim.is_cited:
                claim.support_status = "UNCITED"
                claim.explanation = "Substantive claim has no citation reference."
                uncited_count += 1
                continue

            cited_count += 1
            all_cited_supported = True
            claim_unsupported_citation = False
            claim_unrelated = False

            for sid in claim.cited_source_ids:
                item = evidence_set.get_by_source_id(sid)
                if item is None:
                    claim_unsupported_citation = True
                    all_cited_supported = False
                    claim.contradictions.append(f"Cited non-existent source {sid}")
                    continue

                # Check metadata consistency (e.g. if claim text mentions page or doc explicitly)
                self._check_metadata_consistency(claim, item)

                # Check topical support between claim and cited evidence
                is_supp, score, explanation = self.assessor.verify_support(
                    claim.text, item.text
                )
                claim.support_score = max(claim.support_score, score)

                if not is_supp:
                    claim_unrelated = True
                    all_cited_supported = False
                    claim.explanation = explanation

            if claim_unsupported_citation:
                claim.support_status = "UNSUPPORTED_CITATION"
                unsupported_citation_count += 1
            elif claim_unrelated:
                claim.support_status = "UNRELATED_EVIDENCE"
                unrelated_count += 1
            elif all_cited_supported:
                claim.support_status = "SUPPORTED"
                supported_count += 1

            metadata_contradictions.extend(claim.contradictions)

        return ClaimAuditResult(
            claims=claims,
            total_claims=total_claims,
            cited_claims=cited_count,
            uncited_claims=uncited_count,
            supported_claims=supported_count,
            unsupported_citation_claims=unsupported_citation_count,
            unrelated_evidence_claims=unrelated_count,
            metadata_contradictions=metadata_contradictions,
        )

    def _check_metadata_consistency(self, claim: Claim, item: EvidenceItem) -> None:
        """Flag claims that explicitly specify incorrect page numbers or documents."""
        # Detect explicit page mentions, e.g. "page 999" or "p. 99"
        page_matches = re.findall(r"\b(?:page|p\.)\s*(\d+)\b", claim.text, flags=re.IGNORECASE)
        for pm in page_matches:
            claimed_page = int(pm)
            if item.page_number is not None and claimed_page != item.page_number:
                mismatch = (
                    f"Metadata attack: Claim cites {item.source_id} claiming page {claimed_page}, "
                    f"but {item.source_id} is on page {item.page_number}."
                )
                claim.contradictions.append(mismatch)

        # Detect explicit document name mentions, e.g. "doc: wrong.pdf" or "in textbook.pdf"
        doc_matches = re.findall(r"\b([a-zA-Z0-9_\-]+\.(?:pdf|txt|docx))\b", claim.text, flags=re.IGNORECASE)
        for dm in doc_matches:
            if item.document_id and dm.lower() != item.document_id.lower():
                mismatch = (
                    f"Metadata attack: Claim cites {item.source_id} referencing document '{dm}', "
                    f"but {item.source_id} belongs to '{item.document_id}'."
                )
                claim.contradictions.append(mismatch)
