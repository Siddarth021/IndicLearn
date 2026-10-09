"""Grounding verification and factual citation auditing for RAG (Phase 5).

Audits answer text to verify:
1. Citation validity (all cited source IDs exist in supplied evidence)
2. Citation coverage (claims reference supporting sources)
3. Detection of unsupported claims or fabricated citations
4. Evidence sufficiency integrity
"""

from __future__ import annotations

import re

from indiclearn.rag.citation import CitationValidator
from indiclearn.rag.models import EvidenceSet, GroundingCheck, GroundingReport


class GroundingVerifier:
    """Verifies citation validity, factual coverage, and evidence grounding."""

    def __init__(
        self,
        validator: CitationValidator | None = None,
        min_coverage_threshold: float = 0.40,
    ) -> None:
        self.validator = validator or CitationValidator()
        self.min_coverage_threshold = min_coverage_threshold

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences while ignoring blank or trivial lines."""
        raw_sentences = re.split(r"(?<=[.!?\n])\s+", text)
        return [s.strip() for s in raw_sentences if len(s.strip()) > 15]

    def verify_grounding(
        self,
        answer_text: str,
        evidence_set: EvidenceSet,
        is_insufficient_evidence_response: bool = False,
        fallback_status: str | None = None,
    ) -> GroundingReport:
        """Perform comprehensive grounding verification on the generated answer.

        Args:
            answer_text: Model-generated answer.
            evidence_set: Evidence items supplied in the prompt.
            is_insufficient_evidence_response: True if the pipeline/model returned
                an explicit fallback response.
            fallback_status: Status to assign on fallback ('INSUFFICIENT_EVIDENCE', 'OUT_OF_SCOPE', 'AMBIGUOUS_SCOPE').

        Returns:
            GroundingReport with quantitative metrics and audit status.
        """
        # If response was an explicit fallback or insufficient evidence
        if is_insufficient_evidence_response or evidence_set.insufficient_evidence:
            assigned_status = fallback_status or "INSUFFICIENT_EVIDENCE"
            check = GroundingCheck(
                is_grounded=True,
                citation_validity_rate=1.0,
                citation_coverage=0.0,
                unsupported_citation_count=0,
                unsupported_citation_rate=0.0,
                evidence_sufficiency=False,
                invalid_citations=[],
                unsupported_claims=[],
                explanation=f"Correctly determined that query cannot be answered ({assigned_status}).",
            )
            return GroundingReport(
                check=check,
                status=assigned_status,
                citations_parsed=[],
                valid_citations=[],
                invalid_citations=[],
            )

        all_citations, valid_citations, invalid_ids = self.validator.validate_citations(
            answer_text, evidence_set
        )

        total_citations = len(all_citations)
        valid_count = len(valid_citations)
        unsupported_count = len(invalid_ids)

        validity_rate = valid_count / total_citations if total_citations > 0 else 0.0
        unsupported_rate = unsupported_count / total_citations if total_citations > 0 else 0.0

        # Calculate sentence-level citation coverage
        sentences = self._split_into_sentences(answer_text)
        cited_sentences_count = 0
        unsupported_claims: list[str] = []

        valid_source_ids = {c.source_id for c in valid_citations}

        for sent in sentences:
            has_valid_citation = False
            for sid in valid_source_ids:
                if f"[{sid}]" in sent or f"[{sid}," in sent or f", {sid}]" in sent or f",{sid}]" in sent:
                    has_valid_citation = True
                    break
            if has_valid_citation:
                cited_sentences_count += 1
            else:
                # Substantive sentence without citation
                unsupported_claims.append(sent)

        coverage = (
            cited_sentences_count / len(sentences)
            if sentences
            else (1.0 if valid_count > 0 else 0.0)
        )

        # Grounding condition:
        # 1. No invalid/fabricated citations
        # 2. At least one valid citation
        # 3. Citation coverage >= threshold
        is_grounded = (
            unsupported_count == 0
            and valid_count > 0
            and coverage >= self.min_coverage_threshold
        )

        if is_grounded:
            status = "GROUNDED"
            explanation = (
                f"Fully grounded. All {valid_count} citations are valid and map to supplied "
                f"sources (coverage: {coverage:.1%})."
            )
        elif valid_count > 0 and unsupported_count == 0:
            status = "PARTIALLY_GROUNDED"
            explanation = (
                f"Partially grounded. All {valid_count} citations are valid, but coverage "
                f"({coverage:.1%}) is below the required {self.min_coverage_threshold:.1%} threshold."
            )
        else:
            status = "UNSUPPORTED"
            explanation = (
                f"Grounding failure. Detected {unsupported_count} invalid or unsupplied citations "
                f"({', '.join(invalid_ids) if invalid_ids else 'none supplied'})."
            )

        check = GroundingCheck(
            is_grounded=is_grounded,
            citation_validity_rate=round(validity_rate, 4),
            citation_coverage=round(coverage, 4),
            unsupported_citation_count=unsupported_count,
            unsupported_citation_rate=round(unsupported_rate, 4),
            evidence_sufficiency=True,
            invalid_citations=invalid_ids,
            unsupported_claims=unsupported_claims,
            explanation=explanation,
        )

        return GroundingReport(
            check=check,
            status=status,
            citations_parsed=all_citations,
            valid_citations=valid_citations,
            invalid_citations=invalid_ids,
        )
