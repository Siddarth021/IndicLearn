"""Curriculum Scope Assessment, Ambiguity Detection, and Retrieval Hardening (Phase 5.2).

Implements Section 4-9:
- ScopeAssessor Protocol & HeuristicScopeAssessor
- Explicit 4-state classification: IN_SCOPE, OUT_OF_SCOPE, AMBIGUOUS, INSUFFICIENT_EVIDENCE
- Multi-signal evaluation:
    Signal A: Canonical concept mapping
    Signal B: Technical terminology specificity (generic vs specific terms)
    Signal C: Concept candidate margin (ambiguity when runner-up is near-identical)
    Signal D: Evidence retrieval strength
    Signal E: Concept-to-evidence consistency
    Signal F: Query-to-concept semantic alignment (guards against generic term hijacking)

Scientific Principle:
    A lexical keyword match (e.g. 'filter', 'signal', 'system') is NOT curriculum grounding.
    Heuristic confidence scores are deterministic ranking metrics, NOT calibrated probabilities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import ClassVar, Protocol, runtime_checkable

from indiclearn.domain.concepts import Concept
from indiclearn.nlp.models import NLPAnalysisResult
from indiclearn.retrieval.base import RetrievalResult


@dataclass
class ScopeDecision:
    """Explicit scope determination for a student query."""

    query: str
    scope: str  # 'IN_SCOPE', 'OUT_OF_SCOPE', 'AMBIGUOUS', 'INSUFFICIENT_EVIDENCE'
    confidence_score: float  # Heuristic ranking score in [0.0, 1.0], NOT calibrated probability
    primary_concept_id: str | None = None
    matched_terms: list[str] = field(default_factory=list)
    generic_terms_detected: list[str] = field(default_factory=list)
    out_of_scope_indicators: list[str] = field(default_factory=list)
    concept_margin: float | None = None
    evidence_consistency_score: float = 0.0
    reason: str = ""


@runtime_checkable
class ScopeAssessor(Protocol):
    """Protocol for curriculum scope determination and retrieval safety."""

    def assess_scope(
        self,
        query: str,
        nlp_result: NLPAnalysisResult,
        retrieval_candidates: list[RetrievalResult],
        curriculum_concepts: list[Concept] | None = None,
    ) -> ScopeDecision:
        """Determine whether query is IN_SCOPE, OUT_OF_SCOPE, AMBIGUOUS, or INSUFFICIENT_EVIDENCE."""
        ...


class HeuristicScopeAssessor:
    """Deterministic, configurable, multi-signal curriculum scope assessor."""

    # Broad technical terms common across multiple engineering disciplines
    GENERIC_TERMS: ClassVar[set[str]] = {
        "system",
        "systems",
        "signal",
        "signals",
        "filter",
        "filters",
        "transform",
        "transforms",
        "algorithm",
        "algorithms",
        "response",
        "responses",
        "frequency",
        "frequencies",
        "function",
        "functions",
        "process",
        "processing",
        # Telugu script and transliterated generic equivalents
        "సిస్టమ్",
        "సిస్టమ్స్",
        "సిగ్నల్",
        "సిగ్నల్స్",
        "ఫిల్టర్",
        "ఫిల్టర్స్",
        "ట్రాన్స్‌ఫార్మ్",
        "ఫ్రీక్వెన్సీ",
        "రెస్పాన్స్",
        "ఆల్గారిథమ్",
    }

    # Distinct domain markers outside introductory continuous-time Signals & Systems
    OUT_OF_SCOPE_DOMAINS: ClassVar[dict[str, list[str]]] = {
        "digital_filter_design": [
            r"\bkaiser\b",
            r"\bbutterworth\b",
            r"\bchebyshev\b",
            r"\belliptic\b",
            r"\bbilinear\s+transform(?:ation)?\b",
            r"\biir\s+filter\b",
            r"\bfir\s+(?:digital\s+)?filter\b",
            r"\bparks[- ]mcclellan\b",
            r"\bwindowing\s+techniques?\b",
        ],
        "control_systems": [
            r"\bkalman\b",
            r"\bstate[- ]space\b",
            r"\bcontrollability\b",
            r"\bobservability\b",
            r"\bpid\s+controller\b",
            r"\bbode\s+plot\b",
            r"\broot\s+locus\b",
            r"\bpole\s+placement\b",
            r"\bfeedback\s+control\b",
        ],
        "image_and_multidimensional": [
            r"\bjpeg\b",
            r"\bimage\s+compression\b",
            r"\bimage\s+processing\b",
            r"\bdiscrete\s+cosine\s+transform\b",
            r"\bdct\b",
            r"\bwavelet\b",
            r"\bdwt\b",
            r"\b2d\s+fourier\b",
            r"\bpixel\b",
        ],
        "digital_communications": [
            r"\bviterbi\b",
            r"\bconvolutional\s+code\b",
            r"\bturbo\s+code\b",
            r"\bofdm\b",
            r"\bqam\b",
            r"\bpsk\b",
            r"\bconstellation\b",
            r"\btrellis\b",
        ],
        "machine_learning_adaptive": [
            r"\badaptive\s+(?:lms\s+)?filter\b",
            r"\blms\b",
            r"\bleast\s+mean\s+squares\b",
            r"\bgradient\s+descent\b",
            r"\bweight\s+update\b",
            r"\bneural\s+network\b",
        ],
        "dsp_fft_algorithms": [
            r"\bcooley[- ]tukey\b",
            r"\bbutterfly\s+algorithm\b",
            r"\bdecimation\s+in\s+time\b",
            r"\bdecimation\s+in\s+frequency\b",
            r"\bfft\s+algorithm\b",
        ],
    }

    # Under-specified patterns that reference only generic words without technical qualifiers
    AMBIGUOUS_PATTERNS: ClassVar[list[str]] = [
        r"^(?:what\s+(?:is|are)\s+(?:a\s+|an\s+)?|explain\s+|define\s+)?(?:filters?|systems?|signals?|transforms?)$",
        r"^(?:filters?|systems?|signals?|transforms?)\s+(?:ante\s+enti|yenti|emiti)$",
        r"^(?:ఫిల్టర్|సిస్టమ్|సిగ్నల్|ట్రాన్స్‌ఫార్మ్)\s+అంటే\s+ఏమిటి$",
        r"^(?:filters?|systems?|signals?)\s+ni\s+explain\s+cheyyi$",
    ]

    def __init__(
        self,
        min_concept_score: float = 0.40,
        min_margin_threshold: float = 0.08,
        min_evidence_score: float = 0.007,
    ) -> None:
        self.min_concept_score = min_concept_score
        self.min_margin_threshold = min_margin_threshold
        self.min_evidence_score = min_evidence_score

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b[a-zA-Z0-9_\u0C00-\u0C7F]{2,}\b", text.lower())

    def _detect_out_of_scope_indicators(self, text: str) -> list[str]:
        indicators: list[str] = []
        lower = text.lower()
        for domain, patterns in self.OUT_OF_SCOPE_DOMAINS.items():
            for pat in patterns:
                match = re.search(pat, lower)
                if match:
                    indicators.append(f"{domain}:{match.group(0)}")
        return indicators

    def _is_ambiguous_phrase(self, text: str) -> bool:
        clean = re.sub(r"[.?!,;:\s]+$", "", text.strip().lower())
        for pat in self.AMBIGUOUS_PATTERNS:
            if re.match(pat, clean):
                return True
        return False

    def assess_scope(
        self,
        query: str,
        nlp_result: NLPAnalysisResult,
        retrieval_candidates: list[RetrievalResult],
        curriculum_concepts: list[Concept] | None = None,
    ) -> ScopeDecision:
        """Evaluate multi-signal scope evidence to produce a deterministic decision."""
        # 1. Check for explicit out-of-scope domain markers (Signal F / Adversarial guard)
        norm_text = nlp_result.normalization.normalized_text
        combined_text = f"{query} {norm_text}"
        out_indicators = self._detect_out_of_scope_indicators(combined_text)

        tokens = self._tokenize(combined_text)
        generic_found = [t for t in tokens if t in self.GENERIC_TERMS]

        # 2. Check for under-specified ambiguous query phrases
        if self._is_ambiguous_phrase(query) or self._is_ambiguous_phrase(norm_text):
            return ScopeDecision(
                query=query,
                scope="AMBIGUOUS",
                confidence_score=0.30,
                primary_concept_id=(
                    nlp_result.concept_mapping.primary_concept.concept_id
                    if nlp_result.concept_mapping.primary_concept
                    else None
                ),
                matched_terms=[],
                generic_terms_detected=generic_found,
                out_of_scope_indicators=out_indicators,
                concept_margin=nlp_result.concept_mapping.score_margin,
                evidence_consistency_score=0.0,
                reason=(
                    "Query is too broad or under-specified. Generic engineering term used without "
                    "identifying course-specific context."
                ),
            )

        # 3. If explicit out-of-scope domain markers are present:
        # Even if generic engineering terms match concepts (e.g. 'filter', 'signal'), reject as OUT_OF_SCOPE
        if out_indicators:
            return ScopeDecision(
                query=query,
                scope="OUT_OF_SCOPE",
                confidence_score=0.10,
                primary_concept_id=None,
                matched_terms=[],
                generic_terms_detected=generic_found,
                out_of_scope_indicators=out_indicators,
                concept_margin=nlp_result.concept_mapping.score_margin,
                evidence_consistency_score=0.0,
                reason=(
                    f"Query references external academic domains outside introductory Signals & Systems: "
                    f"{', '.join(out_indicators)}."
                ),
            )

        # 4. Evaluate Concept Mapping Evidence (Signal A, B, C)
        primary = nlp_result.concept_mapping.primary_concept
        has_primary = primary is not None and primary.score >= self.min_concept_score

        # Check if mapped concept name is merely a generic term (e.g., 'reconstruction_filter' matching on 'filter')
        primary_is_solely_generic = False
        if primary:
            cpt_words = set(self._tokenize(primary.canonical_name))
            non_generic_cpt_words = cpt_words - self.GENERIC_TERMS
            query_non_generic = set(tokens) - self.GENERIC_TERMS
            # If the intersection between non-generic query tokens and non-generic concept words is empty
            if (
                non_generic_cpt_words
                and not (query_non_generic & non_generic_cpt_words)
                and primary.match_method not in {"exact_canonical", "alias_match"}
            ):
                primary_is_solely_generic = True

        margin = nlp_result.concept_mapping.score_margin
        is_ambiguous_margin = (
            margin is not None
            and margin < self.min_margin_threshold
            and has_primary
            and primary.score < 0.80
        )

        # 5. Evaluate Retrieval Evidence & Consistency (Signal D, E)
        strong_candidates = [
            c for c in retrieval_candidates if c.score >= self.min_evidence_score
        ]
        evidence_consistency_score = 0.0

        if primary and strong_candidates:
            # Check how many retrieved top chunks contain primary concept or its canonical words
            target_cid = primary.concept_id
            consistent_count = 0
            for cand in strong_candidates[:5]:
                cand_cids = cand.metadata.get("concept_ids", []) if cand.metadata else []
                if target_cid in cand_cids or primary.canonical_name.lower() in cand.text.lower():
                    consistent_count += 1
            evidence_consistency_score = round(consistent_count / min(len(strong_candidates), 5), 4)

        # 6. Synthesize Scope Decision
        if is_ambiguous_margin and primary_is_solely_generic:
            return ScopeDecision(
                query=query,
                scope="AMBIGUOUS",
                confidence_score=0.40,
                primary_concept_id=primary.concept_id if primary else None,
                matched_terms=[primary.canonical_name] if primary else [],
                generic_terms_detected=generic_found,
                out_of_scope_indicators=[],
                concept_margin=margin,
                evidence_consistency_score=evidence_consistency_score,
                reason=(
                    "Ambiguous concept mapping: Competing concept candidates have near-identical "
                    f"ranking scores (margin={margin}) on generic terminology."
                ),
            )

        if not has_primary and not strong_candidates:
            return ScopeDecision(
                query=query,
                scope="OUT_OF_SCOPE",
                confidence_score=0.05,
                primary_concept_id=None,
                matched_terms=[],
                generic_terms_detected=generic_found,
                out_of_scope_indicators=[],
                concept_margin=margin,
                evidence_consistency_score=0.0,
                reason="Query does not map to any curriculum concept and retrieves zero relevant evidence chunks.",
            )

        if not strong_candidates and has_primary:
            return ScopeDecision(
                query=query,
                scope="INSUFFICIENT_EVIDENCE",
                confidence_score=0.35,
                primary_concept_id=primary.concept_id,
                matched_terms=[primary.canonical_name],
                generic_terms_detected=generic_found,
                out_of_scope_indicators=[],
                concept_margin=margin,
                evidence_consistency_score=0.0,
                reason=(
                    f"Query maps to curriculum concept '{primary.canonical_name}', but uploaded course material "
                    "contains insufficient supporting evidence chunks."
                ),
            )

        # When primary concept exists and evidence is found
        if has_primary and not primary_is_solely_generic:
            # Strong in-scope curriculum question
            heuristic_conf = min(
                1.0,
                0.40
                + (0.30 if primary.score >= 0.70 else 0.15)
                + (0.20 if evidence_consistency_score >= 0.20 else 0.05)
                + (0.10 if (margin or 1.0) >= self.min_margin_threshold else 0.0),
            )
            return ScopeDecision(
                query=query,
                scope="IN_SCOPE",
                confidence_score=round(heuristic_conf, 4),
                primary_concept_id=primary.concept_id,
                matched_terms=[primary.canonical_name],
                generic_terms_detected=generic_found,
                out_of_scope_indicators=[],
                concept_margin=margin,
                evidence_consistency_score=evidence_consistency_score,
                reason=f"Strong curriculum grounding for concept '{primary.canonical_name}' with verified evidence chunks.",
            )

        # If primary concept exists but was matched solely on generic terms and evidence consistency is zero
        if primary_is_solely_generic and evidence_consistency_score == 0.0:
            return ScopeDecision(
                query=query,
                scope="OUT_OF_SCOPE",
                confidence_score=0.20,
                primary_concept_id=None,
                matched_terms=[],
                generic_terms_detected=generic_found,
                out_of_scope_indicators=["generic_hijack_unsupported"],
                concept_margin=margin,
                evidence_consistency_score=0.0,
                reason=(
                    "Query matched concept candidate solely via generic terminology and lacks "
                    "content consistency with retrieved curriculum evidence."
                ),
            )

        # Baseline fallback for queries with moderate lexical/semantic match
        if strong_candidates:
            return ScopeDecision(
                query=query,
                scope="IN_SCOPE",
                confidence_score=0.60,
                primary_concept_id=primary.concept_id if primary else None,
                matched_terms=[primary.canonical_name] if primary else [],
                generic_terms_detected=generic_found,
                out_of_scope_indicators=[],
                concept_margin=margin,
                evidence_consistency_score=evidence_consistency_score,
                reason="Corpus retrieval yielded consistent academic evidence for the query.",
            )

        return ScopeDecision(
            query=query,
            scope="INSUFFICIENT_EVIDENCE",
            confidence_score=0.20,
            primary_concept_id=None,
            matched_terms=[],
            generic_terms_detected=generic_found,
            out_of_scope_indicators=[],
            concept_margin=margin,
            evidence_consistency_score=0.0,
            reason="Insufficient evidence to establish safe curriculum scope.",
        )
