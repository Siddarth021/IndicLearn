"""Strongly typed domain models for Indic NLP and Concept Mapping (Phase 4).

All NLP analysis entities are structured Pydantic models.
Strictly decoupled from student knowledge state (no mastery scores or ignorance assumptions).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Language(str, Enum):
    """Supported query language classifications."""

    ENGLISH = "en"
    TELUGU = "te"
    ROMANIZED_TELUGU = "te-Latn"
    TELUGU_ENGLISH_CODE_MIXED = "te-en"
    ENGLISH_TELUGU_CODE_MIXED = "en-te"
    UNKNOWN = "unknown"


class Script(str, Enum):
    """Detected orthographic script."""

    LATIN = "Latin"
    TELUGU = "Telugu"
    MIXED = "Mixed"
    UNKNOWN = "Unknown"


class LanguageDetectionResult(BaseModel):
    """Result of language and script identification."""

    primary_language: Language
    detected_script: Script
    confidence: float = Field(ge=0.0, le=1.0)
    script_distribution: dict[str, float] = Field(
        default_factory=dict,
        description="Character script ratios, e.g. {'Latin': 0.8, 'Telugu': 0.2}",
    )
    token_language_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Token language distribution counts, e.g. {'en': 2, 'te-Latn': 3}",
    )
    is_code_mixed: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class CodeMixSpan(BaseModel):
    """A token-level or multi-token span annotated with language and classification."""

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str
    language: Language
    is_technical_term: bool = False
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class CodeMixResult(BaseModel):
    """Comprehensive token-level code-mixing breakdown."""

    original_text: str = ""
    spans: list[CodeMixSpan] = Field(default_factory=list)
    code_mixing_index: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Ratio of non-matrix language tokens (0.0 for monolingual)",
    )
    matrix_language: Language = Language.ENGLISH
    embedded_languages: list[Language] = Field(default_factory=list)

    @property
    def cmi(self) -> float:
        """Alias for code_mixing_index."""
        return self.code_mixing_index


class NormalizationCandidate(BaseModel):
    """An alternative normalized candidate preserving ambiguity."""

    variant_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    transformation_type: str = Field(
        description="Type of normalization: 'phonetic_variant', 'transliteration', 'clean_spelling'",
    )
    explanation: str | None = None

    @property
    def normalized_text(self) -> str:
        """Alias for variant_text."""
        return self.variant_text


class NormalizationResult(BaseModel):
    """Normalization result preserving the original query and candidate variants."""

    original_text: str
    canonical_normalized_text: str
    candidate_variants: list[NormalizationCandidate] = Field(default_factory=list)
    is_romanized: bool = False
    transliterated_telugu: str | None = None
    tokens_normalized: dict[str, str] = Field(default_factory=dict)

    @property
    def normalized_text(self) -> str:
        """Alias property for canonical_normalized_text."""
        return self.canonical_normalized_text


class TechnicalTerm(BaseModel):
    """Identified academic/engineering technical term in query text."""

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    matched_text: str
    canonical_name: str
    concept_id: str | None = None
    term_type: str = Field(
        default="canonical",
        description="'canonical', 'alias', 'acronym', 'telugu_alias', 'romanized_alias'",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @property
    def matched_span(self) -> str:
        """Alias property for matched_text."""
        return self.matched_text


class ConceptCandidate(BaseModel):
    """Ranked concept candidate retrieved for a student query."""

    concept_id: str
    canonical_name: str
    score: float = Field(ge=0.0, le=1.0)
    match_method: str = Field(
        description="'exact_canonical', 'alias_match', 'normalized_lexical', 'semantic_embedding', 'hybrid'",
    )
    evidence_spans: list[str] = Field(default_factory=list)
    rank: int = Field(default=1, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConceptMappingResult(BaseModel):
    """Final mapping from student query to canonical academic concept ID(s)."""

    query: str
    subject_id: str
    primary_concept: ConceptCandidate | None = None
    candidates: list[ConceptCandidate] = Field(default_factory=list)
    is_ambiguous: bool = False
    mapping_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    score_margin: float | None = Field(
        default=None,
        description="Heuristic score difference between top candidate and second-best candidate",
    )
    unmapped_terms: list[str] = Field(default_factory=list)


class NLPAnalysisResult(BaseModel):
    """Unified container encapsulating all NLP pipeline stages for a query."""

    original_text: str
    language_detection: LanguageDetectionResult
    code_mix: CodeMixResult
    normalization: NormalizationResult
    technical_terms: list[TechnicalTerm] = Field(default_factory=list)
    concept_mapping: ConceptMappingResult
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
