"""Unified text normalizer integrating NFC cleaning and Romanization normalization (Phase 4)."""

from __future__ import annotations

import re
import unicodedata

from indiclearn.nlp.models import Language, NormalizationResult
from indiclearn.nlp.normalization.base import TextNormalizer
from indiclearn.nlp.romanization.normalizer import RuleBasedRomanizationNormalizer


class UnifiedQueryNormalizer(TextNormalizer):
    """Orchestrates Unicode normalization, whitespace hygiene, and Romanized Telugu normalization."""

    def __init__(
        self,
        romanization_normalizer: RuleBasedRomanizationNormalizer | None = None,
    ) -> None:
        self.romanization_normalizer = romanization_normalizer or RuleBasedRomanizationNormalizer()

    def normalize(
        self,
        text: str,
        primary_language: Language | None = None,
    ) -> NormalizationResult:
        """Clean and normalize student query text while preserving original text."""
        raw = text or ""
        # 1. Unicode NFC normalization (crucial for Indic vowels & conjuncts)
        nfc_text = unicodedata.normalize("NFC", raw)

        # 2. Collapse internal whitespace but preserve general punctuation
        cleaned = re.sub(r"[ \t\r\f\v]+", " ", nfc_text).strip()

        # 3. Delegate to Romanization normalizer
        res = self.romanization_normalizer.normalize(cleaned)

        # Preserve the absolute raw input in original_text
        return NormalizationResult(
            original_text=raw,
            canonical_normalized_text=res.canonical_normalized_text,
            candidate_variants=res.candidate_variants,
            is_romanized=res.is_romanized,
            transliterated_telugu=res.transliterated_telugu,
            tokens_normalized=res.tokens_normalized,
        )
