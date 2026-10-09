"""Unit tests for Romanization and Unified Normalization (Phase 4)."""

import unicodedata

import pytest

from indiclearn.nlp.models import Language
from indiclearn.nlp.normalization.normalizer import UnifiedQueryNormalizer
from indiclearn.nlp.romanization.normalizer import RuleBasedRomanizationNormalizer


class TestRomanizationAndNormalization:
    """Test suite for Romanization and Text Normalization."""

    @pytest.fixture
    def romanizer(self) -> RuleBasedRomanizationNormalizer:
        return RuleBasedRomanizationNormalizer()

    @pytest.fixture
    def normalizer(self) -> UnifiedQueryNormalizer:
        return UnifiedQueryNormalizer()

    def test_spelling_variants_normalization(
        self, romanizer: RuleBasedRomanizationNormalizer
    ) -> None:
        # Test spelling variation mapping
        res = romanizer.normalize("convolution antey yenti?")
        assert res.canonical_normalized_text == "convolution ante enti?"
        assert res.is_romanized is True

        # Test verb variants
        res2 = romanizer.normalize("system explain cheyi")
        assert res2.canonical_normalized_text == "system explain cheyyi"

        # Test avuthundi -> avutundi
        res3 = romanizer.normalize("aliasing enduku avuthundi?")
        assert "avutundi" in res3.canonical_normalized_text

    def test_ambiguity_preservation(
        self, romanizer: RuleBasedRomanizationNormalizer
    ) -> None:
        res = romanizer.normalize("yenti antey")
        # Ensure candidate variants contain multiple scored hypotheses
        assert len(res.candidate_variants) >= 2
        # Check that original text is untouched
        assert res.original_text == "yenti antey"
        for cand in res.candidate_variants:
            assert cand.confidence > 0.0
            assert cand.normalized_text

    def test_unicode_nfc_handling(self, normalizer: UnifiedQueryNormalizer) -> None:
        # NFD vs NFC Telugu text
        te_text = "కన్వల్యూషన్"
        nfd_text = unicodedata.normalize("NFD", te_text)
        res = normalizer.normalize(nfd_text, primary_language=Language.TELUGU)
        assert res.canonical_normalized_text == te_text
        assert unicodedata.is_normalized("NFC", res.canonical_normalized_text)

    def test_original_query_preservation(
        self, normalizer: UnifiedQueryNormalizer
    ) -> None:
        raw_query = "  convolution   antey  emiti???  "
        res = normalizer.normalize(raw_query)
        # Original text preserves exact input
        assert res.original_text == raw_query
        # Normalized text cleans whitespace and variants
        assert "convolution ante enti" in res.canonical_normalized_text

    def test_transliterated_telugu_generation(
        self, romanizer: RuleBasedRomanizationNormalizer
    ) -> None:
        res = romanizer.normalize("ante enti")
        assert res.transliterated_telugu is not None
        assert "అంటే" in res.transliterated_telugu
        assert "ఏమిటి" in res.transliterated_telugu
