"""Unit tests for Language and Script Identification (Phase 4)."""

import pytest

from indiclearn.nlp.language_detection.detector import HeuristicLanguageDetector
from indiclearn.nlp.models import Language, Script


class TestLanguageDetection:
    """Test suite for HeuristicLanguageDetector."""

    @pytest.fixture
    def detector(self) -> HeuristicLanguageDetector:
        return HeuristicLanguageDetector()

    def test_english_query(self, detector: HeuristicLanguageDetector) -> None:
        res = detector.detect("What is convolution?")
        assert res.primary_language == Language.ENGLISH
        assert res.detected_script == Script.LATIN
        assert res.is_code_mixed is False

    def test_telugu_script_query(self, detector: HeuristicLanguageDetector) -> None:
        res = detector.detect("కన్వల్యూషన్ అంటే ఏమిటి?")
        assert res.primary_language == Language.TELUGU
        assert res.detected_script == Script.TELUGU
        assert res.is_code_mixed is False

    def test_romanized_telugu_query(self, detector: HeuristicLanguageDetector) -> None:
        # Technical English word with Romanized Telugu question words
        res = detector.detect("convolution ante enti?")
        assert res.primary_language == Language.ROMANIZED_TELUGU
        assert res.detected_script == Script.LATIN

    def test_code_mixed_query(self, detector: HeuristicLanguageDetector) -> None:
        # English imperative + Romanized Telugu
        res = detector.detect("convolution ante enti explain cheyyi")
        assert res.primary_language in [
            Language.TELUGU_ENGLISH_CODE_MIXED,
            Language.ROMANIZED_TELUGU,
        ]
        assert res.detected_script == Script.LATIN

    def test_empty_input(self, detector: HeuristicLanguageDetector) -> None:
        res = detector.detect("")
        assert res.primary_language == Language.UNKNOWN
        assert res.detected_script == Script.UNKNOWN
        assert res.confidence == 0.0

    def test_whitespace_and_punctuation_only(
        self, detector: HeuristicLanguageDetector
    ) -> None:
        res = detector.detect("   ??? !!! ... ,,, ")
        assert res.primary_language == Language.UNKNOWN
        assert res.detected_script == Script.UNKNOWN

    def test_numbers_only(self, detector: HeuristicLanguageDetector) -> None:
        res = detector.detect("12345 67890")
        assert res.primary_language == Language.UNKNOWN

    def test_technical_english_inside_telugu_script(
        self, detector: HeuristicLanguageDetector
    ) -> None:
        # LTI acronym in Latin script followed by Telugu text
        res = detector.detect("LTI సిస్టమ్ అంటే ఏమిటి?")
        assert res.detected_script in [Script.MIXED, Script.TELUGU]
        assert res.primary_language in [
            Language.TELUGU,
            Language.TELUGU_ENGLISH_CODE_MIXED,
        ]

    def test_script_ratios_populated(self, detector: HeuristicLanguageDetector) -> None:
        res = detector.detect("Hello కన్వల్యూషన్")
        assert "Latin" in res.script_distribution
        assert "Telugu" in res.script_distribution
        assert res.detected_script == Script.MIXED
