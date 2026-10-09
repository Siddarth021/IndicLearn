"""Unit tests for Code-Mix Span Detection (Phase 4)."""

import pytest

from indiclearn.nlp.code_mix.detector import RuleBasedCodeMixDetector
from indiclearn.nlp.models import Language


class TestCodeMixDetection:
    """Test suite for RuleBasedCodeMixDetector."""

    @pytest.fixture
    def detector(self) -> RuleBasedCodeMixDetector:
        return RuleBasedCodeMixDetector()

    def test_code_mixed_span_extraction(self, detector: RuleBasedCodeMixDetector) -> None:
        query = "convolution ante enti explain cheyyi"
        res = detector.detect_spans(query)

        assert res.original_text == query
        assert len(res.spans) >= 4
        # Verify tokens like 'ante' are identified as Romanized Telugu
        ante_span = next((s for s in res.spans if s.text.lower() == "ante"), None)
        assert ante_span is not None
        assert ante_span.language == Language.ROMANIZED_TELUGU

        # Verify 'explain' is identified as English
        explain_span = next((s for s in res.spans if s.text.lower() == "explain"), None)
        assert explain_span is not None
        assert explain_span.language == Language.ENGLISH

    def test_pure_english_query(self, detector: RuleBasedCodeMixDetector) -> None:
        query = "What is continuous-time convolution?"
        res = detector.detect_spans(query)
        assert res.cmi == 0.0
        assert res.matrix_language == Language.ENGLISH

    def test_pure_telugu_query(self, detector: RuleBasedCodeMixDetector) -> None:
        query = "కన్వల్యూషన్ అంటే ఏమిటి?"
        res = detector.detect_spans(query)
        assert res.matrix_language == Language.TELUGU

    def test_empty_string(self, detector: RuleBasedCodeMixDetector) -> None:
        res = detector.detect_spans("")
        assert len(res.spans) == 0
        assert res.cmi == 0.0
        assert res.matrix_language == Language.UNKNOWN
