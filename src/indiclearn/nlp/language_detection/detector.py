"""Deterministic heuristic Language and Script Detector (Phase 4).

Identifies English, Telugu script, Romanized Telugu, and code-mixed inputs
using Unicode script ratios and linguistic lexicon heuristics.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from indiclearn.nlp.language_detection.base import LanguageDetector
from indiclearn.nlp.models import Language, LanguageDetectionResult, Script

# Lexicon of frequent Romanized Telugu particles, verbs, and function words
ROMANIZED_TELUGU_WORDS = {
    "ante", "antey", "antee", "enti", "yenti", "emiti", "yemiti",
    "ela", "enduku", "cheppu", "ceppu", "cheyyi", "ceyyi", "chey",
    "ardham", "artham", "kaadu", "kadu", "avuthundi", "avutundi", "avtundi",
    "untundi", "yokka", "lo", "ki", "ni", "nundi", "kosam", "gurinchi",
    "chala", "malli", "mariyu", "kuda", "idi", "adi", "ivi", "avi",
    "cheppandi", "telusukovali", "nerchukovali", "eppudu", "akkada", "ikkada",
    "anaga", "anagane", "evaru", "enta", "ento", "chudu", "pettandi",
    "cheyandi", "cheppava", "cheyava", "undali", "undadu", "ledu",
    "kavale", "kavali", "teliyadu", "anukuntunna", "kada", "kadha",
    "baga", "baaga", "choodu", "ivvandi", "ippudu", "taruvatha",
}

# Common English non-technical function words and pedagogical phrasing
ENGLISH_COMMON_WORDS = {
    "what", "is", "are", "explain", "define", "definition", "how", "why",
    "does", "do", "can", "you", "tell", "give", "me", "the", "a", "an",
    "in", "of", "to", "for", "with", "between", "difference", "and",
    "simple", "easy", "manner", "terms", "overview", "describe", "meaning",
    "details", "concept", "properties", "example", "examples", "solve",
    "calculate", "find", "determine", "state", "prove", "summary",
}

# Common academic technical terms in Signals and Systems
ACADEMIC_TECHNICAL_WORDS = {
    "convolution", "lti", "system", "systems", "signal", "signals",
    "fourier", "transform", "series", "ctft", "dtft", "laplace", "z-transform",
    "impulse", "response", "sampling", "theorem", "nyquist", "rate",
    "aliasing", "roc", "region", "convergence", "stability", "bibo",
    "linearity", "time", "invariance", "causality", "energy", "power",
    "frequency", "discrete", "continuous", "filter", "filters", "poles",
}


class HeuristicLanguageDetector(LanguageDetector):
    """Deterministic, rule-based detector for language and script."""

    def __init__(self, custom_romanized_words: set[str] | None = None) -> None:
        self.romanized_words = set(ROMANIZED_TELUGU_WORDS)
        if custom_romanized_words:
            self.romanized_words.update(w.lower() for w in custom_romanized_words)
        self.english_words = set(ENGLISH_COMMON_WORDS)
        self.technical_words = set(ACADEMIC_TECHNICAL_WORDS)

    def detect(self, text: str) -> LanguageDetectionResult:
        """Detect language and script from text."""
        norm_text = unicodedata.normalize("NFC", text or "").strip()
        if not norm_text:
            return LanguageDetectionResult(
                primary_language=Language.UNKNOWN,
                detected_script=Script.UNKNOWN,
                confidence=0.0,
                script_distribution={"Latin": 0.0, "Telugu": 0.0, "Other": 0.0},
                token_language_counts={},
                is_code_mixed=False,
                metadata={"reason": "empty_input"},
            )

        # 1. Character-level script distribution
        total_chars = 0
        telugu_chars = 0
        latin_chars = 0
        other_chars = 0

        for ch in norm_text:
            if ch.isspace() or unicodedata.category(ch).startswith("P"):
                continue
            total_chars += 1
            code = ord(ch)
            if 0x0C00 <= code <= 0x0C7F:
                telugu_chars += 1
            elif ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
                latin_chars += 1
            else:
                other_chars += 1

        if total_chars == 0:
            return LanguageDetectionResult(
                primary_language=Language.UNKNOWN,
                detected_script=Script.UNKNOWN,
                confidence=0.0,
                script_distribution={"Latin": 0.0, "Telugu": 0.0, "Other": 0.0},
                token_language_counts={},
                is_code_mixed=False,
                metadata={"reason": "no_alphanumeric_characters"},
            )

        te_ratio = round(telugu_chars / total_chars, 4)
        lat_ratio = round(latin_chars / total_chars, 4)
        oth_ratio = round(other_chars / total_chars, 4)
        script_dist = {"Latin": lat_ratio, "Telugu": te_ratio, "Other": oth_ratio}

        # 2. Token-level classification
        raw_tokens = re.findall(r"[\w\-]+", norm_text.lower(), flags=re.UNICODE)
        token_counts: dict[str, int] = {
            "te_script": 0,
            "te_romanized": 0,
            "en_common": 0,
            "technical": 0,
            "unknown": 0,
        }

        for token in raw_tokens:
            # Check script of token
            has_te = any(0x0C00 <= ord(c) <= 0x0C7F for c in token)
            has_lat = any(("a" <= c <= "z") or ("A" <= c <= "Z") for c in token)

            if has_te and not has_lat:
                token_counts["te_script"] += 1
            elif token in self.technical_words:
                token_counts["technical"] += 1
            elif token in self.romanized_words:
                token_counts["te_romanized"] += 1
            elif token in self.english_words:
                token_counts["en_common"] += 1
            elif has_te and has_lat:
                token_counts["te_script"] += 1  # Mixed script token
            else:
                token_counts["unknown"] += 1

        # 3. Determine Script
        if te_ratio > 0.85:
            detected_script = Script.TELUGU
        elif lat_ratio > 0.85:
            detected_script = Script.LATIN
        elif te_ratio > 0.10 and lat_ratio > 0.10:
            detected_script = Script.MIXED
        elif lat_ratio > 0.0:
            detected_script = Script.LATIN
        elif te_ratio > 0.0:
            detected_script = Script.TELUGU
        else:
            detected_script = Script.UNKNOWN

        # 4. Determine Primary Language
        is_code_mixed = False
        meta: dict[str, Any] = {}

        # Case A: Pure or predominant Telugu script
        if detected_script == Script.TELUGU:
            if token_counts["technical"] > 0 or token_counts["en_common"] > 0:
                primary_language = Language.TELUGU_ENGLISH_CODE_MIXED
                is_code_mixed = True
            else:
                primary_language = Language.TELUGU
            confidence = min(0.95, 0.70 + te_ratio * 0.25)

        # Case B: Mixed scripts (e.g. "LTI సిస్టమ్ అంటే ఏమిటి?")
        elif detected_script == Script.MIXED:
            primary_language = Language.TELUGU_ENGLISH_CODE_MIXED
            is_code_mixed = True
            confidence = 0.90

        # Case C: Latin script (English, Romanized Telugu, or Code-mixed Latin)
        else:
            te_rom_cnt = token_counts["te_romanized"]
            en_cnt = token_counts["en_common"]
            tech_cnt = token_counts["technical"]

            # Code-mixed in Latin script (e.g. "convolution ante enti explain cheyyi")
            if te_rom_cnt > 0 and en_cnt > 0:
                primary_language = Language.TELUGU_ENGLISH_CODE_MIXED
                is_code_mixed = True
                confidence = 0.90
            # Romanized Telugu with optional technical term (e.g. "convolution ante enti?")
            elif te_rom_cnt > 0:
                primary_language = Language.ROMANIZED_TELUGU
                is_code_mixed = tech_cnt > 0  # Technical borrowing
                confidence = min(0.95, 0.75 + (te_rom_cnt / max(1, len(raw_tokens))) * 0.20)
            # Pure English with technical terms (e.g. "What is convolution?")
            elif en_cnt > 0:
                primary_language = Language.ENGLISH
                is_code_mixed = False
                confidence = min(0.95, 0.75 + (en_cnt / max(1, len(raw_tokens))) * 0.20)
            # Only technical words or acronyms (e.g. "LTI System", "Convolution")
            elif tech_cnt > 0:
                primary_language = Language.ENGLISH
                is_code_mixed = False
                confidence = 0.85
                meta["note"] = "classified_by_technical_lexicon"
            else:
                primary_language = Language.UNKNOWN
                confidence = 0.40

        return LanguageDetectionResult(
            primary_language=primary_language,
            detected_script=detected_script,
            confidence=round(confidence, 4),
            script_distribution=script_dist,
            token_language_counts=token_counts,
            is_code_mixed=is_code_mixed,
            metadata=meta,
        )
