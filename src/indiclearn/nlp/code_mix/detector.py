"""Token-level code-mixing analyzer and CMI metric calculator (Phase 4)."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from indiclearn.nlp.code_mix.base import CodeMixDetector
from indiclearn.nlp.language_detection.detector import (
    ACADEMIC_TECHNICAL_WORDS,
    ENGLISH_COMMON_WORDS,
    ROMANIZED_TELUGU_WORDS,
)
from indiclearn.nlp.models import CodeMixResult, CodeMixSpan, Language


class RuleBasedCodeMixDetector(CodeMixDetector):
    """Token-level span classifier and Code-Mixing Index (CMI) analyzer."""

    def __init__(
        self,
        romanized_words: set[str] | None = None,
        technical_words: set[str] | None = None,
    ) -> None:
        self.romanized_words = set(ROMANIZED_TELUGU_WORDS)
        if romanized_words:
            self.romanized_words.update(w.lower() for w in romanized_words)
        self.technical_words = set(ACADEMIC_TECHNICAL_WORDS)
        if technical_words:
            self.technical_words.update(w.lower() for w in technical_words)
        self.english_words = set(ENGLISH_COMMON_WORDS)

    def analyze(self, text: str) -> CodeMixResult:
        """Analyze text at the span level, classifying each token's language."""
        norm_text = unicodedata.normalize("NFC", text or "")
        if not norm_text.strip():
            return CodeMixResult(
                spans=[],
                code_mixing_index=0.0,
                matrix_language=Language.UNKNOWN,
                embedded_languages=[],
            )

        spans: list[CodeMixSpan] = []
        lang_counts: Counter[Language] = Counter()

        # Tokenize preserving character offsets
        token_pattern = re.compile(r"[\w\-]+", flags=re.UNICODE)
        for match in token_pattern.finditer(norm_text):
            token = match.group(0)
            low_token = token.lower()
            start, end = match.span()

            has_te = any(0x0C00 <= ord(c) <= 0x0C7F for c in token)
            has_lat = any(("a" <= c <= "z") or ("A" <= c <= "Z") for c in token)

            is_tech = low_token in self.technical_words

            if has_te:
                lang = Language.TELUGU
            elif is_tech:
                lang = Language.ENGLISH
            elif low_token in self.romanized_words:
                lang = Language.ROMANIZED_TELUGU
            elif low_token in self.english_words:
                lang = Language.ENGLISH
            elif has_lat:
                # Disambiguation: if length <= 2 and common Indic suffix/postposition
                if low_token in {"lo", "ki", "ni", "ga", "to", "la"}:
                    lang = Language.ROMANIZED_TELUGU
                else:
                    lang = Language.ENGLISH
            else:
                lang = Language.UNKNOWN

            spans.append(
                CodeMixSpan(
                    start=start,
                    end=end,
                    text=token,
                    language=lang,
                    is_technical_term=is_tech,
                    confidence=0.90 if lang != Language.UNKNOWN else 0.50,
                )
            )

            if lang != Language.UNKNOWN:
                lang_counts[lang] += 1

        total_lang_tokens = sum(lang_counts.values())
        if total_lang_tokens == 0:
            return CodeMixResult(
                original_text=text or "",
                spans=spans,
                code_mixing_index=0.0,
                matrix_language=Language.UNKNOWN,
                embedded_languages=[],
            )

        matrix_lang, max_count = lang_counts.most_common(1)[0]
        cmi = (total_lang_tokens - max_count) / total_lang_tokens
        embedded = [l for l, c in lang_counts.items() if l != matrix_lang and c > 0]

        return CodeMixResult(
            original_text=text or "",
            spans=spans,
            code_mixing_index=round(cmi, 4),
            matrix_language=matrix_lang,
            embedded_languages=embedded,
        )

    def detect_spans(self, text: str) -> CodeMixResult:
        """Alias for analyze method."""
        return self.analyze(text)
