"""Rule-based normalizer for Romanized Telugu student questions (Phase 4).

Standardizes irregular transliterated spellings, handles phonetic variations,
and generates structured normalization candidates preserving ambiguity.
"""

from __future__ import annotations

import re
import unicodedata

from indiclearn.nlp.models import NormalizationCandidate, NormalizationResult
from indiclearn.nlp.romanization.base import RomanizationNormalizer

# Mapping from informal/irregular Romanized forms to (canonical_romanized, telugu_script)
ROMANIZED_CANONICAL_MAP: dict[str, tuple[str, str]] = {
    # Question words
    "enti": ("enti", "ఏమిటి"),
    "yenti": ("enti", "ఏమిటి"),
    "emiti": ("enti", "ఏమిటి"),
    "yemiti": ("enti", "ఏమిటి"),
    "entii": ("enti", "ఏమిటి"),
    "yentii": ("enti", "ఏమిటి"),
    "ento": ("ento", "ఏంటో"),
    "ela": ("ela", "ఎలా"),
    "yela": ("ela", "ఎలా"),
    "elaa": ("ela", "ఎలా"),
    "enduku": ("enduku", "ఎందుకు"),
    "yenduku": ("enduku", "ఎందుకు"),
    "yendhuku": ("enduku", "ఎందుకు"),
    "eppudu": ("eppudu", "ఎప్పుడు"),
    "yeppudu": ("eppudu", "ఎప్పుడు"),
    "evaru": ("evaru", "ఎవరు"),
    "yevaru": ("evaru", "ఎవరు"),
    "enta": ("enta", "ఎంత"),
    "yenta": ("enta", "ఎంత"),

    # Definitional / copula particles
    "ante": ("ante", "అంటే"),
    "antey": ("ante", "అంటే"),
    "antee": ("ante", "అంటే"),
    "anaga": ("anaga", "అనగా"),
    "anagane": ("anaga", "అనగానే"),

    # Verbs / imperatives
    "cheyyi": ("cheyyi", "చెయ్యి"),
    "cheyi": ("cheyyi", "చెయ్యి"),
    "ceyyi": ("cheyyi", "చెయ్యి"),
    "chey": ("cheyyi", "చెయ్యి"),
    "cheyandi": ("cheyyandi", "చెయ్యండి"),
    "cheyyandi": ("cheyyandi", "చెయ్యండి"),
    "cheppu": ("cheppu", "చెప్పు"),
    "ceppu": ("cheppu", "చెప్పు"),
    "cheppandi": ("cheppandi", "చెప్పండి"),
    "cheppava": ("cheppava", "చెప్పవా"),
    "chudu": ("chudu", "చూడు"),
    "choodu": ("chudu", "చూడు"),
    "telusukovali": ("telusukovali", "తెలుసుకోవాలి"),
    "nerchukovali": ("nerchukovali", "నేర్చుకోవాలి"),

    # States & negation
    "ardham": ("ardham", "అర్థం"),
    "artham": ("ardham", "అర్థం"),
    "arthamu": ("ardham", "అర్థం"),
    "ardhamu": ("ardham", "అర్థం"),
    "kaadu": ("kaadu", "కాదు"),
    "kadu": ("kaadu", "కాదు"),
    "ledu": ("ledu", "లేదు"),
    "avuthundi": ("avutundi", "అవుతుంది"),
    "avutundi": ("avutundi", "అవుతుంది"),
    "avtundi": ("avutundi", "అవుతుంది"),
    "untundi": ("untundi", "ఉంటుంది"),
    "undali": ("undali", "ఉండాలి"),
    "undaali": ("undali", "ఉండాలి"),

    # Postpositions & particles
    "yokka": ("yokka", "యొక్క"),
    "gurinchi": ("gurinchi", "గురించి"),
    "kosam": ("kosam", "కోసం"),
    "nundi": ("nundi", "నుండి"),
    "malli": ("malli", "మళ్లీ"),
    "mariyu": ("mariyu", "మరియు"),
    "chala": ("chala", "చాలా"),
    "chaala": ("chala", "చాలా"),
    "kuda": ("kuda", "కూడా"),
    "kooda": ("kuda", "కూడా"),
    "idi": ("idi", "ఇది"),
    "adi": ("adi", "అది"),
    "ivi": ("ivi", "ఇవి"),
    "avi": ("avi", "అవి"),
    "lo": ("lo", "లో"),
    "ki": ("ki", "కి"),
    "ni": ("ni", "ని"),
    "to": ("to", "తో"),
    "ga": ("ga", "గా"),
    "la": ("la", "లా"),
    "kada": ("kada", "కదా"),
    "kadha": ("kada", "కదా"),
}


class RuleBasedRomanizationNormalizer(RomanizationNormalizer):
    """Normalizes Romanized Telugu queries using phonetic rules and variant catalogs."""

    def __init__(self, custom_mapping: dict[str, tuple[str, str]] | None = None) -> None:
        self.mapping = dict(ROMANIZED_CANONICAL_MAP)
        if custom_mapping:
            self.mapping.update(custom_mapping)

    def normalize(self, text: str) -> NormalizationResult:
        """Normalize Romanized Telugu tokens while preserving original text and generating candidate variants."""
        norm_text = unicodedata.normalize("NFC", text or "").strip()
        if not norm_text:
            return NormalizationResult(
                original_text=text,
                canonical_normalized_text="",
                candidate_variants=[],
                is_romanized=False,
                transliterated_telugu=None,
                tokens_normalized={},
            )

        tokens = re.findall(r"[\w\-]+|[^\w\s\-]+|\s+", norm_text, flags=re.UNICODE)
        normalized_tokens: list[str] = []
        telugu_tokens: list[str] = []
        token_transforms: dict[str, str] = {}
        has_romanized = False
        candidates: list[NormalizationCandidate] = []

        for tok in tokens:
            if not tok.strip() or not re.match(r"^[\w\-]+$", tok):
                normalized_tokens.append(tok)
                telugu_tokens.append(tok)
                continue

            low_tok = tok.lower()

            # Rule 1: Direct lexicon match
            if low_tok in self.mapping:
                canon_rom, te_script = self.mapping[low_tok]
                normalized_tokens.append(canon_rom if tok.islower() else canon_rom.capitalize())
                telugu_tokens.append(te_script)
                has_romanized = True
                if low_tok != canon_rom:
                    token_transforms[tok] = canon_rom
                continue

            # Rule 2: Phonetic rewrites
            rewritten = low_tok
            changed = False

            # Initial 'ye' -> 'e' (e.g. yekkada -> ekkada)
            if rewritten.startswith("ye") and len(rewritten) > 2:
                rewritten = "e" + rewritten[2:]
                changed = True
            # Initial 'ya' -> 'a' (e.g. yela -> ela)
            elif rewritten.startswith("ya") and len(rewritten) > 2:
                rewritten = "a" + rewritten[2:]
                changed = True

            # Intervocalic 'th' -> 't' (e.g. avuthundi -> avutundi)
            if "th" in rewritten and not rewritten.startswith("th"):
                # Check if variant exists in lexicon
                alt = rewritten.replace("th", "t")
                if alt in self.mapping:
                    rewritten = alt
                    changed = True

            # Double vowels collapse: 'ee' -> 'i', 'oo' -> 'u', 'aa' -> 'a'
            if "ee" in rewritten:
                alt = rewritten.replace("ee", "i")
                if alt in self.mapping:
                    rewritten = alt
                    changed = True
            if "oo" in rewritten:
                alt = rewritten.replace("oo", "u")
                if alt in self.mapping:
                    rewritten = alt
                    changed = True

            if changed and rewritten in self.mapping:
                canon_rom, te_script = self.mapping[rewritten]
                normalized_tokens.append(canon_rom if tok.islower() else canon_rom.capitalize())
                telugu_tokens.append(te_script)
                has_romanized = True
                token_transforms[tok] = canon_rom
            else:
                normalized_tokens.append(tok)
                telugu_tokens.append(tok)

        canonical_text = "".join(normalized_tokens).strip()
        telugu_text = "".join(telugu_tokens).strip()

        # Generate alternative candidates
        # Candidate 1: Canonical normalized text
        if canonical_text != norm_text:
            candidates.append(
                NormalizationCandidate(
                    variant_text=canonical_text,
                    confidence=0.92,
                    transformation_type="canonical_spelling",
                    explanation=f"Normalized {len(token_transforms)} Romanized Telugu variation(s)",
                )
            )

        # Candidate 2: Transliterated Telugu script representation
        if has_romanized and telugu_text != canonical_text and any(0x0C00 <= ord(c) <= 0x0C7F for c in telugu_text):
            candidates.append(
                NormalizationCandidate(
                    variant_text=telugu_text,
                    confidence=0.85,
                    transformation_type="telugu_script_transliteration",
                    explanation="Mapped Romanized Telugu functional particles to Telugu script",
                )
            )

        # Candidate 3: Phonetic relaxation variant (e.g. 'emiti' variant if 'enti' present)
        if "enti" in canonical_text.lower():
            emiti_variant = re.sub(r"(?i)\benti\b", "emiti", canonical_text)
            candidates.append(
                NormalizationCandidate(
                    variant_text=emiti_variant,
                    confidence=0.80,
                    transformation_type="phonetic_variant",
                    explanation="Formal variant substitution (enti -> emiti)",
                )
            )

        return NormalizationResult(
            original_text=text,
            canonical_normalized_text=canonical_text,
            candidate_variants=candidates,
            is_romanized=has_romanized,
            transliterated_telugu=telugu_text if has_romanized else None,
            tokens_normalized=token_transforms,
        )
