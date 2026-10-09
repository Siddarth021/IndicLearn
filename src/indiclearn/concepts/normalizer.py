"""Concept normalization and deterministic ID generation (Phase 3).

Standardizes academic expressions, manages acronyms and aliases,
and generates deterministic IDs scoped by subject.
"""

from __future__ import annotations

import re
import unicodedata


def slugify(text: str) -> str:
    """Generate a clean URL/ID slug from a text string.

    Preserves alphanumeric characters across Unicode scripts (Latin, Telugu, etc.)
    and replaces non-alphanumeric separators with underscores.
    """
    if not text:
        return ""
    # Normalize unicode
    text = unicodedata.normalize("NFC", text).strip().lower()
    # Replace whitespace and hyphens with underscores
    slug = re.sub(r"[\s\-\.]+", "_", text)
    # Remove characters that are neither word characters nor underscores
    slug = re.sub(r"[^\w_]+", "", slug, flags=re.UNICODE)
    # Collapse multiple consecutive underscores
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug


def generate_concept_id(subject_id: str, canonical_name: str) -> str:
    """Generate a deterministic concept ID from subject and canonical name."""
    sub_slug = slugify(subject_id)
    cpt_slug = slugify(canonical_name)
    return f"cpt_{sub_slug}_{cpt_slug}"


def generate_relation_id(
    subject_id: str,
    source_concept_id: str,
    relation_type: str,
    target_concept_id: str,
) -> str:
    """Generate a deterministic relation ID from source, type, and target."""
    sub_slug = slugify(subject_id)
    s_slug = source_concept_id.replace(f"cpt_{sub_slug}_", "").replace("cpt_", "")
    t_slug = target_concept_id.replace(f"cpt_{sub_slug}_", "").replace("cpt_", "")
    rel_type_slug = relation_type.lower()
    return f"rel_{sub_slug}_{s_slug}_{rel_type_slug}_{t_slug}"


class ConceptNormalizer:
    """Normalizes candidate terms and resolves acronyms to canonical concept representations."""

    def __init__(self, subject_id: str = "signals_and_systems") -> None:
        self.subject_id = subject_id
        # Canonical dictionary of domain acronyms and abbreviations
        self._acronym_map: dict[str, str] = {
            "lti": "LTI System",
            "lti system": "LTI System",
            "lti systems": "LTI System",
            "linear time invariant system": "LTI System",
            "linear time-invariant system": "LTI System",
            "bibo": "BIBO Stability",
            "bibo stable": "BIBO Stability",
            "bibo stability": "BIBO Stability",
            "ctft": "Continuous-Time Fourier Transform",
            "dtft": "Discrete-Time Fourier Transform",
            "roc": "Region of Convergence",
            "the roc": "Region of Convergence",
            "fft": "Fast Fourier Transform",
            "dft": "Discrete Fourier Transform",
            "time-invariance": "Time Invariance",
            "time invariance": "Time Invariance",
            "nyquist-shannon sampling theorem": "Sampling Theorem",
            "sampling theorem": "Sampling Theorem",
            "nyquist rate": "Nyquist Rate",
            "energy signals": "Energy Signal",
            "power signals": "Power Signal",
            "bilateral laplace transform": "Laplace Transform",
            "the bilateral laplace transform": "Laplace Transform",
            "fourier transform": "Continuous-Time Fourier Transform",
            "continuous-time fourier transform": "Continuous-Time Fourier Transform",
            "the fourier transform": "Continuous-Time Fourier Transform",
            "the z-transform": "Z-Transform",
            "z-transform": "Z-Transform",
            "z transform": "Z-Transform",
            "impulse response": "Impulse Response",
            "unit impulse response": "Impulse Response",
        }

    def canonicalize(self, raw_term: str) -> str:
        """Derive standard display canonical name from raw surface expression."""
        cleaned = unicodedata.normalize("NFC", raw_term).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)

        # Strip leading articles and introductory descriptors
        cleaned = re.sub(r"^(?:the|a|an|this|that|these)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^(?:properties\s+of|properties\s+and|characteristics\s+of)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+(?:properties|overview|introduction)$", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()

        key = cleaned.lower()

        # Reject noise fragments ending or starting with conjunctions, pronouns, or prepositions
        stopwords = {
            "and", "or", "in", "of", "to", "for", "a", "an", "the", "s", "is",
            "by", "with", "two", "core", "which", "as", "that", "this", "from",
        }
        words_test = key.split()
        if not words_test or words_test[-1] in stopwords or words_test[0] in stopwords:
            return ""

        # Reject overly generic terms that aren't specific concepts
        if key in {"system", "properties", "signals", "signal", "operation", "criterion", "condition", "definition", "domain", "or"}:
            return ""

        # Check known acronym / synonym mapping
        if key in self._acronym_map:
            return self._acronym_map[key]

        # Standardize casing for English phrases (Title Case for acronyms and major terms)
        def title_token(w: str) -> str:
            if w.isupper() and len(w) <= 5:
                return w
            if "-" in w:
                return "-".join(p.capitalize() for p in w.split("-"))
            return w.capitalize()

        words = cleaned.split(" ")
        capitalized_words: list[str] = []
        for w in words:
            if w.isupper() and len(w) <= 5:
                capitalized_words.append(w)
            elif w.lower() in {"and", "or", "of", "in", "to", "for", "the", "a", "an"}:
                capitalized_words.append(w.lower())
            else:
                capitalized_words.append(title_token(w))

        # Ensure first word is always capitalized
        if capitalized_words:
            capitalized_words[0] = title_token(capitalized_words[0])

        res = " ".join(capitalized_words)
        return res if len(res) >= 3 else ""

    def extract_aliases(self, canonical_name: str, raw_occurrences: list[str] | None = None) -> list[str]:
        """Generate deduplicated set of aliases for a canonical concept."""
        aliases: set[str] = set()

        # Lowercase version
        aliases.add(canonical_name.lower())

        # Acronym expansions and inverse lookups
        for acr, canon in self._acronym_map.items():
            if canon.lower() == canonical_name.lower():
                aliases.add(acr)
                aliases.add(acr.upper())

        # Hyphenated vs unhyphenated variations
        if "-" in canonical_name:
            aliases.add(canonical_name.replace("-", " "))
            aliases.add(canonical_name.replace("-", " ").lower())
        else:
            # Check for words like "time invariant" -> "time-invariant"
            if "time invariant" in canonical_name.lower():
                aliases.add(canonical_name.lower().replace("time invariant", "time-invariant"))

        # Add raw observed occurrences
        if raw_occurrences:
            for raw in raw_occurrences:
                r_clean = raw.strip()
                if r_clean and r_clean != canonical_name:
                    aliases.add(r_clean)
                    aliases.add(r_clean.lower())

        # Do not include the canonical name itself in aliases
        aliases.discard(canonical_name)
        return sorted(aliases)

    get_curated_aliases = extract_aliases
