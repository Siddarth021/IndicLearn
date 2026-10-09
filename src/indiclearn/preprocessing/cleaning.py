"""Text cleaning for extracted document text.

Purpose: make extracted text consistent without destroying content.
Input: raw string. Output: cleaned string.
Algorithm: Unicode normalization (NFC by default), strip NUL/BOM/soft-hyphen, de-hyphenate
line-break splits of ASCII words, collapse runs of spaces/tabs and 3+ newlines.
Assumptions/limitations:
  * ZWJ (U+200D) and ZWNJ (U+200C) are PRESERVED: they are meaningful in Indic scripts
    (conjunct control) and must not be stripped as "invisible junk".
  * De-hyphenation only joins ``word-\\nword`` when the continuation starts with a lowercase
    ASCII letter; it can wrongly merge genuinely hyphenated compounds split at a line end.
  * Does not repair PDF-extraction artefacts specific to Telugu fonts (legacy-encoded PDFs);
    that is a known risk tracked in docs/TODO.md.
Complexity: O(n).
"""

from __future__ import annotations

import re
import unicodedata

_STRIP_CHARS = dict.fromkeys(map(ord, "\x00\ufeff\u00ad"), None)
_HYPHEN_BREAK = re.compile(r"([A-Za-z])-\n([a-z])")
_SPACES = re.compile(r"[ \t\f\v\u00a0]+")
_BLANKS = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r" +\n")


def clean_text(text: str, unicode_form: str = "NFC", join_hyphenated_lines: bool = True) -> str:
    text = text.translate(_STRIP_CHARS)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if unicode_form:
        text = unicodedata.normalize(unicode_form, text)
    if join_hyphenated_lines:
        text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _SPACES.sub(" ", text)
    text = _TRAILING_WS.sub("\n", text)
    text = _BLANKS.sub("\n\n", text)
    return text.strip()
