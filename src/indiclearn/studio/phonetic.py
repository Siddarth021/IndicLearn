"""Cross-script *phonetic skeleton* matching for transliterated technical terms.

Indic-medium engineering text is full of English terms written in native script
(``కన్వల్యూషన్`` / ``कन्वोल्यूशन`` for *convolution*), and students type the same terms in Latin
script.  A rough consonant skeleton lets us link all of these spellings without any model::

    convolution  -> k n v l s n
    కన్వల్యూషన్   -> k n v l s n
    कन्वोल्यूशन    -> k n v l s n

Voiced/unvoiced pairs are merged (t/d, k/g, p/b, s/z) because Telugu and Hindi spellings of
English loan-words are not consistent about them.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

_TE = {
    "క": "k", "ఖ": "k", "గ": "k", "ఘ": "k", "ఙ": "n", "చ": "c", "ఛ": "c", "జ": "c", "ఝ": "c", "ఞ": "n",
    "ట": "t", "ఠ": "t", "డ": "t", "ఢ": "t", "ణ": "n", "త": "t", "థ": "t", "ద": "t", "ధ": "t", "న": "n",
    "ప": "p", "ఫ": "p", "బ": "p", "భ": "p", "మ": "m", "య": "", "ర": "r", "ఱ": "r", "ల": "l", "ళ": "l",
    "వ": "v", "శ": "s", "ష": "s", "స": "s", "హ": "", "ం": "n", "ః": "",
}
_DV = {
    "क": "k", "ख": "k", "ग": "k", "घ": "k", "ङ": "n", "च": "c", "छ": "c", "ज": "c", "झ": "c", "ञ": "n",
    "ट": "t", "ठ": "t", "ड": "t", "ढ": "t", "ण": "n", "त": "t", "थ": "t", "द": "t", "ध": "t", "न": "n",
    "प": "p", "फ": "p", "ब": "p", "भ": "p", "म": "m", "य": "", "र": "r", "ल": "l", "ळ": "l", "व": "v",
    "श": "s", "ष": "s", "स": "s", "ह": "", "ं": "n", "ँ": "n", "ः": "", "ज़": "s", "फ़": "p", "ड़": "r",
}
_MAP = {**_TE, **_DV}
_MERGE = str.maketrans({"m": "n", "d": "t", "g": "k", "b": "p", "z": "s", "w": "v", "j": "c", "q": "k", "f": "p", "x": "k"})


def _latin_skeleton(w: str) -> str:
    w = w.lower()
    w = re.sub(r"[^a-z]", "", w)
    w = re.sub(r"(?:tion|sion|cian|tian)", "shn", w)
    w = w.replace("ph", "f").replace("ch", "c").replace("sh", "s").replace("th", "t").replace("kh", "k").replace("gh", "g")
    w = w.replace("ck", "k").replace("qu", "k").replace("x", "ks")
    w = re.sub(r"c(?=[eiy])", "s", w)
    w = w.replace("c", "k")
    # undo: "ch" was mapped to 'c' above and then to 'k'; keep it distinct
    return w


def skeleton(text: str) -> str:
    """Consonant skeleton for a word or short phrase (Latin, Telugu or Devanagari)."""
    out: list[str] = []
    for word in re.split(r"[\s\-_/]+", text.strip().lower()):
        if not word:
            continue
        buf: list[str] = []
        latin: list[str] = []
        for ch in word:
            if ch in _MAP:
                if latin:
                    buf.append(_latin_skeleton("".join(latin)))
                    latin = []
                buf.append(_MAP[ch])
            elif "a" <= ch <= "z":
                latin.append(ch)
            else:
                if latin:
                    buf.append(_latin_skeleton("".join(latin)))
                    latin = []
        if latin:
            buf.append(_latin_skeleton("".join(latin)))
        s = "".join(buf)
        s = re.sub(r"[aeiouyh]", "", s)
        s = s.translate(_MERGE)
        s = re.sub(r"(.)\1+", r"\1", s)
        out.append(s)
    return "".join(out)


def phonetic_similarity(a: str, b: str) -> float:
    sa, sb = skeleton(a), skeleton(b)
    if len(sa) < 3 or len(sb) < 3:
        return 0.0
    return SequenceMatcher(None, sa, sb).ratio()
