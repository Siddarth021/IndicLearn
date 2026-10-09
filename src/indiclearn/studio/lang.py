"""Language / script handling for English, Telugu, Hindi and Sanskrit (+ Romanized forms).

Design notes
------------
* Python's ``\\w`` does NOT match Indic combining marks (vowel signs, virama), so a naive
  ``re.findall(r"\\w+")`` shreds ``కన్వల్యూషన్`` into four fragments.  Everything in this
  module therefore uses an explicit token pattern that includes the Telugu (U+0C00-0C7F) and
  Devanagari (U+0900-097F) blocks as well as ZWJ / ZWNJ.
* Detection is deterministic and lexicon + script based (no network, no model download), so the
  bot can always decide which language it is talking to.  Hindi vs. Sanskrit share the
  Devanagari script and are separated with marker words and Sanskrit morphology
  (visarga, ``-म्``, ``-स्य``, ``-ेन`` endings...).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

SUPPORTED_LANGS = ("en", "te", "hi", "sa")
LANG_NAMES = {"en": "English", "te": "తెలుగు", "hi": "हिन्दी", "sa": "संस्कृतम्"}
LANG_NAMES_EN = {"en": "English", "te": "Telugu", "hi": "Hindi", "sa": "Sanskrit"}

_TE = r"ఀ-౿"
_DEVA = r"ऀ-ॿ꣠-ꣿ᳐-᳿"
_ZW = r"‌‍"
_LETTERS = rf"A-Za-z0-9{_TE}{_DEVA}À-ɏ"

# A word: letters/marks plus internal hyphen/apostrophe/dot-joined parts (Z-transform, don't, e.g.)
WORD_RE = re.compile(rf"[{_LETTERS}][{_LETTERS}{_ZW}]*(?:[-'’][{_LETTERS}][{_LETTERS}{_ZW}]*)*")

_ZW_RE = re.compile(rf"[{_ZW}]")
_SPACE_RE = re.compile(r"\s+")
_TELUGU_RE = re.compile(rf"[{_TE}]")
_DEVA_RE = re.compile(rf"[{_DEVA}]")
_LATIN_RE = re.compile(r"[A-Za-z]")


# ---------------------------------------------------------------------------
# Stop-words (small, hand-curated; used for keyphrase boundaries and retrieval hygiene)
# ---------------------------------------------------------------------------
STOP_EN = set(
    ["a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "to", "in", "on", "at", "by", "for", "with", "without", "from", "into", "onto", "over", "under", "between", "among", "through", "during", "before", "after", "above", "below", "up", "down", "out", "off", "again", "further", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will", "just", "should", "now", "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "doing", "have", "has", "had", "having", "it", "its", "itself", "this", "that", "these", "those", "which", "who", "whom", "whose", "what", "i", "me", "my", "we", "our", "you", "your", "he", "him", "his", "she", "her", "they", "them", "their", "as", "also", "may", "might", "must", "shall", "would", "could", "one", "two", "three", "however", "thus", "hence", "therefore", "moreover", "whereas", "while", "although", "though", "because", "since", "per", "via", "etc", "eg", "ie", "vs", "using", "used", "use", "uses", "based", "given", "shown", "show", "shows", "see", "seen", "fig", "figure", "table", "section", "eq", "equation", "paper", "work", "works", "result", "results", "proposed", "approach", "approaches", "many", "much", "well", "often", "different", "following", "first", "second", "third", "new", "thus", "within", "across", "upon", "whether", "either", "neither", "yet", "still", "even", "ever", "never", "always", "sometimes", "can't", "cannot", "don't", "doesn't", "isn't", "aren't"]
)
STOP_TE = set(
    ["ఈ", "ఆ", "ఒక", "మరియు", "లేదా", "కూడా", "అది", "ఇది", "అవి", "ఇవి", "ఉంది", "ఉన్నాయి", "ఉంటుంది", "ఉంటాయి", "అని", "అంటే", "అనగా", "అనేది", "అనేవి", "కాని", "కానీ", "అయితే", "ఎందుకంటే", "కాబట్టి", "తో", "లో", "ను", "కు", "కి", "నుండి", "యొక్క", "గురించి", "వల్ల", "చేయడం", "చేసే", "చేసిన", "అయ్యే", "అవుతుంది", "అవుతాయి", "ఏ", "ఏమి", "ఎలా", "ఎందుకు", "వంటి", "మరొక", "ఇలా", "అలా", "అన్ని", "ప్రతి", "కొన్ని", "చాలా", "మాత్రమే", "లేదు", "కాదు", "ఉన్న", "ఉండే", "కలిగి", "ద్వారా", "లోని", "పై", "క్రింద", "మధ్య", "తర్వాత", "ముందు", "కోసం", "వద్ద", "అలాగే", "ఇక్కడ", "అక్కడ", "దీని", "దాని", "వీటి", "వాటి", "నేను", "మీరు", "మనం", "మేము", "వారు", "అంటారు", "చెప్పండి", "చేయండి", "ఏమిటి", "అర్థం", "పూర్తిగా", "నిర్వచించబడిన", "నిర్వచించబడినది", "మార్చే", "ఇచ్చే", "మారే", "తెలిపే", "ఇచ్చినప్పుడు", "వస్తుంది", "వస్తే", "చేస్తే", "వస్తాయి", "అయ్యే", "ఇస్తే", "ఉన్నప్పుడు", "ఏర్పడుతుంది", "కలిపి", "కొలిచి", "అంతే", "రెండు", "ఒకటి"]
)
STOP_HI = set(
    ["का", "की", "के", "को", "में", "से", "पर", "और", "या", "भी", "तो", "है", "हैं", "था", "थी", "थे", "हो", "होता", "होती", "होते", "यह", "वह", "ये", "वे", "इस", "उस", "इन", "उन", "एक", "कि", "जो", "जिस", "जिन", "कोई", "कुछ", "सब", "सभी", "बहुत", "नहीं", "न", "ही", "लिए", "साथ", "द्वारा", "कर", "करना", "करते", "करता", "करती", "किया", "गया", "गई", "गए", "जाता", "जाती", "जाते", "सकता", "सकती", "सकते", "चाहिए", "तथा", "एवं", "अथवा", "अर्थात", "जैसे", "जब", "तब", "तक", "अगर", "यदि", "लेकिन", "परंतु", "क्योंकि", "इसलिए", "क्या", "कैसे", "क्यों", "कब", "कहाँ", "कौन", "अपने", "अपनी", "अपना", "हम", "आप", "तुम", "मैं", "उसके", "इसके", "उनके", "उसकी", "इसकी", "रहा", "रही", "रहे", "वाला", "वाली", "वाले", "बारे", "ऊपर", "नीचे", "बाद", "पहले", "अंदर", "बीच", "कहते", "कहलाता", "कहलाती", "कहलाते", "हुआ", "हुई", "हुए", "पूरी", "तरह", "बदलती", "बदलता", "मिलाकर", "देती", "देता", "मिलता", "मिलती", "जाए", "किया", "जा", "दिया", "दी", "दिए", "केवल", "कुछ", "मान", "मापकर", "बनाया", "उत्पन्न", "बन"]
)
STOP_SA = set(
    ["च", "वा", "अपि", "तु", "एव", "इति", "अस्ति", "सन्ति", "भवति", "भवन्ति", "स", "सः", "सा", "तत्", "ते", "ताः", "तानि", "एषः", "एषा", "एतत्", "अयम्", "इयम्", "इदम्", "यः", "या", "यत्", "ये", "याः", "यानि", "कः", "का", "किम्", "के", "काः", "कानि", "तथा", "यथा", "अत्र", "तत्र", "कुत्र", "यत्र", "सर्वे", "सर्वम्", "न", "नास्ति", "हि", "खलु", "नाम", "अथ", "ततः", "यदा", "तदा", "कदा", "अहम्", "त्वम्", "वयम्", "यूयम्", "मम", "तव", "अस्य", "तस्य", "एकः", "एका", "एकम्", "उच्यते", "कथम्", "कीदृशम्"]
)
STOPWORDS = {"en": STOP_EN, "te": STOP_TE, "hi": STOP_HI, "sa": STOP_SA}
ALL_STOP = STOP_EN | STOP_TE | STOP_HI | STOP_SA

# ---------------------------------------------------------------------------
# Romanized markers (Latin script typed Telugu / Hindi)
# ---------------------------------------------------------------------------
ROMAN_TE = set(
    ["ante", "antey", "enti", "yenti", "emiti", "ela", "enduku", "cheppu", "ceppu", "cheyyi", "ceyyi", "chey", "ardham", "artham", "kaadu", "kadu", "avuthundi", "avutundi", "avtundi", "untundi", "yokka", "lo", "ki", "ni", "nundi", "kosam", "gurinchi", "chala", "malli", "mariyu", "kuda", "idi", "adi", "ivi", "avi", "cheppandi", "telusukovali", "nerchukovali", "eppudu", "akkada", "ikkada", "anaga", "anagane", "evaru", "enta", "ento", "chudu", "cheyandi", "cheppava", "undali", "undadu", "ledu", "kavali", "teliyadu", "kada", "baga", "baaga", "ivvandi", "ippudu", "taruvatha"]
)
ROMAN_HI = set(
    ["kya", "hai", "hain", "kaise", "kyun", "kyu", "kab", "kahan", "kaun", "matlab", "samjhao", "samjhaiye", "batao", "bataiye", "mujhe", "mera", "meri", "nahi", "nahin", "aur", "ka", "ke", "ki", "mein", "se", "par", "ko", "hota", "hoti", "hote", "kaam", "karta", "karte", "karna", "hum", "aap", "tum", "yeh", "woh", "isme", "isse", "iska", "uska", "liye", "wala", "wali", "bhi", "toh", "lekin", "agar", "jab", "tab", "tak", "sirf", "bahut", "thoda", "zyada", "ho", "sakta"]
)
# Words that are fine in both: don't count "ki", "ka", "ke", "se", "ko", "lo", "ni" alone as evidence
_ROMAN_AMBIGUOUS = {"ki", "ka", "ke", "se", "ko", "lo", "ni", "par", "ho", "hum", "tum", "kya", "ante"}

# ---------------------------------------------------------------------------
# Hindi vs Sanskrit (both Devanagari)
# ---------------------------------------------------------------------------
HI_MARKERS = set(
    ["है", "हैं", "था", "थी", "थे", "का", "की", "के", "को", "में", "से", "पर", "और", "यह", "वह", "ये", "वे", "क्या", "कैसे", "क्यों", "नहीं", "कि", "जो", "तो", "भी", "हम", "आप", "मैं", "तुम", "होता", "होती", "होते", "करना", "करते", "जाता", "सकते", "चाहिए", "लिए", "साथ", "बारे", "द्वारा", "कहते", "कहलाता", "कहलाती", "कहलाते", "होता"]
)
SA_MARKERS = set(
    ["अस्ति", "सन्ति", "किम्", "कः", "का", "कथम्", "कुत्र", "कदा", "तत्", "एतत्", "इदम्", "इति", "च", "अपि", "तथा", "यत्", "यदा", "तदा", "भवति", "भवन्ति", "अत्र", "तत्र", "एव", "नाम", "सः", "सा", "अहम्", "त्वम्", "वयम्", "यूयम्", "कृत्वा", "गच्छति", "उच्यते", "उच्यन्ते", "कथ्यते", "वर्तते", "वर्तन्ते", "भवेत्", "स्यात्", "नास्ति", "अस्य", "तस्य", "एषः", "एषा", "अयम्", "इयम्", "सर्वम्", "सर्वे", "यः", "या", "ये"]
)
_SA_SUFFIXES = ("ः", "म्", "ाम्", "स्य", "ेन", "ाय", "ात्", "ेषु", "ानाम्", "ाणि", "न्ति", "ाः", "ान्", "ेभ्यः")


@dataclass
class LangInfo:
    """Outcome of :func:`detect_language`."""

    code: str  # en | te | hi | sa | te-Latn | hi-Latn | unknown
    script: str  # Latin | Telugu | Devanagari | Mixed | Unknown
    confidence: float
    code_mixed: bool = False
    ratios: dict[str, float] = field(default_factory=dict)

    @property
    def base(self) -> str:
        """Language to *reply* in when no explicit preference exists (maps Romanized -> native)."""
        return {"te-Latn": "te", "hi-Latn": "hi"}.get(self.code, self.code if self.code in SUPPORTED_LANGS else "en")


def normalize(text: str) -> str:
    """NFC-normalise and collapse whitespace (keeps ZWJ/ZWNJ, which are meaningful in Indic)."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace(" ", " ").replace("­", "")
    return _SPACE_RE.sub(" ", text).strip()


def match_key(text: str) -> str:
    """Aggressive key for matching/dedup: NFC, lowercase, no ZW chars, no punctuation spacing."""
    text = unicodedata.normalize("NFC", text or "").lower()
    text = _ZW_RE.sub("", text)
    return _SPACE_RE.sub(" ", text).strip(" \t\n.,;:!?()[]{}\"'“”‘’")


def script_ratios(text: str) -> dict[str, float]:
    te = len(_TELUGU_RE.findall(text))
    dv = len(_DEVA_RE.findall(text))
    la = len(_LATIN_RE.findall(text))
    tot = te + dv + la
    if not tot:
        return {"Latin": 0.0, "Telugu": 0.0, "Devanagari": 0.0}
    return {"Latin": la / tot, "Telugu": te / tot, "Devanagari": dv / tot}


def tokenize(text: str) -> list[str]:
    """Unicode-safe word tokens (case preserved, ZW chars stripped from token ends)."""
    return [_ZW_RE.sub("", w) for w in WORD_RE.findall(text or "")]


def tokenize_lower(text: str) -> list[str]:
    return [t.lower() for t in tokenize(text)]


def detect_language(text: str) -> LangInfo:
    """Detect en / te / hi / sa / te-Latn / hi-Latn with a code-mix flag."""
    text = normalize(text)
    if not text:
        return LangInfo("unknown", "Unknown", 0.0)
    r = script_ratios(text)
    toks = tokenize_lower(text)
    n = max(len(toks), 1)

    indic_ratio = r["Telugu"] + r["Devanagari"]
    code_mixed = r["Latin"] > 0.12 and indic_ratio > 0.12
    if r["Telugu"] > 0.5 or (code_mixed and r["Telugu"] > 0.12 and r["Telugu"] >= r["Devanagari"]):
        return LangInfo("te", "Mixed" if code_mixed else "Telugu", min(1.0, 0.6 + r["Telugu"] * 0.4), code_mixed, r)
    if r["Devanagari"] > 0.5 or (r["Devanagari"] > 0.12 and code_mixed):
        dev_tokens = [t for t in toks if _DEVA_RE.search(t)] or toks
        sa_hits = sum(1 for t in dev_tokens if t in SA_MARKERS)
        hi_hits = sum(1 for t in dev_tokens if t in HI_MARKERS)
        sa_morph = sum(1 for t in dev_tokens if len(t) > 3 and t.endswith(_SA_SUFFIXES))
        sa_score = 2.0 * sa_hits + 0.5 * sa_morph
        hi_score = 2.0 * hi_hits
        is_sa = sa_score > hi_score and (sa_hits >= 1 or sa_morph / max(len(dev_tokens), 1) > 0.4)
        code = "sa" if is_sa else "hi"
        conf = 0.6 + 0.3 * min(1.0, abs(sa_score - hi_score) / max(len(dev_tokens), 1) + 0.2)
        return LangInfo(code, "Mixed" if code_mixed else "Devanagari", min(conf, 0.95), code_mixed, r)

    # Latin: English vs Romanized Telugu / Hindi
    te_hits = [t for t in toks if t in ROMAN_TE]
    hi_hits_l = [t for t in toks if t in ROMAN_HI]
    te_strong = sum(1 for t in te_hits if t not in _ROMAN_AMBIGUOUS)
    hi_strong = sum(1 for t in hi_hits_l if t not in _ROMAN_AMBIGUOUS)
    if (te_strong >= 1 and len(te_hits) >= 2) or te_strong >= 2:
        if len(te_hits) >= hi_strong + 1:
            return LangInfo("te-Latn", "Latin", min(0.95, 0.5 + 0.15 * len(te_hits)), False, r)
    if (hi_strong >= 1 and len(hi_hits_l) >= 2) or hi_strong >= 2:
        return LangInfo("hi-Latn", "Latin", min(0.95, 0.5 + 0.15 * len(hi_hits_l)), False, r)
    conf = 0.9 if sum(1 for t in toks if t in STOP_EN) / n > 0.15 else 0.7
    return LangInfo("en", "Latin", conf, False, r)


def detect_document_language(text: str, sample: int = 20000) -> LangInfo:
    """Document-level detection: majority language over chunks of the text."""
    text = text or ""
    if len(text) > sample:
        step = len(text) // 8
        parts = [text[i : i + sample // 8] for i in range(0, len(text), step)][:8]
    else:
        parts = [text]
    votes: dict[str, float] = {}
    last: LangInfo | None = None
    for p in parts:
        info = detect_language(p)
        last = info
        votes[info.base] = votes.get(info.base, 0.0) + len(p)
    if not votes:
        return LangInfo("unknown", "Unknown", 0.0)
    best = max(votes, key=lambda k: votes[k])
    base = detect_language(text[:sample]) if best == "en" else last
    return LangInfo(best, base.script if base else "Unknown", 0.9, bool(base and base.code_mixed), base.ratios if base else {})


# ---------------------------------------------------------------------------
# Light stemming (retrieval + concept matching).  Deliberately conservative.
# ---------------------------------------------------------------------------
_TE_SUFFIXES = sorted(
    ["యొక్క", "నుండి", "గురించి", "లోని", "తోను", "ల్లో", "లకు", "ులు", "లు", "ను", "ని", "కు", "కి", "లో",
     "తో", "ది", "ము", "ం", "ల", "ా", "ి", "ు", "ే"],
    key=len,
    reverse=True,
)
_HI_SUFFIXES = sorted(["ियों", "ियाँ", "ाओं", "ाएँ", "ों", "ें", "ीय", "ता", "ती", "ते", "ाँ", "ों"], key=len, reverse=True)
_SA_SUFFIXES_STEM = sorted(
    ["ानाम्", "ेभ्यः", "ेषु", "ाणि", "ाभ्याम्", "स्य", "ेन", "ाय", "ात्", "ाः", "ान्", "ाम्", "म्", "ः"],
    key=len,
    reverse=True,
)


def stem(token: str, lang: str | None = None) -> str:
    """Strip a single common inflectional suffix.  Never shortens below 3 characters."""
    t = _ZW_RE.sub("", token.lower())
    if not t:
        return t
    if _TELUGU_RE.search(t):
        for suf in _TE_SUFFIXES:
            if t.endswith(suf) and len(t) - len(suf) >= 3:
                return t[: -len(suf)]
        return t
    if _DEVA_RE.search(t):
        sufs = _SA_SUFFIXES_STEM if lang == "sa" else _HI_SUFFIXES + _SA_SUFFIXES_STEM[-3:]
        for suf in sufs:
            if t.endswith(suf) and len(t) - len(suf) >= 3:
                return t[: -len(suf)]
        return t
    return _stem_en(t)


def _stem_en(t: str) -> str:
    """Plural-only English normaliser (high precision): signals->signal, processes->process."""
    if len(t) <= 3:
        return t
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("sses"):
        return t[:-2]
    if t.endswith(("xes", "ches", "shes", "zes")) and len(t) > 4:
        return t[:-2]
    if t.endswith("s") and not t.endswith(("ss", "us", "is", "ics")) and len(t) > 3:
        return t[:-1]
    return t


def index_tokens(text: str, lang: str | None = None) -> list[str]:
    """Tokens used for lexical retrieval: lowercase words (+ stems), stop-words removed."""
    out: list[str] = []
    for w in tokenize_lower(text):
        if w in ALL_STOP or len(w) < 2 and not w.isdigit():
            continue
        out.append(w)
        s = stem(w, lang)
        if s != w:
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------
_ABBREV = re.compile(
    r"\b(e\.g|i\.e|etc|vs|fig|figs|eq|eqs|sec|al|no|dr|mr|mrs|ms|prof|approx|cf|resp|ref|vol|pp)\.", re.IGNORECASE
)
_DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")
_INITIAL = re.compile(r"\b([A-Z])\.(?=\s?[A-Z])")


def split_sentences(text: str) -> list[str]:
    """Split on . ! ? । ॥ while protecting abbreviations, decimals and initials."""
    text = normalize(text)
    if not text:
        return []
    protected = _ABBREV.sub(lambda m: m.group(0).replace(".", "§"), text)
    protected = _DECIMAL.sub("§", protected)
    protected = _INITIAL.sub(lambda m: m.group(1) + "§", protected)
    parts = re.split(r"(?<=[.!?।॥])\s+(?=[\"'“‘(\[]?[A-Z" + _TE + _DEVA + r"0-9])|(?<=[।॥])\s*", protected)
    return [p.replace("§", ".").strip() for p in parts if p and p.strip()]


def is_indic(text: str) -> bool:
    r = script_ratios(text)
    return (r["Telugu"] + r["Devanagari"]) > 0.3
