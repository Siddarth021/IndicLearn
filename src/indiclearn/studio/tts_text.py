"""Make text safe to read aloud with the offline voices (MMS-TTS).

Two jobs:

1. ``english_for_tts`` - the English voice is character-based, so it spells out or skips acronyms, digits and
   symbols. Spell acronyms letter by letter, turn numbers and symbols into words, respell the technical words
   it gets wrong.
2. ``to_native_script`` - a Telugu/Hindi voice cannot pronounce Latin letters, so a reply is normally read by
   TWO voices (one Telugu, one English) and the voice changes in the middle of the sentence. Writing the English
   words in the native script lets ONE voice read the whole reply, the way Telugu lecturers actually say them
   ("ఆపరేటింగ్ సిస్టమ్").  Approximate by design: a small dictionary of common CS words + letter rules.
"""
from __future__ import annotations

import re

# ----------------------------------------------------------------------------------------- English cleanup
_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_words(n: int) -> str:
    if n < 0:
        return "minus " + number_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + number_words(n % 100))
    if n < 1_000_000:
        return number_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + number_words(n % 1000))
    return " ".join(_ONES[int(c)] for c in str(n))  # very long numbers: digit by digit


# words the character-based voice mispronounces -> a respelling it reads correctly
RESPELL = {
    "mutex": "mew tex", "semaphore": "sem a four", "semaphores": "sem a fours", "thrash": "thrash", "thrashing": "thrashing",
    "belady": "bell a dee", "pthread": "p thread", "pthreads": "p threads", "linux": "lih nux", "unix": "you nix", "ubuntu": "oo boon too",
    "kernel": "kernel", "daemon": "dee mon", "cache": "cash", "caches": "cashes", "queue": "kew", "queues": "kews", "heuristic": "hew ris tick",
    "paging": "pay jing", "segmentation": "seg men tay shun", "dijkstra": "dike stra", "banker's": "bankers", "tlb": "t l b",
    "fifo": "fye foe", "lifo": "lye foe", "sjf": "s j f", "srtf": "s r t f", "lru": "l r u", "mru": "m r u", "lfu": "l f u",
    "scan": "scan", "cscan": "c scan", "ipc": "i p c", "api": "a p i", "gui": "g u i", "cli": "c l i", "bios": "bye ose", "posix": "pozz icks",
    "sql": "s q l", "nosql": "no s q l", "json": "jay son", "xml": "x m l", "ascii": "ass key", "wifi": "wy fy", "iot": "i o t",
    "boolean": "boo lee un", "modulo": "mod you low", "mod": "mod", "iff": "if and only if", "dsma": "d s m a", "gate": "gate",
    "eigenvalue": "eye gen value", "eigenvalues": "eye gen values", "eigenvector": "eye gen vector", "eigenvectors": "eye gen vectors",
    "gaussian": "gow see an", "euler": "oiler", "euler's": "oilers", "hamiltonian": "ham il toe nee an", "lagrange": "la grahnj",
    "bijective": "by jective", "injective": "in jective", "surjective": "sir jective", "isomorphism": "eye so more fizz um",
    "iaas": "i a a s", "os": "o s", "cpu": "c p u", "gpu": "g p u", "ram": "ram", "rom": "rom", "io": "i o", "i/o": "i o",
    "e.g.": "for example", "i.e.": "that is", "etc.": "etcetera", "vs.": "versus", "vs": "versus", "approx.": "approximately",
}
_SYMBOLS = [
    ("<=", " is less than or equal to "), (">=", " is greater than or equal to "), ("!=", " is not equal to "), ("==", " equals "),
    ("->", " goes to "), ("=>", " implies "), ("<->", " if and only if "), ("<-", " from "), ("&&", " and "), ("||", " or "),
    ("→", " goes to "), ("⇒", " implies "), ("↔", " if and only if "), ("⇔", " if and only if "), ("∧", " and "), ("∨", " or "), ("¬", " not "),
    ("∀", " for all "), ("∃", " there exists "), ("∈", " belongs to "), ("∉", " does not belong to "), ("⊆", " is a subset of "), ("⊂", " is a subset of "),
    ("∪", " union "), ("∩", " intersection "), ("∅", " empty set "), ("≤", " is less than or equal to "), ("≥", " is greater than or equal to "),
    ("≠", " is not equal to "), ("≈", " is approximately "), ("×", " times "), ("·", " times "), ("÷", " divided by "), ("±", " plus or minus "),
    ("∞", " infinity "), ("√", " square root of "), ("%", " percent "), ("+", " plus "), ("=", " equals "), ("<", " less than "), (">", " greater than "),
    ("&", " and "), ("@", " at "), ("#", " number "), ("/", " or "), ("^", " to the power "), ("~", " about "), ("*", " "), ("_", " "),
]
_ACRO = re.compile(r"\b[A-Z]{2,6}s?\b")
_NUM = re.compile(r"(?<![\w.])(-?\d+)(?:\.(\d+))?(?![\w])")


def _acro(m: re.Match[str]) -> str:
    w = m.group(0)
    low = w.lower()
    if low in RESPELL:
        return RESPELL[low]
    plural = w.endswith("s") and len(w) > 2
    core = w[:-1] if plural else w
    return " ".join(core) + (" s" if plural else "")


def english_for_tts(text: str) -> str:
    s = str(text or "")
    s = re.sub(r"\$([^$]*)\$", r" \1 ", s)  # formulas read as their symbols
    s = re.sub(r"\\[a-zA-Z]+", " ", s).replace("{", " ").replace("}", " ")
    s = re.sub(r"\[(?:p\.|page |slide |part |lesson )\s*\d+[^\]]*\]", " ", s, flags=re.I)  # citations like [p. 3]
    s = re.sub(r"https?://\S+", " link ", s)
    s = re.sub(r"\bI/O\b", "i o", s)
    s = re.sub(r"(?i)\b(\w+)'s\b", lambda m: m.group(1) + "s", s)
    s = _ACRO.sub(_acro, s)
    for sym, word in _SYMBOLS:
        s = s.replace(sym, word)
    s = _NUM.sub(lambda m: ("minus " if m.group(1).startswith("-") else "") + number_words(abs(int(m.group(1)))) + (" point " + " ".join(_ONES[int(c)] for c in m.group(2)) if m.group(2) else ""), s)
    s = re.sub(r"[()\[\]\"“”‘’`:;|]", ", ", s).replace("—", ", ").replace("–", " ")
    words = [RESPELL.get(w.lower().strip(".,!?"), w) if w.lower().strip(".,!?") in RESPELL else w for w in s.split()]
    s = " ".join(words)
    s = re.sub(r"\s+([,.!?])", r"\1", s)
    s = re.sub(r"(,\s*){2,}", ", ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


# ----------------------------------------------------------------------------------------- native script
# common CS words as a Telugu / Hindi lecturer says them
_TE_WORDS = {
    "operating": "ఆపరేటింగ్", "system": "సిస్టమ్", "systems": "సిస్టమ్స్", "process": "ప్రాసెస్", "processes": "ప్రాసెసెస్", "processor": "ప్రాసెసర్",
    "thread": "థ్రెడ్", "threads": "థ్రెడ్స్", "memory": "మెమరీ", "file": "ఫైల్", "files": "ఫైల్స్", "kernel": "కెర్నల్", "scheduling": "షెడ్యూలింగ్",
    "scheduler": "షెడ్యూలర్", "deadlock": "డెడ్‌లాక్", "semaphore": "సెమాఫోర్", "mutex": "మ్యూటెక్స్", "page": "పేజ్", "paging": "పేజింగ్",
    "segment": "సెగ్మెంట్", "segmentation": "సెగ్మెంటేషన్", "virtual": "వర్చువల్", "cache": "క్యాష్", "disk": "డిస్క్", "buffer": "బఫర్",
    "queue": "క్యూ", "stack": "స్టాక్", "heap": "హీప్", "tree": "ట్రీ", "graph": "గ్రాఫ్", "node": "నోడ్", "nodes": "నోడ్స్", "edge": "ఎడ్జ్",
    "set": "సెట్", "sets": "సెట్స్", "function": "ఫంక్షన్", "functions": "ఫంక్షన్స్", "relation": "రిలేషన్", "relations": "రిలేషన్స్",
    "matrix": "మ్యాట్రిక్స్", "matrices": "మ్యాట్రిసెస్", "vector": "వెక్టర్", "vectors": "వెక్టర్స్", "logic": "లాజిక్", "proposition": "ప్రపోజిషన్",
    "propositions": "ప్రపోజిషన్స్", "conjunction": "కంజంక్షన్", "disjunction": "డిస్‌జంక్షన్", "negation": "నెగేషన్", "implication": "ఇంప్లికేషన్",
    "true": "ట్రూ", "false": "ఫాల్స్", "and": "అండ్", "or": "ఆర్", "not": "నాట్", "if": "ఇఫ్", "then": "దెన్", "algorithm": "ఆల్గారిథం",
    "data": "డేటా", "structure": "స్ట్రక్చర్", "structures": "స్ట్రక్చర్స్", "example": "ఎగ్జాంపుల్", "definition": "డెఫినిషన్", "theorem": "థీరమ్",
    "proof": "ప్రూఫ్", "time": "టైమ్", "space": "స్పేస్", "complexity": "కాంప్లెక్సిటీ", "input": "ఇన్‌పుట్", "output": "అవుట్‌పుట్", "user": "యూజర్",
    "resource": "రిసోర్స్", "resources": "రిసోర్సెస్", "allocation": "అలొకేషన్", "state": "స్టేట్", "block": "బ్లాక్", "ready": "రెడీ", "running": "రన్నింగ్",
    "waiting": "వెయిటింగ్", "burst": "బర్స్ట్", "priority": "ప్రయారిటీ", "round": "రౌండ్", "robin": "రాబిన్", "first": "ఫస్ట్", "come": "కమ్",
    "served": "సర్వ్డ్", "shortest": "షార్టెస్ట్", "job": "జాబ్", "next": "నెక్స్ట్", "average": "యావరేజ్", "turnaround": "టర్నరౌండ్", "frame": "ఫ్రేమ్",
    "frames": "ఫ్రేమ్స్", "fault": "ఫాల్ట్", "faults": "ఫాల్ట్స్", "hit": "హిట్", "miss": "మిస్", "table": "టేబుల్", "address": "అడ్రస్", "physical": "ఫిజికల్",
    "logical": "లాజికల్", "interrupt": "ఇంటరప్ట్", "driver": "డ్రైవర్", "network": "నెట్‌వర్క్", "security": "సెక్యూరిటీ", "protection": "ప్రొటెక్షన్",
    "banker": "బ్యాంకర్", "bankers": "బ్యాంకర్స్", "safe": "సేఫ్", "unsafe": "అన్‌సేఫ్", "sequence": "సీక్వెన్స్", "request": "రిక్వెస్ట్", "release": "రిలీజ్",
    "critical": "క్రిటికల్", "section": "సెక్షన్", "race": "రేస్", "condition": "కండిషన్", "synchronization": "సింక్రనైజేషన్", "lock": "లాక్",
    "pigeonhole": "పిజియన్‌హోల్", "principle": "ప్రిన్సిపల్", "permutation": "పర్ముటేషన్", "combination": "కాంబినేషన్", "counting": "కౌంటింగ్",
    "probability": "ప్రాబబిలిటీ", "eigenvalue": "ఐగెన్‌వాల్యూ", "determinant": "డిటర్మినెంట్", "inverse": "ఇన్వర్స్", "rank": "ర్యాంక్",
    "is": "ఈజ్", "the": "ది", "a": "ఏ", "of": "ఆఫ్", "to": "టు", "for": "ఫర్", "in": "ఇన్", "on": "ఆన్", "with": "విత్", "by": "బై", "all": "ఆల్",
    "program": "ప్రోగ్రామ్", "programs": "ప్రోగ్రామ్స్", "replacement": "రీప్లేస్‌మెంట్", "ok": "ఓకే", "yes": "యస్", "no": "నో", "page": "పేజ్", "fifo": "ఫైఫో", "lifo": "లైఫో",
}
_LETTER_TE = {"a": "ఏ", "b": "బీ", "c": "సీ", "d": "డీ", "e": "ఈ", "f": "ఎఫ్", "g": "జీ", "h": "ఏచ్", "i": "ఐ", "j": "జే", "k": "కే", "l": "ఎల్", "m": "ఎమ్",
              "n": "ఎన్", "o": "ఓ", "p": "పీ", "q": "క్యూ", "r": "ఆర్", "s": "ఎస్", "t": "టీ", "u": "యూ", "v": "వీ", "w": "డబ్ల్యూ", "x": "ఎక్స్", "y": "వై", "z": "జెడ్"}
_LETTER_HI = {"a": "ए", "b": "बी", "c": "सी", "d": "डी", "e": "ई", "f": "एफ", "g": "जी", "h": "एच", "i": "आई", "j": "जे", "k": "के", "l": "एल", "m": "एम",
              "n": "एन", "o": "ओ", "p": "पी", "q": "क्यू", "r": "आर", "s": "एस", "t": "टी", "u": "यू", "v": "वी", "w": "डब्ल्यू", "x": "एक्स", "y": "वाई", "z": "ज़ेड"}

_SCRIPT = {
    "te": dict(
        cons={"k": "క", "g": "గ", "c": "క", "j": "జ", "t": "ట", "d": "డ", "n": "న", "p": "ప", "b": "బ", "m": "మ", "y": "య", "r": "ర", "l": "ల", "v": "వ",
              "w": "వ", "s": "స", "h": "హ", "f": "ఫ", "q": "క", "z": "జ", "x": "క్స",
              "ch": "చ", "sh": "ష", "th": "థ", "ph": "ఫ", "kh": "ఖ", "gh": "గ", "ck": "క్", "ng": "ంగ్", "wh": "వ"},
        vow={"a": "", "e": "ె", "i": "ి", "o": "ొ", "u": "ు", "aa": "ా", "ee": "ీ", "oo": "ూ", "ai": "ై", "ay": "ే", "ou": "ౌ", "ow": "ౌ", "oa": "ో", "ea": "ీ",
             "ie": "ీ", "oi": "ాయ్", "au": "ా", "ey": "ే"},
        ind={"a": "అ", "e": "ఎ", "i": "ఇ", "o": "ఒ", "u": "ఉ", "aa": "ఆ", "ee": "ఈ", "oo": "ఊ", "ai": "ఐ", "ay": "ఏ", "ou": "ఔ", "ow": "ఔ", "oa": "ఓ", "ea": "ఈ",
             "ie": "ఈ", "oi": "ఆయ్", "au": "ఆ", "ey": "ఏ"},
        halant="్", words=_TE_WORDS, letters=_LETTER_TE),
    "hi": dict(
        cons={"k": "क", "g": "ग", "c": "क", "j": "ज", "t": "ट", "d": "ड", "n": "न", "p": "प", "b": "ब", "m": "म", "y": "य", "r": "र", "l": "ल", "v": "व",
              "w": "व", "s": "स", "h": "ह", "f": "फ", "q": "क", "z": "ज़", "x": "क्स",
              "ch": "च", "sh": "श", "th": "थ", "ph": "फ", "kh": "ख", "gh": "घ", "ck": "क्", "ng": "ंग्", "wh": "व"},
        vow={"a": "", "e": "े", "i": "ि", "o": "ो", "u": "ु", "aa": "ा", "ee": "ी", "oo": "ू", "ai": "ै", "ay": "े", "ou": "ौ", "ow": "ौ", "oa": "ो", "ea": "ी",
             "ie": "ी", "oi": "ॉय", "au": "ा", "ey": "े"},
        ind={"a": "अ", "e": "ए", "i": "इ", "o": "ओ", "u": "उ", "aa": "आ", "ee": "ई", "oo": "ऊ", "ai": "ऐ", "ay": "ए", "ou": "औ", "ow": "औ", "oa": "ओ", "ea": "ई",
             "ie": "ई", "oi": "ऑय", "au": "आ", "ey": "ए"},
        halant="्", words={}, letters=_LETTER_HI),
}
_VOWELS = "aeiou"
_DIGRAPH_V = ("ee", "oo", "ai", "ay", "ou", "ow", "oa", "ea", "ie", "oi", "au", "ey")
_DIGRAPH_C = ("ch", "sh", "th", "ph", "kh", "gh", "ck", "ng", "wh")


def _translit_word(word: str, lang: str) -> str:
    sc = _SCRIPT[lang]
    w = re.sub(r"([^aeiou])\1+", r"\1", word.lower())  # "process" -> "proces": doubled consonants are one sound
    if word.lower() in sc["words"]:
        return sc["words"][word.lower()]
    if w in sc["words"]:
        return sc["words"][w]
    if not w.isalpha():
        return word
    if w.isupper() or (word.isupper() and len(word) <= 5):
        return " ".join(sc["letters"].get(c, c) for c in w)
    for suf, rep in (("tion", "shan"), ("sion", "shan"), ("ing", "ing"), ("ed", "d")):
        if w.endswith(suf) and len(w) > len(suf) + 1 and suf in ("tion", "sion"):
            w = w[: -len(suf)] + rep
            break
    if w.endswith("e") and len(w) > 3 and w[-2] not in _VOWELS:  # silent final e: "page", "state"
        w = w[:-1]
    out: list[str] = []
    i, n = 0, len(w)
    pending: str | None = None  # a consonant waiting for its vowel
    while i < n:
        two, ch = w[i : i + 2], w[i]
        if two in _DIGRAPH_V and ch in _VOWELS + "y":
            v, i = two, i + 2
        elif ch in _VOWELS:
            v, i = ch, i + 1
        else:
            v = None
        if v is not None:
            if pending is None:
                out.append(sc["ind"].get(v, ""))
            else:
                out.append(pending + sc["vow"].get(v, ""))
                pending = None
            continue
        # consonant
        if pending is not None:
            out.append(pending + sc["halant"])
            pending = None
        if two in _DIGRAPH_C:
            c, i = sc["cons"][two], i + 2
        else:
            c, i = sc["cons"].get(ch, ch), i + 1
            if ch == "c" and i < n and w[i] in "eiy":
                c = sc["cons"]["s"]
            if ch == "g" and i < n and w[i] in "ei" and i > 1:
                c = sc["cons"]["j"]
        pending = c
        if c.endswith(sc["halant"]) or c in ("ంగ్", "ంగ్"):
            out.append(c)
            pending = None
    if pending is not None:
        out.append(pending + sc["halant"])
    return "".join(out)


_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'’]*")


def to_native_script(text: str, lang: str) -> str:
    """Write the English words of a Telugu/Hindi reply in the native script so one voice reads everything."""
    if lang not in _SCRIPT:
        return text
    s = re.sub(r"\$([^$]*)\$", r" \1 ", str(text or ""))
    s = re.sub(r"\\[a-zA-Z]+", " ", s).replace("{", " ").replace("}", " ")
    s = re.sub(r"\[(?:p\.|page |slide |part |lesson )\s*\d+[^\]]*\]", " ", s, flags=re.I)
    s = re.sub(r"\bI/O\b", "I O", s)
    for sym, word in _SYMBOLS:
        if sym not in ("_", "*", "#", "@", "/", "&", "~", "^"):  # keep punctuation-like marks out of speech
            s = s.replace(sym, word)
    s = _NUM.sub(lambda m: number_words(abs(int(m.group(1)))) if not m.group(2) else m.group(0), s)
    return re.sub(r"\s+", " ", _LATIN_WORD.sub(lambda m: _translit_word(m.group(0).replace("’", "'").replace("'", ""), lang), s)).strip()
