"""Concept extraction + relation inference for the knowledge map.

No neural model and no API are needed: concepts are mined statistically and from the *language of
teaching* (definitions, headings, emphasis), which works the same way for English, Telugu, Hindi
and Sanskrit text.  Every concept carries provenance (page, section, source sentence) so the UI
can always show *where in the student's own material* a node comes from.

Pipeline
--------
1. Split every section into sentences and compute stem sequences (hyphen-aware, plural-normalised).
2. Generate n-gram candidates from runs of content words (stop-words / generic words are boundaries).
3. Boost candidates that are *defined* ("X is defined as", "X అంటే", "X कहलाता है", "X इति"),
   appear in headings, are emphasised (markdown bold) or are acronym-introduced ``Long Form (ABBR)``.
4. Score (frequency x specificity x length + boosts), drop fragments of longer terms, balance across
   sections so that every topic gets children.
5. Attach a definition, examples, formulas and key sentences to each concept.
6. Infer ``prerequisite`` edges (a term used inside another term's definition and introduced earlier)
   and ``related`` edges (sentence co-occurrence).
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from indiclearn.studio.lang import (
    ALL_STOP,
    WORD_RE,
    detect_language,
    is_indic,
    split_sentences,
    stem,
)
from indiclearn.studio.parse import Block
from indiclearn.studio.structure import Section

# ---------------------------------------------------------------------------
GENERIC_EN = set(
    ["important", "major", "various", "several", "significant", "simple", "recent", "existing", "previous", "novel", "effective", "efficient", "common", "general", "specific", "particular", "main", "key", "typical", "potential", "possible", "overall", "early", "high", "low", "large", "small", "better", "best", "good", "strong", "weak", "further", "recently", "earlier", "called", "known", "define", "defined", "describe", "described", "achieve", "achieves", "achieved", "introduce", "introduces", "propose", "proposes", "show", "shows", "shown", "provide", "provides", "provided", "obtain", "obtained", "perform", "performs", "performed", "improve", "improves", "improved", "reduce", "reduces", "reduced", "require", "requires", "required", "allow", "allows", "enable", "enables", "include", "includes", "included", "consist", "consists", "contain", "contains", "follow", "follows", "denote", "denotes", "represent", "represents", "let", "note", "consider", "considered", "assume", "assumed", "get", "gets", "make", "makes", "made", "take", "takes", "taken", "find", "finds", "found", "need", "needs", "want", "like", "become", "becomes", "apply", "applied", "compute", "computed", "calculate", "calculated", "produce", "produces", "depend", "depends", "vary", "varies", "change", "changes", "increase", "increases", "decrease", "decreases", "et", "al", "thus", "hence", "also", "however", "therefore", "moreover", "mainly", "mostly", "usually", "generally", "typically", "respectively", "approximately", "roughly", "almost", "nearly", "similar", "different", "various", "certain", "way", "ways", "kind", "lot", "thing", "things", "work", "works", "study", "studies", "paper", "papers", "section", "sections", "figure", "figures", "table", "tables", "page", "pages", "result", "results", "well", "able", "first", "second", "third", "last", "next", "previous", "current", "present", "example", "examples", "according", "within", "without", "across", "among", "among", "whereas", "upon", "either", "neither", "rather", "quite", "easy", "hard", "difficult", "usually", "often", "sometimes", "every", "each", "many", "much", "more", "most", "less", "least", "become", "becomes", "became", "goes", "go", "going", "went", "come", "comes", "coming", "new", "old", "own", "same", "other", "another", "help", "helps", "helped", "lead", "leads", "led", "give", "gives", "given", "keep", "keeps", "kept", "put", "puts", "set", "sets", "smaller", "fewer", "larger", "higher", "lower", "greater", "bigger", "reproduced", "substantially", "remaining", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "twenty", "thirty", "hundred", "thousand", "million", "billion", "half", "whole", "entire", "total", "overall", "final", "initial", "complete", "full", "partial", "additional", "extra", "other", "others", "called", "named", "using", "use", "used", "uses", "user", "users", "performance", "approach", "approaches", "method", "methods", "technique", "techniques", "task", "tasks", "information", "problem", "problems", "solution", "solutions", "overall", "following", "proposed", "recordings", "recording", "amount", "amounts"]
)
# words that may legitimately appear inside terms but never start/end one
EDGE_WEAK = {
    "based", "related", "type", "types", "kind", "form", "forms", "number", "numbers", "set", "sets", "level", "levels",
    "case", "cases", "part", "parts", "group", "groups", "order", "value", "values", "time", "times", "step", "steps",
    "way", "ways", "rule", "rules", "side", "sides", "point", "points", "term", "terms", "factor", "factors",
}
EXAMPLE_CUES = (
    "for example", "for instance", "e.g.", "such as", "consider ", "suppose ", "let us take", "e.g ",
    "ఉదాహరణకు", "ఉదాహరణ", "ఉదా", "उदाहरण", "जैसे", "मान लीजिए", "उदाहरणम्", "यथा", "उदाहरणार्थम्",
)
_DET = {"the", "a", "an", "this", "that", "these", "those", "its", "their", "our", "each", "every", "any", "such"}
_PRONOUN_START = {"it", "this", "these", "they", "we", "there", "he", "she", "i", "you", "which", "that", "here", "one"}
_COPULA = {"is", "are"}
_COPULA_NEXT = {
    "defined", "called", "known", "termed", "referred", "a", "an", "the", "said", "considered", "one",
    "described", "denoted", "any", "simply", "basically", "essentially", "formally", "commonly", "also",
}
_VERB_TRIGGERS_EN = {"refers", "means", "denotes", "represents", "describes", "stands", "involves", "enables", "allows", "maps", "computes", "measures", "captures"}
_DEF_TRIGGERS = {
    "te": {"అంటే", "అనగా", "అనేది", "అనేవి", "అంటారు", "అనబడును", "పిలుస్తారు"},
    "hi": {"कहलाता", "कहलाती", "कहलाते", "तात्पर्य", "कहते", "कहा", "अर्थ", "कहलाता है"},
    "sa": {"इति", "इत्युच्यते", "उच्यते", "नाम", "इत्यर्थः", "कथ्यते"},
}
_MAX_NGRAM = {"en": 4, "te": 3, "hi": 3, "sa": 3}
_CITATION = re.compile(r"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]|\(\s*[A-Z][A-Za-z]+(?: et al\.)?,?\s*\d{4}\s*\)")
_ACRONYM_PAT = re.compile(r"(?P<long>(?:[A-Z][A-Za-z\-]+|[a-z]+)(?:\s+(?:[A-Za-z][A-Za-z\-]+|of|and|for|the)){1,5})\s*\(\s*(?P<abbr>[A-Z][A-Za-z0-9\-]{1,9}s?)\s*\)")


@dataclass
class Sent:
    idx: int
    section_id: str
    order: int  # section order
    block_idx: int
    page: int
    text: str
    words: list[str]
    stems: list[str]
    first_in_block: bool
    kind: str
    wmap: list[int]  # word index -> start index in stems
    brk: list[bool] = field(default_factory=list)  # True if punctuation separates word i-1 and i


@dataclass
class Concept:
    id: str
    key: str
    label: str
    section_id: str
    score: float = 0.0
    importance: float = 0.0
    definition: str | None = None
    definition_quality: str | None = None  # defined | context
    definition_page: int | None = None
    examples: list[str] = field(default_factory=list)
    formulas: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    mentions: list[dict] = field(default_factory=list)
    stems: tuple[str, ...] = ()
    first_pos: tuple[int, int] = (0, 0)
    first_sent: int = 0
    sections: list[str] = field(default_factory=list)
    count: int = 0
    section_topic: str | None = None  # set when the concept *is* a section (heading == term)


@dataclass
class Relation:
    source: str
    target: str
    type: str  # prereq | related
    weight: float = 1.0


# ---------------------------------------------------------------------------
def _flat_stems(word: str, lang: str) -> list[str]:
    parts = [p for p in re.split(r"[-’']", word.lower()) if p]
    return [stem(p, lang) for p in parts]


def _content_token(w: str, lang_stop: set[str]) -> bool:
    lw = w.lower()
    if lw in ALL_STOP or lw in GENERIC_EN or lw in lang_stop:
        return False
    if w.isdigit() or len(w) < 2 or w[0].isdigit():
        return False
    if re.fullmatch(r"[\d.\-]+", w):
        return False
    if re.fullmatch(r"[IVX]{1,4}", w) or (lw.endswith("ly") and len(lw) >= 7 and w.islower()):
        return False
    if re.search(r"[a-z]{2,}[A-Z]", w) and not re.fullmatch(r"[A-Za-z][a-z]+[A-Z][a-z]+", w):
        return False  # glued PDF text such as "TheMIMIC" / "whereL"
    letters = sum(ch.isalpha() for ch in w)
    if letters < max(2, int(0.5 * len(w))):
        return False
    return True


_BREAK_SEP = re.compile(r"[,;:()\[\]{}\"“”‘’—–/|•]|\s-\s|\.")
_POSS = re.compile(r"[’']s$", re.IGNORECASE)


def build_sentences(sections: list[Section], lang: str) -> list[Sent]:
    out: list[Sent] = []
    gi = 0
    for sec in sections:
        if sec.skip:
            continue
        for bi, b in enumerate(sec.blocks):
            if b.kind not in ("para", "bullet") or b.meta.get("caption") or b.meta.get("table"):
                continue
            text = _CITATION.sub("", b.text)
            for si, st in enumerate(split_sentences(text)):
                spans = [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(st)]
                if len(spans) < 3:
                    continue
                words: list[str] = []
                brk: list[bool] = []
                prev_end = 0
                for i, (w, a, e) in enumerate(spans):
                    words.append(_POSS.sub("", w))
                    brk.append(i == 0 or bool(_BREAK_SEP.search(st[prev_end:a])))
                    prev_end = e
                stems: list[str] = []
                wmap: list[int] = []
                for w in words:
                    wmap.append(len(stems))
                    stems.extend(_flat_stems(w, lang))
                out.append(Sent(gi, sec.id, sec.order, bi, b.page, st, words, stems, si == 0, b.kind, wmap, brk))
                gi += 1
    return out


def _phrase_positions(stems: list[str], phrase: tuple[str, ...], index: dict[str, list[int]] | None = None) -> list[int]:
    n = len(phrase)
    if n == 0:
        return []
    starts = index.get(phrase[0], []) if index is not None else [i for i, s in enumerate(stems) if s == phrase[0]]
    return [i for i in starts if tuple(stems[i : i + n]) == phrase]


def _make_id(key: str) -> str:
    return "c" + hashlib.md5(key.encode("utf-8")).hexdigest()[:8]


def _tidy_label(surface: str) -> str:
    s = surface.strip(" -–—,:;.()[]")
    if not s:
        return s
    if s.islower():
        s = s[0].upper() + s[1:]
    return s


def _best_surface(surfaces: Counter[str], lang: str, key: str) -> str:
    # prefer the un-inflected (key-like) form among frequent ones
    if not surfaces:
        return key
    top = surfaces.most_common(6)
    maxc = top[0][1]
    cands = [s for s, c in top if c >= max(1, maxc * 0.5)] or [top[0][0]]
    # prefer title-case/acronym surface forms and shorter ones
    def rank(s: str) -> tuple:
        letters = [ch for ch in s if ch.isalpha()]
        upper_ratio = sum(ch.isupper() for ch in letters) / max(len(letters), 1)
        acr = 1 if (upper_ratio > 0.8 and len(letters) <= 6) else 0
        return (-acr, len(s))
    cands.sort(key=rank)
    return _tidy_label(cands[0])


# ---------------------------------------------------------------------------
def _definitional_candidates(s: Sent, lang: str) -> list[tuple[int, int, str]]:
    """Return [(start_word, end_word_excl, kind)] spans of terms that this sentence *defines*."""
    spans: list[tuple[int, int, str]] = []
    words = s.words
    low = [w.lower() for w in words]
    n = len(words)
    sl = ALL_STOP if lang == "en" else set()

    def run_before(i: int, max_len: int = 4) -> tuple[int, int] | None:
        j = i
        while j > 0 and (low[j - 1] in _DET) and False:
            j -= 1
        end = i
        while end > 0 and (low[end - 1] in ALL_STOP or low[end - 1] in {"को", "का", "के", "की"}):
            end -= 1
        start = end
        while start > 0 and end - start < max_len and _content_token(words[start - 1], sl) and not (low[start - 1] in ALL_STOP):
            start -= 1
        if end - start >= 1:
            return start, end
        return None

    if lang in ("en",) or not is_indic(s.text):
        if low and low[0] in {"this", "that", "these", "those", "such", "its", "their", "it", "they", "he", "she"}:
            return spans  # anaphoric subject - not a definition of a new term
        # copula pattern in first 8 tokens
        for i in range(1, min(n - 1, 9)):
            if low[i] in _COPULA and (low[i + 1] in _COPULA_NEXT):
                subj = list(range(i))
                while subj and low[subj[0]] in _DET:
                    subj.pop(0)
                while len(subj) > 1 and words[subj[-1]].isupper() and 2 <= len(words[subj[-1]]) <= 9 and s.brk[subj[-1]]:
                    subj.pop()  # "Electrocardiogram (ECG) is ..." -> subject = Electrocardiogram
                if subj and low[subj[0]] not in _PRONOUN_START and 1 <= len(subj) <= 6:
                    # trim to content run at the end of the subject
                    end = subj[-1] + 1
                    start = end
                    while start > subj[0] and _content_token(words[start - 1], sl) and not s.brk[start]:
                        start -= 1
                    if 1 <= end - start <= 5 and start == subj[0]:
                        spans.append((start, end, "copula"))
                break
        for i in range(1, min(n, 9)):
            if low[i] in _VERB_TRIGGERS_EN and (i + 1 < n) and (low[i] != "stands" or low[i + 1] == "for"):
                subj = list(range(i))
                while subj and low[subj[0]] in _DET:
                    subj.pop(0)
                if subj and low[subj[0]] not in _PRONOUN_START:
                    end = subj[-1] + 1
                    start = end
                    while start > subj[0] and _content_token(words[start - 1], sl):
                        start -= 1
                    if 1 <= end - start <= 4 and start == subj[0]:
                        spans.append((start, end, "verb"))
                break
        for i in range(n - 1):
            if low[i] in {"called", "termed", "named", "denoted"} or (low[i] == "known" and i + 1 < n and low[i + 1] == "as") or (
                low[i] == "referred" and i + 1 < n and low[i + 1] == "to"
            ):
                j = i + 1
                while j < n and low[j] in {"as", "to", "by", "the", "a", "an"}:
                    j += 1
                start = j
                while j < n and j - start < 4 and _content_token(words[j], sl):
                    j += 1
                if j > start:
                    spans.append((start, j, "called"))
                break
    trig = _DEF_TRIGGERS.get(lang)
    if trig is None and is_indic(s.text):
        li = detect_language(s.text).base
        trig = _DEF_TRIGGERS.get(li)
    if trig:
        for i in range(1, n):
            if words[i] in trig:
                r = run_before(i, 3)
                if r:
                    spans.append((r[0], r[1], "indic"))
                break
    return spans


_ABBR_PAREN = re.compile(r"\(\s*(?P<abbr>[A-Z][A-Z0-9\-]{1,9}s?)\s*\)")
_SMALL = {"of", "and", "for", "the", "in", "on", "with", "to"}


def _acronym_pairs(text: str) -> list[tuple[str, str]]:
    """``Convolutional Neural Networks (CNNs)`` -> ("Convolutional Neural Networks", "CNNs")."""
    out: list[tuple[str, str]] = []
    for m in _ABBR_PAREN.finditer(text):
        abbr = m.group("abbr")
        core = re.sub(r"[^A-Z0-9]", "", abbr.upper().rstrip("S")) or re.sub(r"[^A-Z0-9]", "", abbr.upper())
        before = WORD_RE.findall(text[: m.start()])[-8:]
        if not before or len(core) < 2:
            continue
        best: list[str] | None = None
        for k in range(1, len(before) + 1):
            tail = before[-k:]
            while tail and tail[0].lower() in _SMALL:
                tail = tail[1:]
            if not tail:
                continue
            parts = [p for w in tail for p in re.split(r"-", w) if p]
            initials = "".join(p[0] for p in parts if p.lower() not in _SMALL).upper()
            letters = "".join(c for c in "".join(tail).upper() if c.isalnum())
            first_ok = tail[0][0].upper() == core[0]
            if not first_ok:
                continue
            if initials == core or (len(tail) == 1 and _is_subsequence(core, letters)) or (
                len(tail) >= 2 and core.startswith(initials[:2]) and len(core) in (len(initials), len(initials) + 1, len(initials) - 1)
            ):
                best = tail
                break
        if best:
            out.append((" ".join(best), abbr))
    return out


def _is_subsequence(small: str, big: str) -> bool:
    it = iter(big)
    return all(ch in it for ch in small)


# ---------------------------------------------------------------------------
def extract_concepts(
    sections: list[Section],
    lang: str = "en",
    max_concepts: int | None = None,
) -> tuple[list[Concept], list[Relation], list[Sent]]:
    """Mine concepts from *sections*.  Returns (concepts, relations, sentences)."""
    sents = build_sentences(sections, lang)
    if not sents:
        return [], [], sents
    sec_by_id = {s.id: s for s in sections}
    live_sections = [s for s in sections if not s.skip and s.level >= 0]
    n_sections = max(len([s for s in live_sections if s.blocks]), 1)
    lang_stop: set[str] = set()
    maxn = _MAX_NGRAM.get(lang, 3)

    stats: dict[tuple[str, ...], dict] = {}

    def st(key: tuple[str, ...]) -> dict:
        d = stats.get(key)
        if d is None:
            d = stats[key] = {
                "tf": 0, "surf": Counter(), "secs": Counter(), "first": (10**9, 10**9), "defs": 0, "emph": 0,
                "head": 0, "tcase": 0, "acr": 0, "n": len(key),
            }
        return d

    def register(words: list[str], s_order: int, s_blk: int, sec_id: str, mid_sentence: bool, weight: int = 1, *, kind: str = "tf") -> tuple[str, ...]:
        flat: list[str] = []
        for w in words:
            flat.extend(_flat_stems(w, lang))
        key = tuple(flat)
        d = st(key)
        if kind == "tf":
            d["tf"] += weight
            d["secs"][sec_id] += weight
        d["surf"][" ".join(words)] += max(weight, 1)
        d["first"] = min(d["first"], (s_order, s_blk))
        if mid_sentence and len(words) >= 2 and all(w[:1].isupper() for w in words if w.lower() not in {"of", "and", "for"}):
            d["tcase"] += 1
        return key

    # --- pass 1: n-gram candidates from content runs ---
    for s_ in sents:
        words = s_.words
        low_ = [w.lower() for w in words]
        cls: list[str] = []  # c = content, w = weak (inside only), o = connector, b = boundary
        for w, lw in zip(words, low_):
            if lw == "of":
                cls.append("o")
            elif _content_token(w, lang_stop):
                cls.append("w" if lw in EDGE_WEAK else "c")
            else:
                cls.append("b")
        i = 0
        while i < len(words):
            if cls[i] in ("b", "o"):
                i += 1
                continue
            j = i + 1
            while j < len(words) and cls[j] != "b" and not s_.brk[j]:
                if cls[j] == "o" and (j + 1 >= len(words) or cls[j + 1] in ("b", "o") or s_.brk[j + 1]):
                    break
                j += 1
            run = words[i:j]
            rc = cls[i:j]
            L = len(run)
            for a in range(L):
                if rc[a] != "c":
                    continue
                for n in range(1, min(maxn + 1, L - a) + 1):
                    if rc[a + n - 1] != "c":
                        continue
                    gram = run[a : a + n]
                    if n == 1 and len(gram[0]) < 3 and not gram[0].isupper():
                        continue
                    mid = (i + a) > 0
                    register(gram, s_.order, s_.block_idx, s_.section_id, mid)
            i = j

    # --- pass 2: definitional spans, acronyms, headings, emphasis ---
    acr_alias: dict[tuple[str, ...], set[str]] = defaultdict(set)
    acr_to_long: dict[str, tuple[str, ...]] = {}
    for s in sents:
        for (a, b, _kind) in _definitional_candidates(s, lang):
            gram = s.words[a:b]
            if not gram:
                continue
            key = register(gram, s.order, s.block_idx, s.section_id, a > 0, 0, kind="def")
            stats[key]["defs"] += 1
        for long_, abbr in _acronym_pairs(s.text):
            lw = re.findall(r"[A-Za-z][A-Za-z\-]*", long_)
            if lw:
                lkey = register(lw, s.order, s.block_idx, s.section_id, True, 0, kind="def")
                stats[lkey]["defs"] += 1
                stats[lkey]["acr"] += 1
                akey = tuple(_flat_stems(abbr, lang))
                acr_alias[lkey].add(abbr)
                acr_to_long[abbr.upper().rstrip("S")] = lkey
    for sec in sections:
        if sec.skip:
            continue
        for bi, b in enumerate(sec.blocks):
            for e in b.meta.get("emph", []) or []:
                ws = WORD_RE.findall(e)
                if 1 <= len(ws) <= 5:
                    key = register(ws, sec.order, bi, sec.id, True, 1)
                    stats[key]["emph"] += 1
        if sec.level >= 1:
            ws_all = WORD_RE.findall(sec.title)
            # whole heading as a candidate and its content runs
            runs: list[list[str]] = []
            cur: list[str] = []
            for w in ws_all:
                if _content_token(w, lang_stop) or (cur and w.lower() in {"of", "and", "for"} and False):
                    cur.append(w)
                else:
                    if cur:
                        runs.append(cur)
                    cur = []
            if cur:
                runs.append(cur)
            for r in runs:
                if 1 <= len(r) <= maxn + 1 and r[0].lower() not in EDGE_WEAK:
                    if len(r) == 1 and len(runs) > 1 and not re.match(r"^[A-Z0-9]{2,}", r[0]):
                        continue  # "Comparison" in "Comparison with IMLE-Net" is not a concept
                    key = register(r, sec.order, 0, sec.id, True, 1)
                    stats[key]["head"] += 1

    # --- merge acronym stats into long forms ---
    for lkey, abbrs in acr_alias.items():
        for abbr in abbrs:
            akey = tuple(_flat_stems(abbr, lang))
            if akey in stats and akey != lkey and lkey in stats:
                a, l = stats[akey], stats[lkey]
                l["tf"] += a["tf"]
                l["secs"].update(a["secs"])
                stats.pop(akey)

    # --- scoring ---
    scored: list[tuple[float, tuple[str, ...]]] = []
    for key, d in stats.items():
        tf, n = d["tf"], d["n"]
        if tf == 0 and not (d["defs"] or d["head"] or d["emph"]):
            continue
        strong = d["defs"] + d["head"] + d["emph"]
        surf0 = d["surf"].most_common(1)[0][0] if d["surf"] else " ".join(key)
        is_acr = n == 1 and surf0.isupper() and 2 <= len(surf0) <= 8
        if n == 1 and not is_acr:
            if tf < 2 and not strong:
                continue
            if tf < 4 and not strong and d["tcase"] == 0:
                continue
            if not strong and not is_indic(surf0) and surf0.lower().endswith(("ed", "ing")):
                continue
        if is_acr and tf < 3 and not strong:
            continue
        if n >= 2 and tf < 2 and not strong:
            continue
        if any(len(t) < 2 and not t.isalpha() for t in key):
            continue
        df = len(d["secs"]) or 1
        idf = math.log((n_sections + 1) / (df + 0.5)) + 1.0
        len_factor = {1: 1.0, 2: 1.35, 3: 1.45}.get(n, 1.35)
        base = (1.0 + math.log(max(tf, 1))) * math.sqrt(idf) * len_factor
        score = base + 3.0 * min(d["defs"], 2) + 2.5 * min(d["head"], 2) + 2.0 * min(d["emph"], 2) + 0.8 * min(d["tcase"], 3)
        if is_acr and tf >= 2:
            score += 1.0
        if d["acr"]:
            score += 1.5
        if n == 1 and not is_acr and not strong:
            score *= 0.55
        if n == 1 and not is_indic(surf0) and surf0.lower() in GENERIC_EN:
            score *= 0.4
        if df / n_sections > 0.6 and n == 1 and not strong:
            score *= 0.6
        scored.append((score, key))
    if not scored:
        return [], [], sents
    scored.sort(key=lambda x: -x[0])
    keyscore = {k: sc for sc, k in scored}

    # --- drop fragments of longer terms ---
    def contained(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
        if len(a) >= len(b):
            return False
        return any(b[i : i + len(a)] == a for i in range(len(b) - len(a) + 1))

    by_len = sorted(keyscore, key=lambda k: -len(k))
    dropped: set[tuple[str, ...]] = set()
    for k in keyscore:
        if k in dropped:
            continue
        dk = stats[k]
        independent = dk["defs"] or dk["head"] or dk["emph"]
        for lk in by_len:
            if len(lk) <= len(k):
                break
            if lk in dropped or not contained(k, lk):
                continue
            dl = stats[lk]
            if dl["tf"] >= 2 and dk["tf"] <= 1.35 * dl["tf"] + 1 and not independent:
                dropped.add(k)
                break
        else:
            pass
    for k in list(keyscore):
        if k in dropped:
            continue
        dk = stats[k]
        if dk["n"] >= 2 and dk["tf"] <= 1 and not (dk["defs"] or dk["head"] or dk["emph"]):
            # one-off long phrase containing a stronger shorter term -> drop
            if any(contained(sk, k) and keyscore.get(sk, 0) > keyscore[k] for sk in keyscore):
                dropped.add(k)
    for k in list(keyscore):
        dk = stats[k]
        if k in dropped or dk["n"] != 1 or dk["defs"] or dk["head"] or dk["emph"]:
            continue
        for lk in by_len:
            if len(lk) < 2 or lk in dropped or not contained(k, lk):
                continue
            dl = stats[lk]
            if dl["tf"] >= 2 and dk["tf"] <= 2.5 * dl["tf"]:
                dropped.add(k)
                break
    ranked = [(sc, k) for sc, k in scored if k not in dropped]

    # --- selection with per-section coverage ---
    n_sent = len(sents)
    leaf_secs = [s for s in live_sections if not [c for c in s.children if not sec_by_id[c].skip] and s.blocks]
    if max_concepts is None:
        max_concepts = int(max(14, min(110, 4 * math.sqrt(n_sent) + 1.2 * len(leaf_secs))))
    chosen: dict[tuple[str, ...], dict] = {}
    per_section: Counter[str] = Counter()
    # primary section per candidate = where it is densest (ties: earliest)
    def primary_section(k: tuple[str, ...]) -> str:
        d = stats[k]
        if not d["secs"]:
            return "s0"
        top = max(d["secs"].values())
        cands = [sid for sid, c in d["secs"].items() if c == top]
        return min(cands, key=lambda sid: sec_by_id[sid].order)

    sec_caps: dict[str, int] = {}
    for s in live_sections:
        total_chars = sum(len(b.text) for b in s.blocks)
        sec_caps[s.id] = 4 + min(6, total_chars // 500)
    for sc, k in ranked:
        if len(chosen) >= max_concepts:
            break
        ps = primary_section(k)
        dk = stats[k]
        overflow = 3 if (dk["defs"] or dk["emph"] or dk["head"]) else 0
        if per_section[ps] >= sec_caps.get(ps, 6) + overflow:
            continue
        chosen[k] = {"score": sc, "sec": ps}
        per_section[ps] += 1
    # guarantee coverage: each leaf section gets at least one concept if any candidate is dense there
    for s in leaf_secs:
        if per_section[s.id] == 0:
            local = [(stats[k]["secs"].get(s.id, 0) * keyscore[k], k) for sc, k in ranked if s.id in stats[k]["secs"] and k not in chosen]
            local.sort(key=lambda x: -x[0])
            for _, k in local[:2]:
                chosen[k] = {"score": keyscore[k], "sec": s.id}
                per_section[s.id] += 1

    # --- build Concept objects ---
    section_title_keys = {tuple(f for w in WORD_RE.findall(s.title) for f in _flat_stems(w, lang)): s.id for s in live_sections if s.level >= 1 and not s.synthetic}
    concepts: list[Concept] = []
    used_ids: set[str] = set()
    # a chosen term whose section title *is* that term (e.g. "Convolution") becomes the section's own concept
    for k, info in chosen.items():
        d = stats[k]
        surf = _best_surface(d["surf"], lang, " ".join(k))
        cid = _make_id(" ".join(k))
        if cid in used_ids:
            continue
        used_ids.add(cid)
        aliases = sorted({a for a in acr_alias.get(k, set())})
        if aliases and not surf.isupper() and f"({aliases[0]})" not in surf and len(surf) + len(aliases[0]) < 48:
            surf = f"{surf} ({aliases[0]})"
        topic_sid = section_title_keys.get(k)
        c = Concept(
            id=cid, key=" ".join(k), label=surf, section_id=topic_sid or info["sec"], score=info["score"], aliases=aliases,
            stems=k, first_pos=d["first"], sections=[sid for sid, _ in d["secs"].most_common()], count=d["tf"],
            section_topic=topic_sid,
        )
        concepts.append(c)
    if not concepts:
        return [], [], sents
    mx = max(c.score for c in concepts)
    for c in concepts:
        c.importance = round(min(1.0, c.score / mx), 3) if mx else 0.0
    concepts.sort(key=lambda c: (c.first_pos, -c.score))

    concepts = _merge_acronym_variants(concepts, lang)
    mx = max(c.score for c in concepts)
    for c in concepts:
        c.importance = round(min(1.0, c.score / mx), 3) if mx else 0.0
    concepts = [c for c in concepts if c.importance >= 0.3 or stats[c.stems]["defs"] or stats[c.stems]["emph"] or stats[c.stems]["head"] or c.section_topic]
    if not concepts:
        return [], [], sents

    # --- mention index ---
    sent_index: list[dict[str, list[int]]] = []
    for s in sents:
        idx: dict[str, list[int]] = defaultdict(list)
        for i, t in enumerate(s.stems):
            idx[t].append(i)
        sent_index.append(idx)
    phrase_variants: dict[str, list[tuple[str, ...]]] = {}
    for c in concepts:
        variants = [c.stems]
        for a in c.aliases:
            av = tuple(f for w in WORD_RE.findall(a) for f in _flat_stems(w, lang))
            if av:
                variants.append(av)
        phrase_variants[c.id] = variants
    first_token_map: dict[str, list[Concept]] = defaultdict(list)
    for c in concepts:
        for v in phrase_variants[c.id]:
            first_token_map[v[0]].append(c)
    mentions_by_concept: dict[str, list[int]] = defaultdict(list)
    sent_concepts: list[set[str]] = [set() for _ in sents]
    for s in sents:
        idx = sent_index[s.idx]
        for t in idx:
            for c in first_token_map.get(t, []):
                for v in phrase_variants[c.id]:
                    if v[0] != t:
                        continue
                    if _phrase_positions(s.stems, v, idx):
                        mentions_by_concept[c.id].append(s.idx)
                        sent_concepts[s.idx].add(c.id)
                        break
    by_id = {c.id: c for c in concepts}
    for c in concepts:
        ms0 = mentions_by_concept.get(c.id)
        c.first_sent = min(ms0) if ms0 else 10**9
    concepts = [c for c in concepts if c.first_sent < 10**9 or c.count > 0]
    concepts.sort(key=lambda c: (c.first_sent, -c.score))
    by_id = {c.id: c for c in concepts}

    # --- definition / examples / formulas / facts ---
    sec_math: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
    for sec in sections:
        if sec.skip:
            continue
        for bi, b in enumerate(sec.blocks):
            if b.kind == "math" and 3 < len(b.text) < 140 and not b.meta.get("code"):
                sec_math[sec.id].append((bi, b.text, b.page))
    def_spans: dict[int, list[tuple[int, int, str]]] = {s.idx: _definitional_candidates(s, lang) for s in sents}
    for c in concepts:
        ms = mentions_by_concept.get(c.id, [])
        c.count = max(c.count, len(ms))
        c.pages = sorted({sents[i].page for i in ms}) or []
        best, best_sc = None, -1.0
        for i in ms:
            s = sents[i]
            nwords = len(s.words)
            if nwords < 5 or nwords > 60 or s.text.rstrip().endswith("?"):
                continue
            sc = 0.0
            for (a, b, _k) in def_spans.get(i, []):
                if tuple(sum((_flat_stems(w, lang) for w in s.words[a:b]), [])) == c.stems or tuple(sum((_flat_stems(w, lang) for w in s.words[a:b]), [])) in phrase_variants[c.id]:
                    sc += 4.0
            if s.first_in_block:
                sc += 0.8
            if s.section_id == c.section_id:
                sc += 1.0
            pos = _phrase_positions(s.stems, c.stems, sent_index[i])
            if pos and pos[0] <= 6:
                sc += 0.8
            if 8 <= nwords <= 35:
                sc += 0.6
            sc -= 0.01 * s.idx / max(len(sents), 1)
            if sc > best_sc:
                best, best_sc = s, sc
        if best is not None:
            c.definition = best.text.strip()
            c.definition_quality = "defined" if best_sc >= 4.0 else "context"
            c.definition_page = best.page
            if not c.section_topic and best_sc >= 4.0:
                c.section_id = best.section_id
        # examples
        ex: list[str] = []
        for i in ms:
            s = sents[i]
            low = s.text.lower()
            if any(cue in low for cue in EXAMPLE_CUES) and s.text != c.definition and 25 < len(s.text) < 300:
                ex.append(s.text.strip())
            if len(ex) >= 2:
                break
        if len(ex) < 2 and best is not None:
            nxt = best.idx + 1
            if nxt < len(sents) and sents[nxt].block_idx == best.block_idx and sents[nxt].section_id == best.section_id:
                if any(cue in sents[nxt].text.lower() for cue in EXAMPLE_CUES) and sents[nxt].text not in ex:
                    ex.append(sents[nxt].text.strip())
        c.examples = ex[:2]
        # formulas: math blocks close to the best mention
        if best is not None:
            fm = [t for (bi, t, pg) in sec_math.get(best.section_id, []) if abs(bi - best.block_idx) <= 2 and pg == best.page]
            c.formulas = fm[:2]
        # key facts: richest sentences
        cand = []
        for i in ms:
            s = sents[i]
            if best is not None and s.idx == best.idx:
                continue
            if 8 <= len(s.words) <= 45:
                cand.append((len(sent_concepts[i]), -s.idx, s.text.strip()))
        cand.sort(reverse=True)
        c.facts = [t for _, _, t in cand[:3]]
        c.mentions = []
        seen_pages: set[int] = set()
        for i in ms:
            s = sents[i]
            if s.page in seen_pages and len(c.mentions) >= 3:
                continue
            seen_pages.add(s.page)
            c.mentions.append({"section_id": s.section_id, "page": s.page, "snippet": s.text[:220]})
            if len(c.mentions) >= 6:
                break

    # a concept that merely repeats its section title and is not actually defined adds nothing
    concepts = [c for c in concepts if not (c.section_topic and not c.definition)]
    by_id = {c.id: c for c in concepts}
    for ids_ in sent_concepts:
        ids_.intersection_update(by_id)

    # --- relations ---
    rel: list[Relation] = []
    prereq_pairs: set[tuple[str, str]] = set()
    for b in concepts:
        if not b.definition:
            continue
        di = next((i for i in mentions_by_concept.get(b.id, []) if sents[i].text.strip() == b.definition), None)
        if di is None or len(sent_concepts[di]) > 5:
            continue  # list-like sentences ("CNNs, RNNs and LSTMs") do not express dependency
        ds = sents[di]
        bpos = _phrase_positions(ds.stems, b.stems, sent_index[di])
        if not bpos:
            continue
        b_end = bpos[0] + len(b.stems)
        cands: list[tuple[float, Concept]] = []
        for aid in sent_concepts[di]:
            if aid == b.id:
                continue
            a = by_id[aid]
            if a.first_sent >= b.first_sent:
                continue
            if any(b.stems[i : i + len(a.stems)] == a.stems for i in range(max(len(b.stems) - len(a.stems) + 1, 0))):
                continue  # A is just a part of B's name
            apos = min((p for v in phrase_variants[a.id] for p in _phrase_positions(ds.stems, v, sent_index[di])), default=-1)
            if apos < b_end:
                continue  # must appear in the explanatory part, not the subject
            if a.importance < 0.25 and a.definition_quality != "defined":
                continue
            if len(a.stems) == 1 and a.definition_quality != "defined" and not a.label.isupper():
                continue
            w = 1.0 + (0.6 if a.definition_quality == "defined" else 0.0) + 0.3 * a.importance
            cands.append((w, a))
        cands.sort(key=lambda x: -x[0])
        for w, a in cands[:3]:
            if len(mentions_by_concept.get(a.id, [])) > 0.2 * len(sents):
                continue  # a document-wide topic word, not a prerequisite of anything specific
            rel.append(Relation(a.id, b.id, "prereq", round(w, 2)))
            prereq_pairs.add((a.id, b.id))
    co: Counter[tuple[str, str]] = Counter()
    for ids in sent_concepts:
        if 2 <= len(ids) <= 5:
            l = sorted(ids)
            for x in range(len(l)):
                for y in range(x + 1, len(l)):
                    co[(l[x], l[y])] += 1
    deg: Counter[str] = Counter()
    for (x, y), n in sorted(co.items(), key=lambda kv: -kv[1]):
        if n < 2 or (x, y) in prereq_pairs or (y, x) in prereq_pairs:
            continue
        if deg[x] >= 3 or deg[y] >= 3:
            continue
        a, b = by_id[x], by_id[y]
        if min(a.importance, b.importance) < 0.2:
            continue
        if any(a.stems[i : i + len(b.stems)] == b.stems for i in range(max(len(a.stems) - len(b.stems) + 1, 0))) or any(
            b.stems[i : i + len(a.stems)] == a.stems for i in range(max(len(b.stems) - len(a.stems) + 1, 0))
        ):
            continue
        rel.append(Relation(x, y, "related", float(n)))
        deg[x] += 1
        deg[y] += 1
    _ = section_title_keys
    # PDF / formula debris such as "Rank rank rank" or "Kinetic energy energy due" is never a concept
    stutter = {c.id for c in concepts if any(a == b for a, b in zip(c.label.lower().split(), c.label.lower().split()[1:]))}
    if stutter:
        concepts = [c for c in concepts if c.id not in stutter]
        rel = [r for r in rel if r.source not in stutter and r.target not in stutter]
    return concepts, rel, sents


def _merge_acronym_variants(concepts: list[Concept], lang: str) -> list[Concept]:
    """Merge ``LTI system`` into ``Linear time-invariant system`` (acronym = initials of the long form)."""
    by_stems = {c.stems: c for c in concepts}
    removed: set[str] = set()
    for c in list(concepts):
        if not c.stems or len(c.stems[0]) < 2 or len(c.stems[0]) > 6:
            continue
        head = c.label.split()[0] if c.label else ""
        if not (head.isupper() and head.isalpha()):
            continue
        acr, rest = c.stems[0], c.stems[1:]
        for d in concepts:
            if d.id == c.id or d.id in removed or len(d.stems) < len(rest) + 2:
                continue
            tail_ok = d.stems[len(d.stems) - len(rest):] == rest if rest else True
            lead = d.stems[: len(d.stems) - len(rest)]
            if not tail_ok or not lead:
                continue
            initials = "".join(p[0] for p in lead)
            if initials == acr:
                d.aliases = sorted(set(d.aliases) | {c.label})
                d.count += c.count
                d.score = max(d.score, c.score)
                d.sections = list(dict.fromkeys(d.sections + c.sections))
                removed.add(c.id)
                _ = by_stems
                break
    return [c for c in concepts if c.id not in removed]


# ---------------------------------------------------------------------------
def summarize_sentences(texts: list[str], k: int = 2) -> list[str]:
    """TextRank-lite extractive summary (centrality over TF-IDF cosine); keeps document order."""
    import re as _re

    def _clean(t: str) -> str:
        t = _re.sub(r"(?:\s*,){2,}", "", t)
        t = _re.sub(r"\s+([,.;:])", r"\1", t)
        return _re.sub(r"\s+", " ", t).strip()

    texts = [_clean(t) for t in texts]
    texts = [t for t in texts if 25 <= len(t) <= 420]
    # linking sentences ("Additionally, ...", "However, ...") and fragments are poor summary material
    weak = _re.compile(r"^(additionally|moreover|furthermore|however|besides|also|in addition|finally|thus|hence|therefore|similarly)\b", _re.IGNORECASE)
    good = [t for t in texts if not weak.match(t) and t[:1].isupper()]
    if len(good) >= k:
        texts = good
    if len(texts) <= k:
        return texts
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        from indiclearn.studio.lang import index_tokens

        vec = TfidfVectorizer(analyzer=lambda t: index_tokens(t))
        m = vec.fit_transform(texts)
        sim = cosine_similarity(m)
        for i in range(len(texts)):
            sim[i, i] = 0.0
        n = len(texts)
        r = [1.0 / n] * n
        for _ in range(20):
            nr = []
            for i in range(n):
                acc = 0.0
                for j in range(n):
                    s = sim[j].sum()
                    if s > 0:
                        acc += sim[j, i] / s * r[j]
                nr.append(0.15 / n + 0.85 * acc)
            r = nr
        # slight bias to early sentences (topic sentences)
        scores = [r[i] * (1.0 + 0.25 / (1 + i)) for i in range(n)]
        top = sorted(sorted(range(n), key=lambda i: -scores[i])[:k])
        return [texts[i] for i in top]
    except Exception:
        return texts[:k]


def block_text(b: Block) -> str:
    return b.text
