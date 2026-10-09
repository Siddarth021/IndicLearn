"""Hybrid, script-aware retrieval over one document + concept linking.

Channels
--------
* **BM25** over Unicode-safe word tokens (+ light stems) - precise for exact terminology.
* **Char n-gram TF-IDF** (``char_wb`` 3-5) - robust to Indic inflection, spelling variants and
  Telugu/Hindi agglutination.
* **Phonetic skeleton concept linking** - maps ``కన్వల్యూషన్`` / ``कन्वोल्यूशन`` / ``convolution`` to the
  same concept so a Telugu question can find an English passage (and vice-versa).
* **Dense embeddings** (optional, local ``multilingual-e5-small``) - fused via reciprocal rank fusion.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from indiclearn.studio.lang import ALL_STOP, WORD_RE, index_tokens, normalize, stem, tokenize_lower
from indiclearn.studio.phonetic import phonetic_similarity, skeleton


@dataclass
class Hit:
    chunk: dict[str, Any]
    score: float
    relevance: float  # 0..1 absolute-ish sufficiency signal
    channels: dict[str, float] = field(default_factory=dict)


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = (sum(self.len) / len(self.len)) if docs else 0.0
        df: Counter[str] = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, q: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tf))
        if not self.tf:
            return out
        for t in set(q):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, c in enumerate(self.tf):
                f = c.get(t, 0)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / (self.avg or 1.0)))
        return out


class DocIndex:
    """In-memory index for one document (cheap to rebuild from the chunks table)."""

    def __init__(self, chunks: list[dict[str, Any]], nodes: list[dict[str, Any]], lang: str = "en", vecs: np.ndarray | None = None, embedder: Any = None) -> None:
        self.chunks = chunks
        self.lang = lang
        self.nodes = nodes
        self.vecs = vecs
        self.embedder = embedder
        self._tokens = [index_tokens(c["text"], lang) for c in chunks]
        self._chunk_stem_sets = [set(t) for t in self._tokens]
        self.bm25 = BM25(self._tokens)
        self._tfidf = None
        self._tfidf_mat = None
        if chunks:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, lowercase=True, min_df=1)
            self._tfidf_mat = self._tfidf.fit_transform([normalize(c["text"]) for c in chunks])
        # concept phrase index (labels + aliases) for linking
        self.concept_nodes = [n for n in nodes if n["type"] in ("concept",) or n.get("is_term") or n["type"] in ("topic", "section", "chapter")]
        self._phrases: list[tuple[str, dict[str, Any], tuple[str, ...], str]] = []
        for n in self.concept_nodes:
            names = [n["full_label"]] + list(n.get("aliases") or [])
            lab = re.sub(r"\s*\([^)]*\)\s*$", "", n["full_label"]).strip()
            if lab and lab not in names:
                names.append(lab)
            for nm in names:
                toks = tuple(stem(w, lang) for w in tokenize_lower(nm) if w)
                if toks:
                    self._phrases.append((nm, n, toks, skeleton(nm)))
        # chunks mentioning each concept (phrase match on stems)
        self.concept_chunks: dict[str, list[int]] = {}
        for nm, n, toks, _sk in self._phrases:
            for ci, tset in enumerate(self._tokens):
                if len(toks) == 1:
                    ok = toks[0] in tset
                else:
                    ok = all(t in tset for t in toks) and self._contains_phrase(ci, toks)
                if ok:
                    self.concept_chunks.setdefault(n["id"], [])
                    if ci not in self.concept_chunks[n["id"]]:
                        self.concept_chunks[n["id"]].append(ci)

    def _contains_phrase(self, ci: int, toks: tuple[str, ...]) -> bool:
        seq = [stem(w, self.lang) for w in tokenize_lower(self.chunks[ci]["text"])]
        n = len(toks)
        return any(tuple(seq[i : i + n]) == toks for i in range(len(seq) - n + 1))

    # ------------------------------------------------------------------ linking
    def link_concepts(self, query: str, max_links: int = 4) -> list[dict[str, Any]]:
        """Map a free-text question to concept nodes (lexical + phonetic)."""
        q = normalize(query)
        words = tokenize_lower(q)
        qstems = [stem(w, self.lang) for w in words]
        qstems_en = [stem(w) for w in words]
        scored: dict[str, tuple[float, str, dict[str, Any]]] = {}
        for nm, n, toks, sk in self._phrases:
            n_t = len(toks)
            best = 0.0
            how = ""
            # exact (stem) phrase match inside the question
            for seq in (qstems, qstems_en):
                for i in range(len(seq) - n_t + 1):
                    if tuple(seq[i : i + n_t]) == toks:
                        sc = 0.75 + 0.1 * min(n_t, 3)
                        if sc > best:
                            best, how = sc, "exact"
            # phonetic match against windows of the question (cross-script)
            if best < 0.8 and len(sk) >= 3:
                phrase_words = [w for w in re.split(r"[\s\-]+", nm) if w]
                for w in range(1, min(len(phrase_words) + 1, 4) + 1):
                    for i in range(len(words) - w + 1):
                        win = [x for x in words[i : i + w]]
                        if win[0] in ALL_STOP or win[-1] in ALL_STOP:
                            continue
                        win_words = [p for x in win for p in re.split(r"-", x) if p]
                        wsk = skeleton(" ".join(win_words))
                        if len(wsk) < 3 or abs(len(wsk) - len(sk)) > 2:
                            continue
                        if len(win_words) == len(phrase_words):
                            sims = []
                            for a_, b_ in zip(win_words, phrase_words):
                                sa_, sb_ = skeleton(a_), skeleton(b_)
                                if min(len(sa_), len(sb_)) < 3:
                                    sims.append(1.0 if sa_ == sb_ else 0.0)
                                else:
                                    sims.append(phonetic_similarity(a_, b_))
                            ok = min(sims) >= 0.75 and sum(sims) / len(sims) >= 0.88
                            sim = sum(sims) / len(sims)
                        else:
                            sim = phonetic_similarity(" ".join(win_words), " ".join(phrase_words))
                            ok = sim >= 0.9
                        if ok:
                            sc = 0.5 + 0.25 * min(1.0, (sim - 0.84) / 0.16) + 0.04 * min(w, 3)
                            if sc > best:
                                best, how = sc, "phonetic"
            # partial: all but one token of a multi-word concept
            if best == 0.0 and n_t >= 2:
                hit = sum(1 for t in toks if t in qstems or t in qstems_en)
                if hit >= max(2, n_t - 1):
                    best, how = 0.5, "partial"
            if best > 0:
                best *= 0.85 + 0.15 * float(n.get("importance", 0.5))
                if n["type"] in ("chapter", "section"):
                    best *= 0.8
                prev = scored.get(n["id"])
                if prev is None or best > prev[0]:
                    scored[n["id"]] = (best, how, n)
        ranked = sorted(scored.values(), key=lambda x: -x[0])
        # prefer the most specific (longest) concept when one contains another
        out: list[dict[str, Any]] = []
        for sc, how, n in ranked:
            out.append({"node": n, "score": round(sc, 3), "how": how})
            if len(out) >= max_links:
                break
        return out

    # ------------------------------------------------------------------ search
    def search(self, query: str, k: int = 5, links: list[dict[str, Any]] | None = None) -> list[Hit]:
        if not self.chunks:
            return []
        q = normalize(query)
        qt = index_tokens(q, self.lang)
        n = len(self.chunks)
        bm = self.bm25.scores(qt)
        bm_n = bm / (bm.max() or 1.0) if bm.max() > 0 else bm
        cos = np.zeros(n)
        if self._tfidf is not None and q:
            qv = self._tfidf.transform([q])
            cos = (self._tfidf_mat @ qv.T).toarray().ravel()
        dense = np.zeros(n)
        if self.vecs is not None and self.embedder is not None:
            try:
                qe = self.embedder.encode_query(q)
                dense = self.vecs @ qe
            except Exception:  # pragma: no cover - model failure must not break chat
                dense = np.zeros(n)
        # coverage of informative query tokens
        qset = {t for t in qt if t not in ALL_STOP}
        cov = np.array([len(qset & ts) / len(qset) if qset else 0.0 for ts in self._chunk_stem_sets])
        boost = np.zeros(n)
        for lk in links or []:
            node = lk["node"]
            definition = (node.get("definition") or "")[:50]
            for ci in self.concept_chunks.get(node["id"], []):
                b = lk["score"]
                if self.chunks[ci].get("section_id") == node.get("section_id"):
                    b += 0.15
                if definition and definition in self.chunks[ci]["text"]:
                    b += 0.25
                boost[ci] = max(boost[ci], b)
        boost = np.clip(boost, 0, 1.2)
        # reciprocal-rank fusion over channels
        def ranks(v: np.ndarray) -> np.ndarray:
            order = np.argsort(-v)
            r = np.empty(n)
            r[order] = np.arange(1, n + 1)
            return r
        fused = np.zeros(n)
        for vec, w in ((bm, 1.0), (cos, 1.0), (dense, 1.2 if dense.any() else 0.0), (boost, 1.3 if boost.any() else 0.0)):
            if w and vec.any():
                fused += w / (60.0 + ranks(vec))
        fused = fused / (fused.max() or 1.0)
        relevance = np.clip(0.45 * cov + 0.35 * np.clip(cos * 2.5, 0, 1) + 0.2 * bm_n, 0, 1)
        if dense.any():
            relevance = np.maximum(relevance, np.clip((dense - 0.72) / 0.2, 0, 1) * 0.9)
        relevance = np.maximum(relevance, boost * 0.85)
        order = np.argsort(-(fused + 0.5 * relevance))[: max(k, 1)]
        hits = []
        for i in order:
            if fused[i] <= 0 and relevance[i] <= 0:
                continue
            hits.append(Hit(self.chunks[int(i)], float(fused[i]), float(relevance[i]), {"bm25": float(bm_n[i]), "char_tfidf": float(cos[i]), "dense": float(dense[i]), "concept": float(boost[i])}))
        return hits


# ---------------------------------------------------------------------------
def best_sentences(text: str, query: str, lang: str, k: int = 2, avoid: set[str] | None = None) -> list[str]:
    """Pick the k sentences of *text* that best overlap the question (stable order)."""
    from indiclearn.studio.lang import split_sentences

    sents = [s for s in split_sentences(text) if len(s) > 20]
    if not sents:
        return [text[:300]]
    qt = {t for t in index_tokens(query, lang) if t not in ALL_STOP}
    scored = []
    for i, s in enumerate(sents):
        if avoid and s in avoid:
            continue
        st = set(index_tokens(s, lang))
        ov = len(qt & st)
        scored.append((ov + 0.15 / (1 + i), i, s))
    scored.sort(reverse=True)
    top = sorted(scored[:k], key=lambda x: x[1])
    return [s for _, _, s in top] or sents[:k]


_WORD = WORD_RE
