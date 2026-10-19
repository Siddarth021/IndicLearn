"""The study tutor: grounded, citation-bearing answers in en / te / hi / sa.

Pipeline:  detect question language -> intent -> link to concepts -> hybrid retrieval ->
sufficiency check -> compose answer (definition, evidence sentences, example, related, prerequisites)
-> return the *graph delta* (evidence + doubt node) for the personal map.

The default composer is extractive (no generation), so every sentence of content is traceable to the
student's own notes.  A local LLM, when configured, may only rephrase that grounded text.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from indiclearn.studio.i18n import detect_intent, tr
from indiclearn.studio.lang import LANG_NAMES, detect_language, split_sentences
from indiclearn.studio.retrieval import DocIndex, best_sentences

log = logging.getLogger("indiclearn.tutor")

SUFFICIENT = 0.3
# words that make a short question a follow-up about "the thing we were just discussing"
_DEMONSTRATIVE = re.compile(r"(?i)\b(this|that|it|these|those|more|why|how|again|explain|elaborate)\b|ఇది|దీని|దీన్ని|ఇంకా|ఎందుకు|यह|इस|इसे|और|क्यों|कैसे|एतत्|अस्य|कथम्|किमर्थम्")


@dataclass
class TutorReply:
    text: str
    intent: str = "explain"
    lang: str = "en"
    grounded: bool = True
    relevance: float = 0.0
    citations: list[dict[str, Any]] = field(default_factory=list)
    linked: list[dict[str, Any]] = field(default_factory=list)  # [{id,label,score,how}]
    evidence: list[tuple[str, str]] = field(default_factory=list)  # (node_id, kind)
    doubt: dict[str, Any] | None = None  # {label, parent_node_id, covered}
    focus_node: str | None = None
    action: str | None = None  # "quiz" | "recommend"
    related: list[dict[str, str]] = field(default_factory=list)
    # generative extras (empty on the extractive path): reasoning, key_terms, follow_ups, check, beyond_notes, engine
    extra: dict[str, Any] = field(default_factory=dict)


def pick_reply_lang(question: str, ui_lang: str, auto: bool = True) -> str:
    """Reply in the language the student wrote in (if confidently one of ours), else the UI language."""
    if not auto:
        return ui_lang
    d = detect_language(question)
    lang = d.code.split("-")[0]
    if lang in ("hi", "sa") and ui_lang in ("hi", "sa"):
        return ui_lang  # Devanagari text is ambiguous between Hindi and Sanskrit: trust the interface language
    if lang in LANG_NAMES and (d.confidence >= 0.5 or lang != "en") and len(question.split()) >= 2:
        return lang
    return ui_lang


def _clean_label(q: str) -> str:
    q = re.sub(r"[?？।!.]+$", "", q.strip())
    q = re.sub(r"\s+", " ", q)
    return (q[:60] + "…") if len(q) > 60 else q


def _neighbours(node_id: str, edges: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    prereq, related, nxt = [], [], []
    for e in edges:
        if e["type"] == "prereq" and e["target"] == node_id and e["source"] in by_id:
            prereq.append(by_id[e["source"]])
        elif e["type"] == "related" and node_id in (e["source"], e["target"]):
            o = e["target"] if e["source"] == node_id else e["source"]
            if o in by_id:
                related.append(by_id[o])
        elif e["type"] == "next" and e["source"] == node_id and e["target"] in by_id:
            nxt.append(by_id[e["target"]])
    return prereq[:3], related[:4], nxt[:1]


def _cite(chunk: dict[str, Any], node_id: str | None, snippet: str) -> dict[str, Any]:
    return {"chunk_id": chunk["id"], "page": chunk.get("page", 1), "section_id": chunk.get("section_id"), "node_id": node_id, "snippet": snippet[:240]}


def answer(
    question: str, ui_lang: str, index: DocIndex, nodes: list[dict[str, Any]], edges: list[dict[str, Any]], title: str, doc_lang: str,
    unit_label: str = "Page", focus_node: str | None = None, auto_lang: bool = True, models: Any = None,
    history: list[dict[str, str]] | None = None, on_delta: Any = None,
) -> TutorReply:
    """Hybrid tutor: a generative explanation when an LLM is reachable, the extractive one otherwise.

    on_delta(text_so_far | None) receives the answer while it is being written (streaming UI)."""
    brain = getattr(models, "brain", None)
    if brain is not None and getattr(brain, "available", False):
        try:
            rep = _answer_with_brain(question, ui_lang, index, nodes, edges, title, doc_lang, unit_label, focus_node, auto_lang, brain, history or [], on_delta)
        except Exception:  # pragma: no cover - the LLM path must never take the chat down
            log.exception("brain tutor failed - falling back to extractive")
            rep = None
        if rep is not None:
            return rep
    rep = _answer_extractive(question, ui_lang, index, nodes, edges, title, doc_lang, unit_label, focus_node, auto_lang, models)
    if brain is not None and rep.action is None and rep.intent != "greet":
        # Say WHY the smart answer is missing instead of silently quoting the notes.
        rep.extra["notice"] = brain.why_unavailable() if hasattr(brain, "why_unavailable") else ""
    return rep


def _answer_extractive(
    question: str, ui_lang: str, index: DocIndex, nodes: list[dict[str, Any]], edges: list[dict[str, Any]], title: str, doc_lang: str,
    unit_label: str = "Page", focus_node: str | None = None, auto_lang: bool = True, models: Any = None,
) -> TutorReply:
    lang = pick_reply_lang(question, ui_lang, auto_lang)
    by_id = {n["id"]: n for n in nodes}
    intents = detect_intent(question)
    unit = tr(lang, f"unit_{unit_label}") if unit_label in ("Page", "Slide", "Part") else tr(lang, "page")

    if "greet" in intents and not any(i in intents for i in ("quiz", "summary", "recommend")):
        return TutorReply(tr(lang, "greeting", title=title), "greet", lang)
    if "quiz" in intents:
        links = index.link_concepts(question)
        node = links[0]["node"] if links else by_id.get(focus_node or "")
        return TutorReply("", "quiz", lang, linked=[_l(x) for x in links], focus_node=node["id"] if node else None, action="quiz")
    if "recommend" in intents:
        return TutorReply("", "recommend", lang, action="recommend")

    links = index.link_concepts(question)
    # Offline cross-lingual search: a Telugu question on English notes is also searched in English
    qlang = detect_language(question).base
    tq = None
    if qlang != doc_lang and hasattr(models, "translate_query"):
        tq = models.translate_query(question, qlang, doc_lang)
        if tq:
            links = _merge_links(links, index.link_concepts(tq))
    follow_up = any(i in intents for i in ("example", "simple", "summary")) or bool(_DEMONSTRATIVE.search(question))
    if not links and focus_node and focus_node in by_id and len(question.split()) <= 8 and follow_up:
        # follow-up like "give an example" - stay on the node the student is viewing
        n = by_id[focus_node]
        links = [{"node": n, "score": 0.6, "how": "focus"}]
    hits = index.search(question, k=4, links=links)
    if tq:
        hits = _merge_hits(hits, index.search(tq, k=4, links=links))[:4]
    top_rel = hits[0].relevance if hits else 0.0
    linked_ok = bool(links) and links[0]["score"] >= 0.55
    grounded = top_rel >= SUFFICIENT or linked_ok
    reply = TutorReply("", "explain", lang, grounded=grounded, relevance=round(min(top_rel, 1.0), 3), linked=[_l(x) for x in links])

    if not grounded:
        label = _clean_label(question)
        parent = None
        if hits and hits[0].relevance >= 0.15:
            parent = hits[0].chunk.get("section_id")
        reply.text = tr(lang, "not_found")
        if hits and hits[0].relevance > 0.08:
            sn = best_sentences(hits[0].chunk["text"], question, doc_lang, 1)[0]
            reply.text += f"\n\n_{tr(lang, 'closest')} ({unit} {hits[0].chunk.get('page', 1)}):_ “{sn}”"
            reply.citations.append(_cite(hits[0].chunk, None, sn))
        reply.doubt = {"label": label, "parent_node_id": parent if parent in by_id else None, "covered": False}
        reply.intent = "unanswered"
        return reply

    primary = links[0]["node"] if links else None
    if primary is not None and primary.get("definition") and re.match(r"(?i)\s*(this|these|that|those|it|they|such)\b", primary["definition"]):
        primary = {**primary, "definition": None}
    if primary is None and hits:
        sec = hits[0].chunk.get("section_id")
        primary = by_id.get(sec)
    reply.focus_node = primary["id"] if primary else None
    parts: list[str] = []
    used: set[str] = set()

    if primary is None and hits:
        primary = by_id.get(hits[0].chunk.get("section_id"))
    if "summary" in intents and primary is not None:
        reply.intent = "summary"
        parts.append(f"**{tr(lang, 'summary_title')} — {primary['full_label']}**")
        if primary.get("summary"):
            parts.append(primary["summary"])
        if primary.get("definition") and primary["definition"] != primary.get("summary"):
            parts.append(primary["definition"])
        kids = [n for n in nodes if n["parent"] == primary["id"] and n["type"] in ("concept", "topic", "section")]
        if kids:
            parts.append(f"**{tr(lang, 'key_points')}**\n" + "\n".join(_kp(k) for k in kids[:6]))
        for h in hits[:2]:
            used.add(h.chunk["id"])
            reply.citations.append(_cite(h.chunk, primary["id"], h.chunk["text"]))
    else:
        if primary is not None and primary.get("definition") and "example" not in intents:
            parts.append(f"**{tr(lang, 'definition')} — {primary['full_label']}**\n{primary['definition']}")
        if "example" in intents and primary is not None and primary.get("examples"):
            parts.append(f"**{tr(lang, 'example')}**\n" + "\n".join(f"• {e}" for e in primary["examples"][:2]))
        avoid = set()
        if primary is not None:
            avoid = {primary.get("definition") or ""} | set(primary.get("examples") or [])
            on_topic = {index.chunks[ci]["id"] for ci in index.concept_chunks.get(primary["id"], [])}
        else:
            on_topic = set()
        for hi, h in enumerate(hits[:3]):
            if hi and (h.relevance < max(SUFFICIENT, 0.6 * top_rel) or (on_topic and h.chunk["id"] not in on_topic)):
                continue
            sents = best_sentences(h.chunk["text"], question + " " + (primary["full_label"] if primary else ""), doc_lang, 2, avoid=avoid)
            sents = [s for s in sents if s not in avoid]
            body = " ".join(sents)
            if primary and primary.get("definition") and body.strip() in primary["definition"]:
                continue
            if body and body not in used:
                used.add(body)
                parts.append(f"**{tr(lang, 'from_notes')} ({unit} {h.chunk.get('page', 1)})**\n{body}")
                reply.citations.append(_cite(h.chunk, primary["id"] if primary else None, body))
        if primary is not None and primary.get("formulas") and ("formula" in question.lower() or "సూత్ర" in question or "सूत्र" in question or "example" not in intents):
            parts.append(f"**{tr(lang, 'formula')}**\n" + "\n".join(f"`{f}`" for f in primary["formulas"][:2]))
        if "example" in intents and primary is not None and not primary.get("examples"):
            ex = [s for h in hits for s in split_sentences(h.chunk["text"]) if re.search(r"example|for instance|e\.g\.|ఉదాహరణ|उदाहरण", s, re.IGNORECASE)]
            if ex:
                parts.append(f"**{tr(lang, 'example')}**\n{ex[0]}")
    if "simple" in intents and parts:
        parts.insert(0, f"_{tr(lang, 'simple')}:_")

    if primary is not None:
        prereq, related, nxt = _neighbours(primary["id"], edges, by_id)
        if prereq:
            parts.append(f"**{tr(lang, 'review_first')}:** " + ", ".join(p["label"] for p in prereq))
        if related:
            parts.append(f"**{tr(lang, 'related')}:** " + ", ".join(r["label"] for r in related))
            reply.related = [{"id": r["id"], "label": r["label"]} for r in related]
        if nxt:
            parts.append(f"**{tr(lang, 'next_up')}:** {nxt[0]['label']}")

    if not parts:
        reply.grounded = False
        reply.text = tr(lang, "not_found")
        reply.doubt = {"label": _clean_label(question), "parent_node_id": None, "covered": False}
        return reply

    text = "\n\n".join(parts)
    if lang != doc_lang:
        note = tr(lang, "src_lang_note", lang=LANG_NAMES.get(doc_lang, doc_lang))
        tl = getattr(models, "translator", None)
        if tl is not None:
            try:
                text = "\n\n".join(p if p.startswith("**") and "\n" not in p else tl.translate(p, doc_lang, lang) for p in parts)
                note = tr(lang, "translated_note", lang=LANG_NAMES.get(doc_lang, doc_lang))
            except Exception:  # pragma: no cover
                pass
        text += f"\n\n_{note}_"
    llm = getattr(models, "llm", None)
    if llm is not None:
        out = llm.rephrase(question, "\n".join(c["snippet"] for c in reply.citations) or text, lang)
        if out:
            text = out + "\n\n" + "\n".join(f"_[{unit} {c['page']}]_" for c in reply.citations[:3])
    reply.text = text
    ev_nodes = [x["node"]["id"] for x in links[:2]] or ([primary["id"]] if primary else [])
    reply.evidence = [(nid, "asked") for nid in ev_nodes]
    return reply


def _kp(k: dict[str, Any]) -> str:
    d = (k.get("definition") or "").strip()
    if not d:
        return f"• {k['label']}"
    d = d if len(d) <= 150 else d[:147].rsplit(" ", 1)[0] + "…"
    return f"• {d}" if d.lower().startswith(k["label"].lower()[:12]) else f"• {k['label']} — {d}"


def _l(x: dict[str, Any]) -> dict[str, Any]:
    return {"id": x["node"]["id"], "label": x["node"]["label"], "score": round(float(x["score"]), 3), "how": x.get("how", "")}


# --------------------------------------------------------------------------- generative path
_PAGE_TAG = re.compile(r"\[(?:p\.|page |slide |part )\s*(\d+)\]", re.IGNORECASE)
_UNIT_TAG = {"Page": "p.", "Slide": "slide ", "Part": "part "}
_BRAIN_INTENTS = {"explain", "example", "simple", "summary", "compare", "formula", "exam", "other"}


def _merge_links(*lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for lst in lists:
        for x in lst:
            nid = x["node"]["id"]
            if nid not in best or x["score"] > best[nid]["score"]:
                best[nid] = x
    return sorted(best.values(), key=lambda x: -x["score"])[:4]


def _merge_hits(*lists: list[Any]) -> list[Any]:
    best: dict[str, Any] = {}
    for lst in lists:
        for h in lst:
            cid = h.chunk["id"]
            if cid not in best or h.relevance > best[cid].relevance:
                best[cid] = h
    return sorted(best.values(), key=lambda h: -(h.relevance + 0.5 * h.score))


_LANG_WORDS = {"te": r"telugu|తెలుగు", "hi": r"hindi|हिन्दी|हिंदी", "en": r"english|ఇంగ్లీష్|ఇంగ్లిష్|अंग्रेज़ी|अंग्रेजी", "ta": r"tamil",
               "kn": r"kannada", "ml": r"malayalam", "mr": r"marathi", "bn": r"bengali|bangla", "gu": r"gujarati", "sa": r"sanskrit|संस्कृत"}


def _asked_lang(question: str) -> str | None:
    """An explicit request for a reply language: 'in Telugu', 'telugu lo', 'తెలుగులో', 'hindi mein'."""
    q = question.lower()
    for code, words in _LANG_WORDS.items():
        for m in re.finditer(rf"(?:\b(?:in|into)\s+(?:{words})\b)|(?:(?:{words})\s*(?:lo|loki|లో|mein|me|में)\b)|(?:(?:{words})లో)", q):
            return code
    return None


def _reply_lang(u: dict[str, Any], question: str, ui_lang: str, auto: bool) -> str:
    """Which language to answer in, judged from THIS question only (not the chat history).

    * Telugu / Devanagari / other Indic script in the question  -> that language.
    * Romanized Telugu or Hindi ("convolution ante enti?")       -> the native script (what students asked for).
    * Plain English                                               -> English, even after a Telugu conversation.
    """
    if not auto:
        return ui_lang
    from indiclearn.studio.brain import TARGET_LANGS

    asked = _asked_lang(question)  # "summarise this in Telugu", "telugu lo cheppu", "hindi mein batao"
    if asked:
        return asked
    d = detect_language(question)
    if d.code not in ("en", "") and "-" not in d.code and d.code in TARGET_LANGS:
        if d.code in ("hi", "sa") and ui_lang in ("hi", "sa"):
            return ui_lang  # Devanagari is ambiguous between Hindi and Sanskrit: trust the interface language
        return d.code
    if d.code in ("te-Latn", "hi-Latn"):
        return d.base
    code = (u.get("lang") or "").split("-")[0]
    latin = not re.search(r"[^\x00-\x7F\u2018-\u201f\u2026]", question)
    if latin:
        # The model may call Latin text Indic because the conversation was; only believe it if it says Romanized.
        return code if (u.get("romanized") and code in TARGET_LANGS and code != "en") else "en"
    if code in TARGET_LANGS:
        return code
    return pick_reply_lang(question, ui_lang, True)


_REFERS_BACK = re.compile(r"(?i)\b(it|its|this|that|these|those|they|them|same|above|previous|again|more)\b|ఇది|దీని|అది|దాని|వాటి|ఇంకా|यह|इस|वह|उस|इसे|उसे")
_LOCAL_INTENTS = ("quiz", "exam", "recommend", "summary", "example", "simple", "compare", "formula")


def _understand_locally(question: str, history: list[dict[str, str]], doc_lang: str) -> dict[str, Any] | None:
    """Skip the extra 'understand' LLM call when it cannot add anything (saves ~1 s and half the free quota):
    a self-contained question in plain English, or in Telugu/Hindi script, needs no rewrite to search the notes.
    Romanized Indic and follow-ups ("explain it again") still go to the LLM, which resolves them."""
    d = detect_language(question)
    if d.code in ("te-Latn", "hi-Latn") or "-" in d.code:
        return None
    if history and _REFERS_BACK.search(question):
        return None
    native = d.code in ("te", "hi", "sa") and d.confidence >= 0.6
    plain_en = d.code == "en" and d.confidence >= 0.9 and len(question.split()) >= 3
    if not (native or plain_en):
        return None
    intents = detect_intent(question)
    intent = next((i for i in _LOCAL_INTENTS if i in intents), "explain")
    return {"lang": d.code, "romanized": False, "search_query": question, "intent": intent, "engine": "local"}


def _answer_with_brain(
    question: str, ui_lang: str, index: DocIndex, nodes: list[dict[str, Any]], edges: list[dict[str, Any]], title: str, doc_lang: str,
    unit_label: str, focus_node: str | None, auto_lang: bool, brain: Any, history: list[dict[str, str]], on_delta: Any = None,
) -> TutorReply | None:
    intents = detect_intent(question)
    if "greet" in intents and len(question.split()) <= 3:
        return None  # the canned, localised greeting is instant and fine
    u = _understand_locally(question, history, doc_lang) or brain.understand(question, history, doc_lang)
    if u is None:
        return None
    lang = _reply_lang(u, question, ui_lang, auto_lang)
    intent = (u.get("intent") or "explain").lower()
    if "exam" in intents and "quiz" not in intents:  # "what will come in the exam?" wants an answer, not a quiz or a to-do list
        intent = "exam"
    by_id = {n["id"]: n for n in nodes}
    sq = (u.get("search_query") or "").strip() or question

    links = _merge_links(index.link_concepts(question), index.link_concepts(sq))
    if intent == "quiz" or ("quiz" in intents and intent not in _BRAIN_INTENTS):
        node = links[0]["node"] if links else by_id.get(focus_node or "")
        return TutorReply("", "quiz", lang, linked=[_l(x) for x in links], focus_node=node["id"] if node else None, action="quiz")
    if intent == "recommend":
        return TutorReply("", "recommend", lang, action="recommend")

    follow_up = bool(_DEMONSTRATIVE.search(question)) or len(question.split()) <= 6
    if not links and focus_node and focus_node in by_id and follow_up:
        links = [{"node": by_id[focus_node], "score": 0.6, "how": "focus"}]
    hits = _merge_hits(index.search(sq, k=5, links=links), index.search(question, k=3, links=links))
    top_rel = hits[0].relevance if hits else 0.0
    grounded = top_rel >= SUFFICIENT or (bool(links) and links[0]["score"] >= 0.55)

    primary = links[0]["node"] if links else (by_id.get(hits[0].chunk.get("section_id")) if hits else None)
    focus_info = None
    if primary is not None:
        prereq, related, _nxt = _neighbours(primary["id"], edges, by_id)
        focus_info = {**primary, "prereq_labels": [p["label"] for p in prereq]}
    passages = [{"page": h.chunk.get("page", 1), "text": h.chunk["text"], "chunk": h.chunk} for h in hits[:5] if h.relevance >= 0.08 or grounded]
    summary_mode = intent == "summary" or "summary" in intents
    if summary_mode and primary is not None:
        # a summary needs the WHOLE topic, not just the best-matching paragraph
        order = {c["id"]: i for i, c in enumerate(index.chunks)}
        pool = [index.chunks[ci] for ci in index.concept_chunks.get(primary["id"], [])]
        pool += [c for c in index.chunks if c.get("section_id") == primary["id"]]
        pool += [h.chunk for h in hits]
        uniq = {c["id"]: c for c in pool}
        picked = sorted(uniq.values(), key=lambda c: order.get(c["id"], 0))[:8]
        if picked:
            passages = [{"page": c.get("page", 1), "text": c["text"], "chunk": c} for c in picked]
            grounded = True
    ask = question
    if intent == "exam":
        # exam prep needs the whole document, not the paragraph closest to the word "exam"
        firsts: dict[str, dict[str, Any]] = {}
        for c in index.chunks:
            firsts.setdefault(c.get("section_id") or c["id"], c)
        pool = {h.chunk["id"]: h.chunk for h in hits}
        for c in list(firsts.values())[:10]:
            pool.setdefault(c["id"], c)
        order = {c["id"]: i for i, c in enumerate(index.chunks)}
        passages = [{"page": c.get("page", 1), "text": c["text"], "chunk": c} for c in sorted(pool.values(), key=lambda c: order.get(c["id"], 0))[:10]]
        grounded = True
        ask = (question + "\n\n(The student is preparing for an exam. Using ONLY these notes: list the most important topics and why each "
               "matters, the kinds of questions likely to be asked on each (definitions, derivations, numericals, comparisons, diagrams), "
               "and a short, practical revision plan. Do not invent past-paper questions.)")
    unit = _UNIT_TAG.get(unit_label, "p.")
    t = brain.tutor(ask, lang, passages if grounded or top_rel >= 0.12 else [], focus_info, history, title, unit, summary=summary_mode,
                    on_delta=on_delta)
    if t is None:
        return None

    reply = TutorReply(t["answer"].strip(), "explain" if intent not in ("summary", "example", "simple") else intent, lang,
                       grounded=grounded, relevance=round(min(top_rel, 1.0), 3), linked=[_l(x) for x in links])
    # citations: pages the model actually tagged, else the passages it was given
    tagged = {int(m) for m in _PAGE_TAG.findall(reply.text)}
    seen: set[str] = set()
    for p in passages:
        if (not tagged or p["page"] in tagged) and p["chunk"]["id"] not in seen:
            seen.add(p["chunk"]["id"])
            reply.citations.append(_cite(p["chunk"], primary["id"] if primary else None, p["text"][:240]))
    if tagged:
        valid = {p["page"] for p in passages}
        # a page tag that points at a passage we never showed the model is not trustworthy - drop it
        reply.text = _PAGE_TAG.sub(lambda m: m.group(0) if int(m.group(1)) in valid else "", reply.text)
    reply.focus_node = primary["id"] if primary and grounded else None
    if primary is not None and grounded:
        _p, related, _n = _neighbours(primary["id"], edges, by_id)
        reply.related = [{"id": r["id"], "label": r["label"]} for r in related]
        reply.evidence = [(x["node"]["id"], "asked") for x in links[:2]] or [(primary["id"], "asked")]
    if not grounded:
        parent = hits[0].chunk.get("section_id") if hits and hits[0].relevance >= 0.15 else None
        reply.doubt = {"label": _clean_label(question), "parent_node_id": parent if parent in by_id else None, "covered": False}
        reply.intent = "beyond_notes"
    reply.extra = {
        "reasoning": t.get("reasoning") or [],
        "key_terms": t.get("key_terms") or [],
        "follow_ups": t.get("follow_ups") or [],
        "check": t.get("check"),
        "beyond_notes": t.get("beyond_notes") or ("" if grounded else tr(lang, "low_grounding")),
        "used_general_knowledge": bool(t.get("used_general_knowledge")) or not grounded,
        "engine": t.get("engine", ""),
        "understood_as": sq,
        "question_lang": u.get("lang", ""),
        "romanized": bool(u.get("romanized")),
    }
    return reply
