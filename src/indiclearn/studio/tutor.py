"""The study tutor: grounded, citation-bearing answers in en / te / hi / sa.

Pipeline:  detect question language -> intent -> link to concepts -> hybrid retrieval ->
sufficiency check -> compose answer (definition, evidence sentences, example, related, prerequisites)
-> return the *graph delta* (evidence + doubt node) for the personal map.

The default composer is extractive (no generation), so every sentence of content is traceable to the
student's own notes.  A local LLM, when configured, may only rephrase that grounded text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from indiclearn.studio.i18n import detect_intent, tr
from indiclearn.studio.lang import LANG_NAMES, detect_language, split_sentences
from indiclearn.studio.retrieval import DocIndex, best_sentences

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
    follow_up = any(i in intents for i in ("example", "simple", "summary")) or bool(_DEMONSTRATIVE.search(question))
    if not links and focus_node and focus_node in by_id and len(question.split()) <= 8 and follow_up:
        # follow-up like "give an example" - stay on the node the student is viewing
        n = by_id[focus_node]
        links = [{"node": n, "score": 0.6, "how": "focus"}]
    hits = index.search(question, k=4, links=links)
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
