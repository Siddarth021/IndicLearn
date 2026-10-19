"""Assemble the *default* knowledge graph: topic hierarchy + prerequisite flow + related links.

The default graph is deliberately **student-independent**: it is derived only from the structure and
content of the uploaded material (hierarchy, relations, flow).  Personalisation is an overlay applied
later (see :mod:`indiclearn.studio.mastery`).
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

from indiclearn.studio.concepts import (
    Concept,
    Relation,
    Sent,
    extract_concepts,
    summarize_sentences,
)
from indiclearn.studio.lang import split_sentences
from indiclearn.studio.parse import ParsedDoc, smart_title
from indiclearn.studio.structure import (
    Chunk,
    Section,
    auto_segment,
    build_sections,
    label_synthetic_sections,
    make_chunks,
)

MAX_ROOT_CHILDREN = 14


log = logging.getLogger(__name__)


@dataclass
class KnowledgeGraph:
    title: str
    language: str
    unit_label: str
    sections: list[Section]
    concepts: list[Concept]
    relations: list[Relation]
    chunks: list[Chunk]
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    path: list[str]
    summary: str
    stats: dict[str, Any] = field(default_factory=dict)


def _short_title(title: str, limit: int = 54) -> str:
    t = re.sub(r"\s+", " ", title).strip()
    if len(t) <= limit:
        return t
    for sep in (":", " — ", " – ", " - "):
        if sep in t:
            head = t.split(sep)[0].strip()
            if 6 <= len(head) <= limit:
                return head
    return t[: limit - 1].rsplit(" ", 1)[0] + "…"


def _group_flat_children(sections: list[Section], concepts: list[Concept]) -> list[Section]:
    """Slide decks / long flat documents: bundle consecutive top-level sections into chapters."""
    root = sections[0]
    kids = [c for c in root.children]
    if len(kids) <= MAX_ROOT_CHILDREN:
        return sections
    by_id = {s.id: s for s in sections}
    n_groups = math.ceil(len(kids) / 8)
    size = math.ceil(len(kids) / n_groups)
    new_children: list[str] = []
    out = [s for s in sections]
    gcount = 0
    top_by_section: dict[str, list[Concept]] = {}
    for c in concepts:
        top_by_section.setdefault(c.section_id, []).append(c)
    for gi in range(n_groups):
        part = kids[gi * size : (gi + 1) * size]
        if not part:
            continue
        gcount += 1
        pool = [c for sid in part for c in top_by_section.get(sid, [])]
        pool.sort(key=lambda c: -c.score)
        label = pool[0].label if pool else f"Part {gcount}"
        first, last = by_id[part[0]], by_id[part[-1]]
        g = Section(f"g{gcount}", smart_title(label), 1, "s0", 0, children=list(part))
        g.synthetic = True
        for pid in part:
            by_id[pid].parent = g.id
            by_id[pid].level += 1
            for sub in sections:
                if sub.parent == pid or (sub.parent and by_id.get(sub.parent) and by_id[sub.parent].parent == pid):
                    pass
        _ = (first, last)
        out.append(g)
        new_children.append(g.id)
    root.children = new_children
    # bump deeper levels consistently
    for s in out:
        if s.parent and s.parent not in ("s0",) and s.id not in kids and s.id[0] != "g":
            p = by_id.get(s.parent)
            if p is not None and s.level <= p.level:
                s.level = p.level + 1
    return out


def build_knowledge_graph(doc: ParsedDoc, max_concepts: int | None = None, use_models: bool = True) -> KnowledgeGraph:
    """use_models=False skips M1/M2/M3 (pointless when a hosted LLM rebuilds the map right after)."""
    sections = auto_segment(build_sections(doc))
    concepts, relations, sents = extract_concepts(sections, doc.language, max_concepts=max_concepts)
    # synthetic topic titles from their strongest concept
    top: dict[str, Concept] = {}
    for c in concepts:
        t = top.get(c.section_id)
        if t is None or c.score > t.score:
            top[c.section_id] = c
    label_synthetic_sections(sections, {sid: c.label for sid, c in top.items()})
    sections = _group_flat_children(sections, concepts)
    chunks = make_chunks([s for s in sections if not s.id.startswith("g")])

    sec_by_id = {s.id: s for s in sections}
    sents_by_sec: dict[str, list[Sent]] = {}
    for st in sents:
        sents_by_sec.setdefault(st.section_id, []).append(st)

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    topic_of_section: dict[str, Concept] = {c.section_topic: c for c in concepts if c.section_topic}
    concept_node_id: dict[str, str] = {}
    for c in concepts:
        concept_node_id[c.id] = c.section_topic if c.section_topic else c.id

    def live_children(sec: Section) -> list[str]:
        return [cid for cid in sec.children if cid in sec_by_id and not sec_by_id[cid].skip]

    # document summary
    abstract = next((s for s in sections if s.title.lower() in ("abstract", "overview", "introduction") and not s.skip), None)
    pool = [st.text for st in (sents_by_sec.get(abstract.id, []) if abstract else [])] or [st.text for st in sents[:60]]
    doc_summary = " ".join(summarize_sentences(pool, 3))

    root = sections[0]
    nodes.append({
        "id": "root", "label": _short_title(doc.title), "full_label": doc.title, "type": "root", "level": 0, "parent": None,
        "importance": 1.0, "definition": None, "definition_quality": None, "summary": doc_summary, "examples": [], "formulas": [],
        "facts": [], "aliases": [], "pages": [], "page": 1, "mentions": [], "section_id": "s0", "n_children": len(live_children(root)),
    })
    order_counter = 0
    for sec in sections[1:]:
        if sec.skip or sec.id.startswith("g") and not sec.children:
            continue
        kids = live_children(sec)
        own = sents_by_sec.get(sec.id, [])
        topic = topic_of_section.get(sec.id)
        if sec.id.startswith("g"):
            ntype = "chapter"
        elif not kids:
            ntype = "topic"
        else:
            ntype = "chapter" if sec.level <= 1 else "section"
        summary = " ".join(summarize_sentences([x.text for x in own], 2)) if own else ""
        if not summary and kids:
            names = [sec_by_id[k].title for k in kids[:6]]
            summary = "This part covers: " + ", ".join(names) + ("…" if len(kids) > 6 else ".")
        n = {
            "id": sec.id, "label": _short_title(sec.title, 44), "full_label": sec.title, "type": ntype, "level": sec.level,
            "parent": "root" if sec.parent == "s0" else sec.parent, "importance": 0.9 if ntype == "chapter" else 0.75 if ntype == "section" else 0.6,
            "definition": topic.definition if topic else None, "definition_quality": topic.definition_quality if topic else None,
            "summary": summary, "examples": topic.examples if topic else [], "formulas": topic.formulas if topic else [],
            "facts": topic.facts if topic else [], "aliases": topic.aliases if topic else [],
            "pages": sec.pages, "page": sec.page_start, "mentions": topic.mentions if topic else [], "section_id": sec.id,
            "n_children": len(kids), "order": order_counter,
        }
        if topic:
            n["importance"] = max(n["importance"], topic.importance)
            n["is_term"] = True
        order_counter += 1
        nodes.append(n)
    for c in concepts:
        if c.section_topic:
            continue
        sec = sec_by_id.get(c.section_id)
        if sec is None or sec.skip:
            continue
        nodes.append({
            "id": c.id, "label": c.label, "full_label": c.label, "type": "concept", "level": (sec.level or 0) + 1,
            "parent": "root" if c.section_id == "s0" else c.section_id, "importance": c.importance,
            "definition": c.definition, "definition_quality": c.definition_quality, "summary": "", "examples": c.examples,
            "formulas": c.formulas, "facts": c.facts, "aliases": c.aliases, "pages": c.pages, "page": c.definition_page or (c.pages[0] if c.pages else 1),
            "mentions": c.mentions, "section_id": c.section_id, "n_children": 0, "order": order_counter, "count": c.count,
        })
        order_counter += 1

    _fix_anaphoric_definitions(nodes, sections)
    if use_models:
        nodes = _apply_concept_filter(nodes)
        _fill_missing_definitions(nodes)
    node_ids = {n["id"] for n in nodes}
    eid = 0
    for n in nodes:
        if n["parent"] and n["parent"] in node_ids:
            eid += 1
            edges.append({"id": f"e{eid}", "source": n["parent"], "target": n["id"], "type": "contains", "weight": 1.0})
    # flow between sibling sections
    for sec in sections:
        kids = [k for k in live_children(sec) if k in node_ids]
        for a, b in zip(kids, kids[1:]):
            eid += 1
            edges.append({"id": f"e{eid}", "source": a, "target": b, "type": "next", "weight": 1.0})
    seen_pairs: set[tuple[str, str, str]] = set()
    for r in relations:
        a, b = concept_node_id.get(r.source), concept_node_id.get(r.target)
        if not a or not b or a == b or a not in node_ids or b not in node_ids:
            continue
        key = (a, b, r.type) if r.type == "prereq" else (min(a, b), max(a, b), r.type)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        eid += 1
        edges.append({"id": f"e{eid}", "source": a, "target": b, "type": r.type, "weight": r.weight})

    if use_models:
        edges = _apply_relation_model(nodes, edges)

    # section-level prerequisite flow (aggregate concept prereqs that cross sections)
    node_sec = {n["id"]: (n["id"] if n["type"] not in ("concept",) else n["parent"]) for n in nodes}
    agg: dict[tuple[str, str], int] = {}
    for e in edges:
        if e["type"] != "prereq":
            continue
        sa, sb = node_sec.get(e["source"]), node_sec.get(e["target"])
        if sa and sb and sa != sb and sa != "root" and sb != "root":
            agg[(sa, sb)] = agg.get((sa, sb), 0) + 1
    # learning path: sections in document order, concepts by first appearance
    order_of = {n["id"]: n.get("order", 0) for n in nodes}
    children_of: dict[str, list[str]] = {}
    for n in nodes:
        if n["parent"]:
            children_of.setdefault(n["parent"], []).append(n["id"])
    for k in children_of:
        children_of[k].sort(key=lambda i: order_of.get(i, 0))
    path: list[str] = []

    def walk(nid: str) -> None:
        if nid != "root":
            path.append(nid)
        for ch in children_of.get(nid, []):
            walk(ch)

    # concepts before sub-sections is wrong for flow; use document order of first appearance
    walk("root")
    stats = {
        "sections": len([s for s in sections if not s.skip]) - 1,
        "concepts": len([n for n in nodes if n["type"] == "concept"]) + len([n for n in nodes if n.get("is_term")]),
        "relations_prereq": len([e for e in edges if e["type"] == "prereq"]),
        "relations_related": len([e for e in edges if e["type"] == "related"]),
        "chunks": len(chunks), "sentences": len(sents), "pages": doc.n_pages,
        "section_flow_links": len(agg),
    }
    return KnowledgeGraph(
        title=doc.title, language=doc.language, unit_label=doc.unit_label, sections=sections, concepts=concepts,
        relations=relations, chunks=chunks, nodes=nodes, edges=edges, path=path, summary=doc_summary, stats=stats,
    )


_ANAPHORA = re.compile(r"(?i)^\s*(this|these|that|those|it|they|such)\b")


def _apply_concept_filter(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop concept nodes the trained M1 filter rejects. No model loaded: keep everything."""
    from indiclearn.studio import neural

    clf = neural.load_concept_filter()
    if clf is None:
        return nodes
    concepts = [n for n in nodes if n["type"] == "concept"]
    if not concepts:
        return nodes
    scores = clf.scores([(n["full_label"], n.get("definition") or n.get("summary") or "") for n in concepts])
    drop: set[str] = set()
    for n, sc in zip(concepts, scores):
        n["concept_score"] = round(float(sc), 3)
        # never drop what the notes explicitly define, or a named term with its acronym - "Micro AUROC", "MLKD (...)"
        protected = n.get("definition_quality") == "defined" or bool(re.search(r"\([A-Z][A-Za-z0-9\-]{1,9}\)\s*$", n["full_label"]))
        if sc < clf.threshold and not protected:
            drop.add(n["id"])
    if drop:
        log.info("M1 removed %d of %d concept nodes", len(drop), len(concepts))
    return [n for n in nodes if n["id"] not in drop]


def _apply_relation_model(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Let the trained M3 classifier judge every rule-made concept link (both directions).

    prereq a->b if the model says so (and not more strongly the other way), related if either direction
    says related, otherwise the co-occurrence link is dropped. No model: the rule links stay as they are."""
    from indiclearn.studio import neural

    m = neural.load_relations()
    if m is None:
        return edges
    by_id = {n["id"]: n for n in nodes}
    isc = lambda i: by_id.get(i, {}).get("type") == "concept"
    cand = [e for e in edges if e["type"] in ("prereq", "related") and isc(e["source"]) and isc(e["target"])]
    if not cand:
        return edges
    ctx = lambda n: (n.get("definition") or n.get("summary") or "")[:150]
    pairs = []
    for e in cand:
        a, b = by_id[e["source"]], by_id[e["target"]]
        both = f"{ctx(a)} {ctx(b)}".strip()
        pairs += [(a["full_label"], b["full_label"], both), (b["full_label"], a["full_label"], both)]
    preds = m.classify_batch(pairs)
    keep_ids, out_new = {id(e) for e in cand}, []
    for k, e in enumerate(cand):
        (lab_ab, p_ab), (lab_ba, p_ba) = preds[2 * k], preds[2 * k + 1]
        if lab_ab == "prereq" and p_ab >= 0.5 and not (lab_ba == "prereq" and p_ba > p_ab):
            out_new.append({**e, "type": "prereq", "weight": round(p_ab, 3), "by": "m3"})
        elif lab_ba == "prereq" and p_ba >= 0.5:
            out_new.append({**e, "source": e["target"], "target": e["source"], "type": "prereq", "weight": round(p_ba, 3), "by": "m3"})
        elif "related" in (lab_ab, lab_ba):
            out_new.append({**e, "type": "related", "weight": round(max(p_ab, p_ba), 3), "by": "m3"})
    log.info("M3 kept %d of %d concept links", len(out_new), len(cand))
    return [e for e in edges if id(e) not in keep_ids] + out_new


def _fill_missing_definitions(nodes: list[dict[str, Any]], lang: str = "en") -> None:
    """Ask the trained M2 writer for a description where the document gave none."""
    from indiclearn.studio import neural

    if neural.load_describer() is None:
        return
    filled = 0
    for n in nodes:
        if n["type"] != "concept" or (n.get("definition") or "").strip():
            continue
        text = neural.describe(n["full_label"], lang, n.get("summary") or "")
        if text:
            n["definition"] = text
            n["definition_quality"] = "generated"
            filled += 1
    if filled:
        log.info("M2 wrote %d missing descriptions", filled)


def _fix_anaphoric_definitions(nodes: list[dict[str, Any]], sections: list[Section]) -> None:
    """"This minimum rate is called the Nyquist rate." is useless alone: prepend the sentence it refers to."""
    sents_by_sec = {s.id: split_sentences(s.text) for s in sections}
    for n in nodes:
        d = n.get("definition")
        if not d or not _ANAPHORA.match(d):
            continue
        sents = sents_by_sec.get(n.get("section_id") or "", [])
        for i, st in enumerate(sents):
            if d.strip() == st.strip() and i > 0:
                n["definition"] = f"{sents[i - 1].strip()} {st.strip()}"
                n["definition_quality"] = "context"
                break


def sentences_of(text: str) -> list[str]:
    return split_sentences(text)
