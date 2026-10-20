"""A course as a study document: the lessons become each student's own "notes" with a ready-made knowledge map.

Every student gets one hidden document per course (documents.course = subject). Because it is an ordinary document,
everything built for uploaded notes works on a course unchanged: the doubt chat (answers grounded in the lessons),
the mind map (default and personal), quizzes, progress and recommendations.

The DEFAULT map is the same for everyone and comes from the syllabus, not from a model:
    course -> units -> topics (one per lesson) -> key terms of the lesson,
with "next" links in teaching order and "prereq" links from the syllabus (e.g. Recurrences need Recursion).
The PERSONAL map is that map coloured by this student's evidence: quick checks and quizzes, "mark as learned",
lessons opened, and doubts asked in the chat (each doubt becomes a node on the map).

Node ids are stable (topic ids, "topic:term"), so when lessons are added or improved the map is refreshed in place
and a student's progress is kept.
"""

from __future__ import annotations

import hashlib
import re
import threading
import uuid
from pathlib import Path
from typing import Any

from indiclearn.studio.courses import lessons as LS
from indiclearn.studio.courses import papers as P
from indiclearn.studio.db import Database, jdump, jload, now

TERMS_PER_TOPIC = 5
BUILD = 3  # bump when build() changes, so every student's course document is rebuilt (progress is kept)
_LOCK = threading.Lock()
_BUILT: dict[tuple[str, str], dict[str, Any]] = {}  # (subject, version) -> built course (shared by all students)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", re.sub(r"\$[^$]*\$", "", s.lower())).strip("-")[:40] or "term"


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def version(subject: str) -> str:
    ls = LS.lessons(subject)
    syl = P.syllabus(subject)
    h = hashlib.sha1()
    h.update(f"build{BUILD}".encode())
    h.update(jdump(syl).encode())
    for k in sorted(ls):
        h.update(k.encode())
        h.update(jdump(ls[k].get("en") or {}).encode())
    h.update(str(len(P.bank(subject))).encode())  # new checked questions reach every student's course quiz
    return h.hexdigest()[:12]


def build(subject: str) -> dict[str, Any]:
    """nodes, edges, path, reader blocks, chunks and quiz rows for a course (cached per lesson version)."""
    ver = version(subject)
    with _LOCK:
        if (subject, ver) in _BUILT:
            return _BUILT[(subject, ver)]
    syl = P.syllabus(subject)
    ls = LS.lessons(subject)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    blocks: list[tuple[int, str, int, str]] = []  # (page, kind, level, text)
    chunks: list[dict[str, Any]] = []
    eid = 0

    def edge(a: str, b: str, kind: str, w: float = 1.0, **extra: Any) -> None:
        nonlocal eid
        eid += 1
        edges.append({"id": f"e{eid}", "source": a, "target": b, "type": kind, "weight": w, **extra})

    def node(**kw: Any) -> dict[str, Any]:
        base = {"definition": None, "definition_quality": None, "summary": "", "examples": [], "formulas": [], "facts": [], "aliases": [],
                "pages": [kw.get("page", 1)], "mentions": [], "n_children": 0}
        n = {**base, **kw}
        nodes.append(n)
        return n

    n_topics = sum(len(u["topics"]) for u in syl["units"])
    node(id="root", label=syl.get("short") or syl["title"], full_label=syl["title"], type="root", level=0, parent=None, importance=1.0,
         summary=f"{syl['title']}: {len(syl['units'])} units, {n_topics} lessons.", page=1, section_id="root", n_children=len(syl["units"]), order=0)
    seen_terms: dict[str, str] = {}
    page, order = 0, 1
    prev_unit = None
    for ui, u in enumerate(syl["units"], 1):
        first_page = page + 1
        un = node(id=u["id"], label=_clean(u["title"])[:44], full_label=f"Unit {ui}: {u['title']}", type="chapter", level=1, parent="root", importance=0.9,
                  summary="This unit covers: " + ", ".join(t["label"] for t in u["topics"]) + ".", page=first_page, section_id=u["id"],
                  n_children=len(u["topics"]), order=order)
        order += 1
        if prev_unit:
            edge(prev_unit, u["id"], "next")
        edge("root", u["id"], "contains")
        prev_unit = u["id"]
        prev_topic = None
        for t in u["topics"]:
            page += 1
            les = (ls.get(t["id"]) or {}).get("en")
            terms = (les or {}).get("key_terms", [])[:TERMS_PER_TOPIC] if les else []
            kids = []
            for k in terms:
                key = _slug(k.get("term", ""))
                if not key or key in seen_terms:
                    if key in seen_terms and seen_terms[key] != t["id"]:
                        edge(seen_terms[key], t["id"], "related", 0.6)  # a term taught earlier and used again here
                    continue
                seen_terms[key] = t["id"]
                kids.append((key, k))
            tn = node(id=t["id"], label=_clean(t["label"].split(" (")[0])[:44], full_label=t["label"], type="section" if kids else "topic", level=2,
                      parent=u["id"], importance=0.9 if t.get("gate") else 0.8, is_term=True, quiz=True,
                      definition=_clean((les or {}).get("summary", "")) or None, definition_quality="lesson" if les else None,
                      summary=_clean((les or {}).get("summary", "")), examples=[_clean((les or {}).get("example", {}).get("title", ""))] if les and les.get("example", {}).get("title") else [],
                      facts=[_clean(x) for x in (les or {}).get("exam_tips", [])[:3]], aliases=[k for k in t.get("keywords", [])[:6] if len(k) > 3],
                      page=page, section_id=t["id"], n_children=len(kids), order=order, numerical=bool(t.get("numerical")), refs=t.get("refs", []))
            order += 1
            edge(u["id"], t["id"], "contains")
            if prev_topic:
                edge(prev_topic, t["id"], "next")
            prev_topic = t["id"]
            for key, k in kids:
                node(id=f"{t['id']}:{key}", label=_clean(k["term"])[:44], full_label=_clean(k["term"]), type="concept", level=3, parent=t["id"], importance=0.6,
                     definition=_clean(k.get("meaning", "")), definition_quality="lesson", page=page, section_id=t["id"], order=order)
                order += 1
                edge(t["id"], f"{t['id']}:{key}", "contains")
            # ---- reader blocks + retrieval chunks for this lesson
            if t is u["topics"][0]:
                blocks.append((page, "heading", 1, f"Unit {ui}: {u['title']}"))
            blocks.append((page, "heading", 2, t["label"]))
            if t.get("refs"):
                blocks.append((page, "ref", 0, "Read: " + "; ".join(t["refs"])))
            if not les:
                blocks.append((page, "para", 0, "This lesson is being written."))
                continue
            texts: list[str] = [f"{t['label']}. {les['summary']}"]
            blocks.append((page, "para", 0, les["summary"]))
            for s in les.get("sections", []):
                blocks.append((page, "heading", 3, s.get("heading", "")))
                for line in str(s.get("body", "")).split("\n"):
                    if line.strip():
                        bullet = re.match(r"^\s*[-*•]\s+(.*)", line)
                        blocks.append((page, "bullet" if bullet else "para", 0, bullet.group(1) if bullet else line.strip()))
                texts.append(f"{t['label']} - {s.get('heading', '')}. {s.get('body', '')}")
            rules = LS.rules_for(subject, t.get("numerical"))
            if rules:
                blocks.append((page, "heading", 3, "Exact rules for numericals"))
                blocks += [(page, "bullet", 0, r) for r in rules]
                texts.append(f"{t['label']} - exact rules used for numericals: " + " ".join(rules))
            if les.get("key_terms"):
                blocks.append((page, "heading", 3, "Key terms"))
                blocks += [(page, "bullet", 0, f"{k['term']} - {k['meaning']}") for k in les["key_terms"]]
                texts.append(f"{t['label']} - key terms: " + "; ".join(f"{k['term']}: {k['meaning']}" for k in les["key_terms"]))
            ex = les.get("example") or {}
            if ex.get("body"):
                blocks.append((page, "heading", 3, "Worked example" + (f": {ex['title']}" if ex.get("title") else "")))
                blocks += [(page, "para", 0, x) for x in str(ex["body"]).split("\n") if x.strip()]
                texts.append(f"{t['label']} - worked example {ex.get('title', '')}: {ex['body']}")
            for head, key in (("How to score full marks", "exam_tips"), ("Common mistakes", "mistakes")):
                if les.get(key):
                    blocks.append((page, "heading", 3, head))
                    blocks += [(page, "bullet", 0, x) for x in les[key]]
                    texts.append(f"{t['label']} - {head.lower()}: " + " ".join(les[key]))
            for x in texts:
                chunks.append({"id": f"c{len(chunks)}", "section_id": t["id"], "page": page, "text": _clean(x)})
    # prerequisites from the syllabus
    ids = {n["id"] for n in nodes}
    for u in syl["units"]:
        for t in u["topics"]:
            for p in t.get("prereq", []):
                if p in ids:
                    edge(p, t["id"], "prereq", 1.0, by="syllabus")
    blocks = [(p, k, lv, tx.replace("**", "")) for p, k, lv, tx in blocks]  # the reader shows plain text (bold is for lesson pages)
    path = [n["id"] for n in sorted(nodes, key=lambda n: n.get("order", 0)) if n["id"] != "root"]
    quiz_rows = []
    by_topic = {n["id"]: n for n in nodes}
    for q in P.bank(subject):
        if q.get("qtype") == "mcq" and isinstance(q.get("correct"), int) and q["topic"] in by_topic and q.get("options"):
            quiz_rows.append((q["topic"], {"q": q["q"], "options": q["options"], "answer": q["correct"], "why": q.get("answer", ""), "page": by_topic[q["topic"]]["page"]}, q["id"]))
    # exactly computed MCQs (the same generators as the exam papers) so numerical topics always have quiz questions
    import random as _random

    calc = P.calc(subject)
    for _u, t in P.topics_of(None, subject):
        kind = t.get("numerical")
        if not kind or kind not in getattr(calc, "MCQ_KINDS", calc.VALUE_GENERATORS) or t["id"] not in by_topic:
            continue
        for k in range(3):
            try:
                g = calc.gen_mcq_numeric(kind, _random.Random(f"{subject}/{t['id']}/{k}"), 2, "medium", t["id"])
            except Exception:  # pragma: no cover - a generator that cannot make an MCQ just adds none
                continue
            if any("table" in b for b in g["blocks"]) or len(g["options"]) != 4:
                continue  # a quiz card shows text and formulas, not tables
            stem = " ".join(b["p"] if "p" in b else f"$${b['math']}$$" for b in g["blocks"])
            why = " ".join(b["p"] if "p" in b else f"$${b['math']}$$" for b in g["answer"] if "table" not in b)
            quiz_rows.append((t["id"], {"q": stem, "options": g["options"], "answer": g["correct"], "why": why[:900], "page": by_topic[t["id"]]["page"]},
                              f"calc-{t['id']}-{k}"))
    stats = {"concepts": sum(1 for n in nodes if n["type"] == "concept") + n_topics, "sections": n_topics, "relations_prereq": sum(1 for e in edges if e["type"] == "prereq"),
             "relations_related": sum(1 for e in edges if e["type"] == "related"), "chunks": len(chunks), "pages": page, "graph_engine": "syllabus", "course": subject,
             "course_version": ver, "lessons": len(ls)}
    out = {"version": ver, "nodes": nodes, "edges": edges, "path": path, "blocks": blocks, "chunks": chunks, "quiz": quiz_rows, "stats": stats,
           "title": syl["title"], "summary": f"{syl['title']} - {len(syl['units'])} units, {n_topics} lessons with solved examples, quick checks and exam papers.",
           "pages": page}
    with _LOCK:
        _BUILT[(subject, ver)] = out
    return out


def _chunks_hash(c: dict[str, Any]) -> str:
    """Vectors depend only on the lesson text (not on the question bank)."""
    return hashlib.sha1("\n".join(x["text"] for x in c["chunks"]).encode()).hexdigest()[:12]


def vectors(subject: str, embedder: Any, cache_dir: Path | None) -> Any:
    """Embeddings of a course's chunks: computed once per lesson version and kept on disk (a course opens instantly)."""
    import numpy as np

    c = build(subject)
    path = cache_dir / f"{subject}-{_chunks_hash(c)}.npy" if cache_dir else None
    if path is not None and path.exists():
        try:
            v = np.load(path)
            if len(v) == len(c["chunks"]):
                return v
        except (OSError, ValueError):
            pass
    v = embedder.encode_docs([x["text"] for x in c["chunks"]])
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, v)
        except OSError:
            pass
    return v


def cached_vectors(subject: str, cache_dir: Path | None) -> Any:
    import numpy as np

    c = build(subject)
    path = cache_dir / f"{subject}-{_chunks_hash(c)}.npy" if cache_dir else None
    if path is not None and path.exists():
        try:
            v = np.load(path)
            return v if len(v) == len(c["chunks"]) else None
        except (OSError, ValueError):
            return None
    return None


def fill_vectors(db: Database, doc_id: str, subject: str, vecs: Any) -> None:
    c = build(subject)
    with db.conn() as x:
        x.executemany("UPDATE chunks SET vec=? WHERE doc_id=? AND id=?", [(vecs[i].astype("float32").tobytes(), doc_id, ch["id"]) for i, ch in enumerate(c["chunks"])])


def ensure(db: Database, user_id: str, subject: str, cache_dir: Path | None = None) -> tuple[str, bool, bool]:
    """The student's document for this course (created, or refreshed when the lessons changed).

    Returns (doc_id, changed, needs_vectors): the map, lessons, chat (keyword search) and quiz work at once; semantic search
    vectors are added in the background the very first time a lesson version is opened on this server."""
    P.syllabus(subject)  # raises "Course not found." for an unknown subject
    c = build(subject)
    row = db.one("SELECT id, stats_json FROM documents WHERE user_id=? AND course=?", (user_id, subject))
    if row and jload(row["stats_json"], {}).get("course_version") == c["version"]:
        has = db.one("SELECT COUNT(*) n FROM chunks WHERE doc_id=? AND vec IS NULL", (row["id"],))["n"] == 0
        return row["id"], False, not has
    doc_id = row["id"] if row else uuid.uuid4().hex[:16]
    vecs = cached_vectors(subject, cache_dir)
    with db.conn() as x:
        if not row:
            x.execute("INSERT INTO documents(id,user_id,title,filename,ext,size_bytes,status,created_at,is_sample,course) VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (doc_id, user_id, c["title"], f"{subject}.course", ".course", 0, "ready", now(), 0, subject))
        for t in ("doc_blocks", "chunks"):
            x.execute(f"DELETE FROM {t} WHERE doc_id=?", (doc_id,))
        x.execute("DELETE FROM quiz_bank WHERE doc_id=? AND engine='course-bank'", (doc_id,))
        x.executemany("INSERT INTO doc_blocks(doc_id,idx,page,kind,level,text,meta_json) VALUES (?,?,?,?,?,?,?)",
                      [(doc_id, i, p, k, lv, tx, "{}") for i, (p, k, lv, tx) in enumerate(c["blocks"])])
        x.execute("INSERT OR REPLACE INTO doc_graph(doc_id,nodes_json,edges_json,path_json) VALUES (?,?,?,?)", (doc_id, jdump(c["nodes"]), jdump(c["edges"]), jdump(c["path"])))
        x.executemany("INSERT INTO chunks(doc_id,id,section_id,page,text,idx,vec) VALUES (?,?,?,?,?,?,?)",
                      [(doc_id, ch["id"], ch["section_id"], ch["page"], ch["text"], i, vecs[i].astype("float32").tobytes() if vecs is not None else None)
                       for i, ch in enumerate(c["chunks"])])
        x.executemany("INSERT OR IGNORE INTO quiz_bank(id,doc_id,node_id,lang,src_id,payload_json,engine,created_at) VALUES (?,?,?,?,?,?,?,?)",
                      [(f"{doc_id}-{src}", doc_id, topic, "en", src, jdump(q), "course-bank", now()) for topic, q, src in c["quiz"]])
        x.execute("UPDATE documents SET title=?, lang='en', script='Latin', unit_label='Lesson', n_pages=?, status='ready', error=NULL, stats_json=?, summary=? WHERE id=?",
                  (c["title"], c["pages"], jdump(c["stats"]), c["summary"], doc_id))
    return doc_id, True, vecs is None
