"""Application service: orchestrates ingest -> graph -> reader -> tutor -> quiz -> progress.

All methods take the authenticated ``user_id`` and verify ownership, so users are isolated from one
another.  Heavy work (parsing / graph building) runs on a small thread pool and is polled via the
document ``status`` field - the same shape works unchanged if the pool is later replaced by a queue.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import random
import re
import tempfile
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from indiclearn.studio import mastery, tutor
from indiclearn.studio import quiz as quizmod
from indiclearn.studio import quizbank
from indiclearn.studio.auth import AuthError, AuthService
from indiclearn.studio.brain import TARGET_LANGS
from indiclearn.studio.config import Settings
from indiclearn.studio.db import Database, jdump, jload, now
from indiclearn.studio.i18n import tr
from indiclearn.studio.kgraph import build_knowledge_graph
from indiclearn.studio.lang import LANG_NAMES, SUPPORTED_LANGS
from indiclearn.studio.llm import Models
from indiclearn.studio.parse import parse_file
from indiclearn.studio.retrieval import DocIndex

log = logging.getLogger("indiclearn.service")
XP_FOR = {"asked": 5, "quiz_correct": 10, "quiz_incorrect": 3, "self_understood": 4, "self_confused": 2, "viewed": 1}
TRANSLATABLE = ("text", "reasoning", "key_terms", "follow_ups", "beyond_notes")
AUDIO_MIME = {".webm": "audio/webm", ".wav": "audio/wav", ".mp3": "audio/mp3", ".m4a": "audio/mp4", ".ogg": "audio/ogg", ".flac": "audio/flac"}
EXTRA_KEYS = ("notice", "reasoning", "key_terms", "follow_ups", "check", "beyond_notes", "used_general_knowledge", "engine", "understood_as")
ALLOWED_EXT = {".pdf", ".pptx", ".docx", ".md", ".txt", ".markdown"}
SAMPLES = [
    ("signals_and_systems_notes.md", "Signals & Systems (English)", "en"),
    ("signals_and_systems_te.md", "సిగ్నల్స్ & సిస్టమ్స్ (తెలుగు)", "te"),
    ("signals_and_systems_hi.md", "सिग्नल और सिस्टम (हिन्दी)", "hi"),
    ("signals_sanskrit.md", "संकेत-प्रणाली (संस्कृतम्)", "sa"),
]


class _NoBrainModels:
    """The models without the AI engine: used when a student has used up today's AI answers."""

    def __init__(self, models: Any) -> None:
        self._m = models
        self.brain = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._m, name)


def _err(msg: str, status: int = 400) -> AuthError:
    return AuthError(msg, status)


def _neural_status() -> dict[str, Any]:
    try:
        from indiclearn.studio import neural
        return neural.status()
    except Exception:  # pragma: no cover - status must never break /api/info
        return {}


class StudioService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or Settings()
        self.s.ensure_dirs()
        self.db = Database(self.s.db_path)
        self.auth = AuthService(self.db, self.s.token_ttl_days)
        self.models = Models(self.s)
        self._progress: dict[str, dict[str, Any]] = {}  # doc_id -> upload progress (page n of N, stage)
        from indiclearn.studio.courses.exams import ExamService

        self.exams = ExamService(self.db, self.models)
        self._course_vecs: dict = {}  # (doc, subject) -> background embedding in progress
        self._course_lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ingest")
        self._cache: OrderedDict[str, tuple[DocIndex, list[dict[str, Any]], list[dict[str, Any]]]] = OrderedDict()
        self._lock = threading.Lock()
        self._tcache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.samples_dir = self._find_samples()
        # documents left "processing" by a crashed process can never finish
        self.db.run("UPDATE documents SET status='error', error='Interrupted by a restart - please upload again.' WHERE status='processing'")

    # ----------------------------------------------------------------- helpers
    def _find_samples(self) -> Path | None:
        for p in (Path(__file__).resolve().parents[3] / "data" / "samples", Path("data/samples")):
            if p.is_dir():
                return p
        return None

    def _doc(self, user_id: str, doc_id: str) -> Any:
        row = self.db.one("SELECT * FROM documents WHERE id=? AND user_id=?", (doc_id, user_id))
        if row is None:
            raise _err("Document not found.", 404)
        return row

    def _ready(self, user_id: str, doc_id: str) -> Any:
        d = self._doc(user_id, doc_id)
        if d["status"] != "ready":
            raise _err("Document is still being processed." if d["status"] == "processing" else (d["error"] or "Document failed to process."), 409)
        return d

    @staticmethod
    def _doc_public(r: Any) -> dict[str, Any]:
        return {"id": r["id"], "title": r["title"], "filename": r["filename"], "ext": r["ext"], "size_bytes": r["size_bytes"], "lang": r["lang"], "script": r["script"],
                "unit_label": r["unit_label"], "n_pages": r["n_pages"], "status": r["status"], "error": r["error"], "created_at": r["created_at"],
                "stats": jload(r["stats_json"], {}), "summary": r["summary"], "is_sample": bool(r["is_sample"]),
                "course": r["course"] if "course" in r.keys() else None}

    # ------------------------------------------------------------------ ingest
    def rename_doc(self, user_id: str, doc_id: str, title: str) -> dict[str, Any]:
        title = re.sub(r"\s+", " ", (title or "")).strip()[:120]
        if len(title) < 2:
            raise _err("Please give the document a name.")
        self._doc(user_id, doc_id)
        self.db.run("UPDATE documents SET title=? WHERE id=? AND user_id=?", (title, doc_id, user_id))
        return self._doc_public(self._doc(user_id, doc_id))

    def list_docs(self, user_id: str) -> list[dict[str, Any]]:
        out = []
        for r in self.db.all("SELECT * FROM documents WHERE user_id=? AND course IS NULL ORDER BY created_at DESC", (user_id,)):
            d = self._doc_public(r)
            if d["status"] == "processing" and r["id"] in self._progress:
                d["progress"] = self._progress[r["id"]]
            out.append(d)
        return out

    def add_upload(self, user_id: str, filename: str, data: bytes, is_sample: bool = False) -> dict[str, Any]:
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            raise _err(f"Unsupported file type {ext or '(none)'}. Upload PDF, PPTX, DOCX, Markdown or TXT.", 415)
        if not data:
            raise _err("The file is empty.")
        if len(data) > self.s.max_upload_mb * 1024 * 1024:
            raise _err(f"File is larger than {self.s.max_upload_mb} MB.", 413)
        n = self.db.one("SELECT COUNT(*) c FROM documents WHERE user_id=? AND course IS NULL", (user_id,))["c"]
        if n >= self.s.max_docs_per_user:
            raise _err(f"You can keep up to {self.s.max_docs_per_user} documents - delete one first.", 429)
        doc_id = uuid.uuid4().hex[:16]
        path = self.s.upload_dir / f"{doc_id}{ext}"
        path.write_bytes(data)
        safe_name = Path(filename).name[:120]
        self.db.run(
            "INSERT INTO documents(id,user_id,title,filename,ext,size_bytes,status,created_at,is_sample) VALUES (?,?,?,?,?,?,?,?,?)",
            (doc_id, user_id, Path(safe_name).stem, safe_name, ext, len(data), "processing", now(), int(is_sample)),
        )
        self.pool.submit(self._build, doc_id, path, safe_name)
        return self._doc_public(self.db.one("SELECT * FROM documents WHERE id=?", (doc_id,)))

    def add_sample(self, user_id: str, key: str) -> dict[str, Any]:
        for fn, _title, code in SAMPLES:
            if code == key or fn == key:
                if not self.samples_dir or not (self.samples_dir / fn).exists():
                    break
                return self.add_upload(user_id, fn, (self.samples_dir / fn).read_bytes(), is_sample=True)
        raise _err("Sample not found.", 404)

    def samples(self) -> list[dict[str, str]]:
        if not self.samples_dir:
            return []
        return [{"key": c, "title": t, "lang": c} for fn, t, c in SAMPLES if (self.samples_dir / fn).exists()]

    def _set_progress(self, doc_id: str, done: int, total: int, stage: str) -> None:
        self._progress[doc_id] = {"done": done, "total": total, "stage": stage}

    def _build(self, doc_id: str, path: Path, filename: str) -> None:
        try:
            pd = parse_file(path, filename, progress=lambda d, t, st: self._set_progress(doc_id, d, t, st))
            self._set_progress(doc_id, pd.n_pages, pd.n_pages, "map")
            brain = getattr(self.models, "brain", None)
            # A local LLM (Ollama on a laptop GPU) needs a minute or more for the map: open the document
            # straight away with the rule-based map and upgrade it in the background.
            defer = bool(brain is not None and brain.available and brain.first_is_local())
            hosted = bool(brain is not None and brain.available and not defer)
            # with a hosted LLM about to rebuild the map, the offline models' work would be thrown away
            kg = build_knowledge_graph(pd, use_models=not hosted)
            try:
                from indiclearn.studio.kgraph_refine import refine

                if hosted:
                    self._set_progress(doc_id, pd.n_pages, pd.n_pages, "ai")
                rep = refine(kg, None if defer else brain)
                if hosted and not str(rep.get("graph_engine", "")).startswith("llm"):  # the LLM failed: use M1-M3 after all
                    kg = build_knowledge_graph(pd)
                    refine(kg, None)
            except Exception as e:  # pragma: no cover - refinement is an improvement, never a blocker
                log.warning("graph refinement skipped: %s", e)
            vecs = None
            if self.models.embedder is not None and kg.chunks:
                try:
                    vecs = self.models.embedder.encode_docs([c.text for c in kg.chunks])
                except Exception as e:  # pragma: no cover
                    log.warning("embedding failed: %s", e)
            with self.db.conn() as c:
                c.execute("DELETE FROM doc_blocks WHERE doc_id=?", (doc_id,))
                c.executemany("INSERT INTO doc_blocks(doc_id,idx,page,kind,level,text,meta_json) VALUES (?,?,?,?,?,?,?)",
                              [(doc_id, i, b.page, b.kind, b.level, b.text, jdump(b.meta)) for i, b in enumerate(pd.blocks)])
                c.execute("INSERT OR REPLACE INTO doc_graph(doc_id,nodes_json,edges_json,path_json) VALUES (?,?,?,?)", (doc_id, jdump(kg.nodes), jdump(kg.edges), jdump(kg.path)))
                c.executemany("INSERT INTO chunks(doc_id,id,section_id,page,text,idx,vec) VALUES (?,?,?,?,?,?,?)",
                              [(doc_id, ch.id, ch.section_id, ch.page, ch.text, ch.idx, vecs[i].astype("float32").tobytes() if vecs is not None else None) for i, ch in enumerate(kg.chunks)])
                c.execute(
                    "UPDATE documents SET title=?, lang=?, script=?, unit_label=?, n_pages=?, status='ready', error=NULL, stats_json=?, summary=? WHERE id=?",
                    (kg.title or pd.title, pd.language, pd.script, pd.unit_label, pd.n_pages, jdump({**kg.stats, "warnings": pd.warnings}), kg.summary, doc_id),
                )
        except Exception as e:
            log.exception("ingest failed for %s", doc_id)
            self.db.run("UPDATE documents SET status='error', error=? WHERE id=?", (f"Could not read this file ({type(e).__name__}: {str(e)[:160]}).", doc_id))
            return
        if defer:
            self.pool.submit(self._refine_later, doc_id, kg, brain)
        else:
            self._start_quiz_bank(doc_id)

    def _refine_later(self, doc_id: str, kg: Any, brain: Any) -> None:
        """Background LLM upgrade of a map that was first built by rules (local-model mode)."""
        try:
            from indiclearn.studio.kgraph_refine import refine

            rep = refine(kg, brain)
            if not str(rep.get("graph_engine", "")).startswith("llm"):
                return
            d = self.db.one("SELECT stats_json FROM documents WHERE id=?", (doc_id,))
            if d is None:
                return
            stats = {**jload(d["stats_json"], {}), **kg.stats}
            with self.db.conn() as c:
                c.execute("UPDATE doc_graph SET nodes_json=?, edges_json=?, path_json=? WHERE doc_id=?", (jdump(kg.nodes), jdump(kg.edges), jdump(kg.path), doc_id))
                c.execute("UPDATE documents SET stats_json=? WHERE id=?", (jdump(stats), doc_id))
            with self._lock:
                self._cache.pop(doc_id, None)
            log.info("map for %s upgraded by %s", doc_id, rep.get("graph_engine"))
        except Exception as e:  # pragma: no cover
            log.warning("background map upgrade failed for %s: %s", doc_id, e)
        self._start_quiz_bank(doc_id)

    def _start_quiz_bank(self, doc_id: str, extra_lang: str | None = None) -> None:
        """Write AI quiz questions in the background so "Start quiz" is instant (never blocks the upload)."""
        try:
            d = self.db.one("SELECT * FROM documents WHERE id=?", (doc_id,))
            if d is None or d["status"] != "ready":
                return
            _d, ix, nodes, _e = self._load(d["user_id"], doc_id)
            langs = [d["lang"] if d["lang"] in SUPPORTED_LANGS else "en"]
            if extra_lang and extra_lang in SUPPORTED_LANGS and extra_lang not in langs:
                langs.append(extra_lang)
            quizbank.build_in_background(self.db, getattr(self.models, "brain", None), doc_id, nodes, ix, langs)
        except Exception as e:  # pragma: no cover - the bank is an extra, never a failure
            log.warning("quiz bank not started for %s: %s", doc_id, e)

    # ------------------------------------------------------------------ courses (lessons as a study document)
    def course_doc(self, user_id: str, subject: str) -> dict[str, Any]:
        """The student's study document for a course: the lessons with a ready knowledge map, chat, quiz and progress."""
        from indiclearn.studio.courses import workspace

        cache_dir = Path(self.s.data_dir) / "course_vectors"
        doc_id, changed, needs = workspace.ensure(self.db, user_id, subject, cache_dir)
        if changed:
            with self._lock:
                self._cache.pop(doc_id, None)
        if needs and self.models.embedder is not None and (doc_id, subject) not in self._course_vecs:
            self._course_vecs[(doc_id, subject)] = True

            def work() -> None:
                try:
                    with self._course_lock:  # one embedding run per course at a time; the others then read the disk cache
                        vecs = workspace.vectors(subject, self.models.embedder, cache_dir)
                    workspace.fill_vectors(self.db, doc_id, subject, vecs)
                    with self._lock:
                        self._cache.pop(doc_id, None)
                except Exception as e:  # pragma: no cover - keyword search keeps working
                    log.warning("course vectors failed for %s: %s", subject, e)
                finally:
                    self._course_vecs.pop((doc_id, subject), None)
            self.pool.submit(work)
        return self._doc_public(self._doc(user_id, doc_id))

    def course_lesson(self, user_id: str, subject: str, topic: str, lang: str = "en", seed: int | None = None, poll: bool = False) -> dict[str, Any]:
        from indiclearn.studio.courses import lessons

        out = lessons.lesson(self.db, user_id, topic, lang, seed, getattr(self.models, "brain", None), subject)
        doc = self.course_doc(user_id, subject)
        if not poll and seed is None:  # a "New example" or a background check is not another visit
            mastery.record_evidence(self.db, user_id, doc["id"], topic, "viewed", {"lesson": True})
        return {**out, "doc_id": doc["id"]}

    def course_mark(self, user_id: str, subject: str, topic: str, status: str) -> dict[str, Any]:
        from indiclearn.studio.courses import lessons

        out = lessons.mark(self.db, user_id, topic, status, subject)
        if status == "learned":
            doc = self.course_doc(user_id, subject)
            mastery.record_evidence(self.db, user_id, doc["id"], topic, "self_understood", {"lesson": True})
        return out

    def course_check(self, user_id: str, subject: str, topic: str, correct: bool) -> dict[str, Any]:
        """A quick-check answer on a lesson page counts like a quiz answer on the personal map."""
        from indiclearn.studio.courses import papers as P

        if topic not in {t["id"] for _u, t in P.topics_of(None, subject)}:
            raise _err("Lesson not found.", 404)
        doc = self.course_doc(user_id, subject)
        mastery.record_evidence(self.db, user_id, doc["id"], topic, "quiz_correct" if correct else "quiz_incorrect", {"lesson_check": True})
        return {"ok": True}

    def delete_doc(self, user_id: str, doc_id: str) -> None:
        d = self._doc(user_id, doc_id)
        for t in ("evidence", "doubts", "chat_messages"):
            self.db.run(f"DELETE FROM {t} WHERE user_id=? AND doc_id=?", (user_id, doc_id))
        self.db.run("DELETE FROM quizzes WHERE user_id=? AND doc_id=?", (user_id, doc_id))
        self.db.run("DELETE FROM documents WHERE id=?", (doc_id,))
        try:
            (self.s.upload_dir / f"{d['id']}{d['ext']}").unlink(missing_ok=True)
        except OSError:
            pass
        with self._lock:
            self._cache.pop(doc_id, None)

    def file_path(self, user_id: str, doc_id: str) -> tuple[Path, str]:
        d = self._doc(user_id, doc_id)
        return self.s.upload_dir / f"{d['id']}{d['ext']}", d["filename"]

    # ------------------------------------------------------------------- data
    def _load(self, user_id: str, doc_id: str) -> tuple[Any, DocIndex, list[dict[str, Any]], list[dict[str, Any]]]:
        d = self._ready(user_id, doc_id)
        with self._lock:
            hit = self._cache.get(doc_id)
            if hit:
                self._cache.move_to_end(doc_id)
                return d, hit[0], hit[1], hit[2]
        g = self.db.one("SELECT * FROM doc_graph WHERE doc_id=?", (doc_id,))
        nodes, edges = jload(g["nodes_json"], []), jload(g["edges_json"], [])
        rows = self.db.all("SELECT * FROM chunks WHERE doc_id=? ORDER BY idx", (doc_id,))
        chunks = [{"id": r["id"], "section_id": r["section_id"], "page": r["page"], "text": r["text"]} for r in rows]
        vecs = None
        if self.models.embedder is not None and rows and all(r["vec"] for r in rows):
            vecs = np.vstack([np.frombuffer(r["vec"], dtype="float32") for r in rows])
        idx = DocIndex(chunks, nodes, d["lang"] or "en", vecs, self.models.embedder if vecs is not None else None)
        with self._lock:
            self._cache[doc_id] = (idx, nodes, edges)
            while len(self._cache) > 16:
                self._cache.popitem(last=False)
        return d, idx, nodes, edges

    def _doubts(self, user_id: str, doc_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.all("SELECT * FROM doubts WHERE user_id=? AND doc_id=? ORDER BY id", (user_id, doc_id))]

    def graph(self, user_id: str, doc_id: str, mode: str = "default") -> dict[str, Any]:
        d, _ix, nodes, edges = self._load(user_id, doc_id)
        if mode == "personal":
            states = mastery.node_states(self.db, user_id, doc_id)
            out = mastery.personal_overlay(nodes, edges, states, self._doubts(user_id, doc_id))
            return {"mode": "personal", **out, "path": jload(self.db.one("SELECT path_json FROM doc_graph WHERE doc_id=?", (doc_id,))["path_json"], []),
                    "doc": self._doc_public(d)}
        return {"mode": "default", "nodes": nodes, "edges": edges, "path": jload(self.db.one("SELECT path_json FROM doc_graph WHERE doc_id=?", (doc_id,))["path_json"], []),
                "doc": self._doc_public(d)}

    def reader(self, user_id: str, doc_id: str) -> dict[str, Any]:
        d, _ix, nodes, _e = self._load(user_id, doc_id)
        rows = self.db.all("SELECT idx,page,kind,level,text,meta_json FROM doc_blocks WHERE doc_id=? ORDER BY idx", (doc_id,))
        pages: dict[int, list[dict[str, Any]]] = {}
        for r in rows:
            pages.setdefault(r["page"], []).append({"i": r["idx"], "kind": r["kind"], "level": r["level"], "text": r["text"], "meta": jload(r["meta_json"], {})})
        # concept surface forms the reader can highlight
        terms = [{"id": n["id"], "label": n["label"], "aliases": n.get("aliases") or [], "type": n["type"]} for n in nodes if n["type"] == "concept" or n.get("is_term")]
        toc = []
        for n in nodes:
            if n["type"] in ("chapter", "section", "topic") and not n.get("is_term") or n["type"] in ("chapter", "section"):
                if toc and re.sub(r"\W+", "", toc[-1]["label"].lower()) == re.sub(r"\W+", "", n["label"].lower()):
                    continue  # one slide title repeated on 10 slides is one entry in the contents
                toc.append({"id": n["id"], "label": n["label"], "level": n["level"], "page": n["page"], "type": n["type"]})
        return {"doc": self._doc_public(d), "pages": [{"page": p, "blocks": b} for p, b in sorted(pages.items())], "terms": terms, "toc": toc,
                "has_original": d["ext"] == ".pdf"}

    # --------------------------------------------------------------- progress
    def progress(self, user_id: str, doc_id: str) -> dict[str, Any]:
        d, _ix, nodes, edges = self._load(user_id, doc_id)
        states = mastery.node_states(self.db, user_id, doc_id)
        ov = mastery.personal_overlay(nodes, edges, states, self._doubts(user_id, doc_id))
        leaves = [n for n in ov["nodes"] if n["type"] == "concept" or (n.get("is_term") and n["type"] != "doubt")]
        counts = {k: 0 for k in mastery.STATUSES}
        for n in leaves:
            counts[n["mastery"]["status"]] = counts.get(n["mastery"]["status"], 0) + 1
        qrows = self.db.all("SELECT correct FROM quiz_questions q JOIN quizzes z ON z.id=q.quiz_id WHERE z.user_id=? AND z.doc_id=? AND q.answered=1", (user_id, doc_id))
        recs = mastery.recommend(nodes, edges, states, 6)
        ndo = len(self._doubts(user_id, doc_id))
        total = max(len(leaves), 1)
        return {"counts": counts, "total_concepts": len(leaves), "mastered_pct": round(100 * counts["mastered"] / total),
                "touched_pct": round(100 * (total - counts["new"]) / total), "quiz_answered": len(qrows), "quiz_correct": sum(1 for r in qrows if r["correct"]),
                "doubts": ndo, "uncovered_doubts": sum(1 for x in self._doubts(user_id, doc_id) if not x["covered"]), "recommendations": recs,
                "lang": d["lang"]}

    def _rec_text(self, recs: list[dict[str, Any]], lang: str) -> str:
        if not recs:
            return tr(lang, "no_progress")
        lines = []
        for r in recs[:5]:
            key = {"weak_quiz": "reason_weak_quiz", "asked_doubt": "reason_asked_doubt", "not_started": "reason_not_started", "keep_practising": "reason_keep_practising"}.get(r["reason"])
            why = tr(lang, "reason_prereq_for", for_label=r.get("for_label", "")) if r["reason"] == "prereq_for" else tr(lang, key or "reason_not_started")
            lines.append(f"• **{r['label']}** — {why}")
        return f"**{tr(lang, 'study_next')}:**\n" + "\n".join(lines)

    # ------------------------------------------------------------------- chat
    def history(self, user_id: str, doc_id: str, limit: int = 60) -> list[dict[str, Any]]:
        self._doc(user_id, doc_id)
        rows = self.db.all("SELECT * FROM (SELECT * FROM chat_messages WHERE user_id=? AND doc_id=? ORDER BY id DESC LIMIT ?) ORDER BY id", (user_id, doc_id, limit))
        return [{"id": r["id"], "role": r["role"], "content": r["content"], "meta": jload(r["meta_json"], {}), "ts": r["ts"]} for r in rows]

    def clear_history(self, user_id: str, doc_id: str) -> None:
        self._doc(user_id, doc_id)
        self.db.run("DELETE FROM chat_messages WHERE user_id=? AND doc_id=?", (user_id, doc_id))

    def chat(self, user_id: str, doc_id: str, message: str, ui_lang: str = "en", focus_node: str | None = None, auto_lang: bool = True,
             on_delta: Any = None) -> dict[str, Any]:
        message = (message or "").strip()
        if not message:
            raise _err("Please type a question.")
        if len(message) > 1000:
            raise _err("Please keep questions under 1000 characters.")
        if ui_lang not in SUPPORTED_LANGS and ui_lang not in TARGET_LANGS:
            ui_lang = "en"
        d, ix, nodes, edges = self._load(user_id, doc_id)
        past = [{"role": m["role"], "content": m["content"]} for m in self.history(user_id, doc_id, 8)]
        self.db.run("INSERT INTO chat_messages(user_id,doc_id,role,content,meta_json,ts) VALUES (?,?,?,?,?,?)", (user_id, doc_id, "user", message, "{}", now()))
        rep = self._cached_answer(doc_id, message, ui_lang, focus_node, auto_lang, past)
        if rep is None:
            models = self.models if self._ai_allowed(user_id) else _NoBrainModels(self.models)
            rep = tutor.answer(message, ui_lang, ix, nodes, edges, d["title"], d["lang"] or "en", d["unit_label"] or "Page", focus_node, auto_lang, models, past,
                               on_delta)
            self._store_answer(doc_id, message, ui_lang, focus_node, auto_lang, past, rep)
        elif on_delta is not None:
            on_delta(None)  # nothing to stream: the cached answer arrives whole
        out: dict[str, Any] = {"lang": rep.lang, "intent": rep.intent, "grounded": rep.grounded, "relevance": rep.relevance, "citations": rep.citations, "linked": rep.linked,
                               "focus_node": rep.focus_node, "related": rep.related, **rep.extra}
        text = rep.text
        quiz_payload = None
        if rep.action == "quiz":
            quiz_payload = self.quiz_new(user_id, doc_id, rep.focus_node, 5, rep.lang)
            if quiz_payload["questions"]:
                topic = next((n["label"] for n in nodes if n["id"] == rep.focus_node), d["title"])
                text = tr(rep.lang, "quiz_intro", n=len(quiz_payload["questions"]), topic=topic)
            else:
                text = tr(rep.lang, "quiz_none")
                quiz_payload = None
        elif rep.action == "recommend":
            states = mastery.node_states(self.db, user_id, doc_id)
            recs = mastery.recommend(nodes, edges, states, 5)
            text = self._rec_text(recs, rep.lang)
            out["recommendations"] = recs
        touched: list[str] = []
        for nid, kind in rep.evidence:
            mastery.record_evidence(self.db, user_id, doc_id, nid, kind, {"q": message[:200]})
            touched.append(nid)
        new_doubt = None
        if rep.doubt:
            did = self.db.run("INSERT INTO doubts(user_id,doc_id,label,question,parent_node_id,lang,covered,created_at) VALUES (?,?,?,?,?,?,?,?)",
                              (user_id, doc_id, rep.doubt["label"], message, rep.doubt["parent_node_id"], rep.lang, int(rep.doubt["covered"]), now()))
            new_doubt = {"id": f"d{did}", "label": rep.doubt["label"], "parent": rep.doubt["parent_node_id"] or "root"}
        out.update({"text": text, "quiz": quiz_payload, "touched": touched, "new_doubt": new_doubt})
        self.db.run("INSERT INTO chat_messages(user_id,doc_id,role,content,meta_json,ts) VALUES (?,?,?,?,?,?)",
                    (user_id, doc_id, "assistant", text, jdump({k: out[k] for k in ("lang", "intent", "grounded", "citations", "linked", "related", "touched", "new_doubt", *EXTRA_KEYS) if k in out}), now()))
        if touched or new_doubt:
            st = mastery.node_states(self.db, user_id, doc_id)
            out["states"] = {n: st[n] for n in touched if n in st}
        return out

    # ------------------------------------------------------------------- quiz
    # ---------------------------------------------------- AI budget: answer cache + optional daily limit per student
    _CACHE_TTL = 7 * 86400

    @staticmethod
    def _answer_key(doc_id: str, message: str, ui_lang: str, focus_node: str | None, auto_lang: bool) -> str:
        norm = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", message.lower())).strip()
        return hashlib.sha1(f"{doc_id}|{norm}|{ui_lang}|{focus_node or ''}|{int(auto_lang)}".encode()).hexdigest()

    @staticmethod
    def _cacheable(message: str, past: list[dict[str, str]], focus_node: str | None = None) -> bool:
        """Only self-contained questions. "Explain it again" depends on the conversation, so it is never served from cache;
        "Summarise this topic" with a focused topic is self-contained (the topic is part of the key)."""
        if len(message.split()) < 3:
            return False
        if past and tutor._REFERS_BACK.search(message):
            return bool(focus_node) and not re.search(r"(?i)\b(again|more|previous|above|same|another)\b|మళ్ళీ|ఇంకా|फिर|और", message)
        return True

    def _cached_answer(self, doc_id: str, message: str, ui_lang: str, focus_node: str | None, auto_lang: bool, past: list) -> Any:
        if not self._cacheable(message, past, focus_node):
            return None
        r = self.db.one("SELECT payload_json, created_at FROM answer_cache WHERE key=?", (self._answer_key(doc_id, message, ui_lang, focus_node, auto_lang),))
        if not r or now() - r["created_at"] > self._CACHE_TTL:
            return None
        d = jload(r["payload_json"], {})
        d["evidence"] = [tuple(x) for x in d.get("evidence", [])]
        d["extra"] = {**d.get("extra", {}), "cached": True}
        return tutor.TutorReply(**d)

    def _store_answer(self, doc_id: str, message: str, ui_lang: str, focus_node: str | None, auto_lang: bool, past: list, rep: Any) -> None:
        # keep only real AI answers (no "AI was busy" fallbacks, no quiz/recommend actions)
        if rep.action or rep.extra.get("notice") or not rep.extra.get("engine") or not self._cacheable(message, past, focus_node):
            return
        import dataclasses

        self.db.run("INSERT OR REPLACE INTO answer_cache(key, payload_json, created_at) VALUES (?,?,?)",
                    (self._answer_key(doc_id, message, ui_lang, focus_node, auto_lang), jdump(dataclasses.asdict(rep)), now()))

    def _ai_allowed(self, user_id: str) -> bool:
        """INDICLEARN_AI_DAILY_LIMIT > 0 caps AI answers per student per day (protects a shared free quota on a VPS)."""
        limit = int(os.environ.get("INDICLEARN_AI_DAILY_LIMIT", "0") or 0)
        if limit <= 0:
            return True
        since = now() - 86400
        n = self.db.one("SELECT COUNT(*) c FROM chat_messages WHERE user_id=? AND role='assistant' AND ts>? AND meta_json LIKE '%\"engine\"%'", (user_id, since))["c"]
        return n < limit

    def quiz_new(self, user_id: str, doc_id: str, scope: str | None, n: int, ui_lang: str) -> dict[str, Any]:
        d, ix, nodes, _e = self._load(user_id, doc_id)
        n = max(1, min(int(n or 5), 10))
        if ui_lang not in SUPPORTED_LANGS:
            ui_lang = "en"
        states = mastery.node_states(self.db, user_id, doc_id)
        prio = {nid: (1.0 - s["score"]) * (1.5 if s["status"] in ("revise", "doubt") else 1.0) for nid, s in states.items() if s["status"] != "mastered"}
        seed = int(now() * 1000) % 100000
        brain = getattr(self.models, "brain", None)
        qs = quizbank.pick(self.db, brain, user_id, doc_id, nodes, ix, scope, n, ui_lang, prio, random.Random(seed))
        for q in qs:
            q["prompt"] = tr(ui_lang, "q_mcq")
        if len(qs) < n:  # no AI available (or a tiny scope): trained M4 / rule-based questions fill the rest
            have = {q["node_id"] for q in qs}
            qs += [q for q in quizmod.generate(nodes, scope, n, ui_lang, seed=seed, priority=prio, use_model=False) if q["node_id"] not in have][: n - len(qs)]
        qid = uuid.uuid4().hex[:12]
        self.db.run("INSERT INTO quizzes(id,user_id,doc_id,created_at,scope_json) VALUES (?,?,?,?,?)", (qid, user_id, doc_id, now(), jdump({"scope": scope, "lang": ui_lang})))
        pub = []
        for i, q in enumerate(qs):
            qq = f"{qid}-{i}"
            q["id"], q["idx"] = qq, i
            self.db.run("INSERT INTO quiz_questions(id,quiz_id,idx,node_id,qtype,payload_json) VALUES (?,?,?,?,?,?)", (qq, qid, i, q["node_id"], q["qtype"], jdump(q)))
            pub.append(quizmod.public(q))
        scope_label = next((x["label"] for x in nodes if x["id"] == scope), d["title"]) if scope else d["title"]
        self._start_quiz_bank(doc_id, ui_lang)  # fill the rest of the bank for this language in the background
        return {"quiz_id": qid, "scope": scope, "scope_label": scope_label, "questions": pub}

    def quiz_retry(self, user_id: str, quiz_id: str) -> dict[str, Any]:
        """A new quiz made of the questions the student got wrong (options reshuffled, so position is not memorised)."""
        z = self.db.one("SELECT * FROM quizzes WHERE id=? AND user_id=?", (quiz_id, user_id))
        if z is None:
            raise _err("Quiz not found.", 404)
        wrong = [jload(r["payload_json"], {}) for r in self.db.all("SELECT payload_json FROM quiz_questions WHERE quiz_id=? AND answered=1 AND correct=0 ORDER BY idx", (quiz_id,))]
        if not wrong:
            raise _err("No wrong answers to retry - well done!")
        rng = random.Random()
        qid = uuid.uuid4().hex[:12]
        self.db.run("INSERT INTO quizzes(id,user_id,doc_id,created_at,scope_json) VALUES (?,?,?,?,?)", (qid, user_id, z["doc_id"], now(), z["scope_json"]))
        pub = []
        for i, q in enumerate(wrong):
            if q.get("qtype") != "tf":
                order = list(range(len(q["options"])))
                rng.shuffle(order)
                q["options"], q["answer"] = [q["options"][k] for k in order], order.index(q["answer"])
            q["id"], q["idx"] = f"{qid}-{i}", i
            self.db.run("INSERT INTO quiz_questions(id,quiz_id,idx,node_id,qtype,payload_json) VALUES (?,?,?,?,?,?)", (q["id"], qid, i, q["node_id"], q["qtype"], jdump(q)))
            pub.append(quizmod.public(q))
        return {"quiz_id": qid, "scope": None, "scope_label": "", "questions": pub, "retry": True}

    def quiz_answer(self, user_id: str, quiz_id: str, qid: str, choice: int, ui_lang: str = "en") -> dict[str, Any]:
        row = self.db.one("SELECT q.*, z.user_id, z.doc_id FROM quiz_questions q JOIN quizzes z ON z.id=q.quiz_id WHERE q.id=? AND q.quiz_id=?", (qid, quiz_id))
        if row is None or row["user_id"] != user_id:
            raise _err("Question not found.", 404)
        q = jload(row["payload_json"], {})
        if row["answered"]:
            raise _err("This question was already answered.", 409)
        if not isinstance(choice, int) or not (0 <= choice < len(q["options"])):
            raise _err("Invalid choice.")
        ok = choice == q["answer"]
        self.db.run("UPDATE quiz_questions SET answered=1, correct=?, answered_at=? WHERE id=?", (int(ok), now(), qid))
        mastery.record_evidence(self.db, user_id, row["doc_id"], q["node_id"], "quiz_correct" if ok else "quiz_incorrect", {"qtype": q["qtype"], "quiz": quiz_id})
        st = mastery.node_states(self.db, user_id, row["doc_id"]).get(q["node_id"])
        ex = q["explain"]
        left = self.db.one("SELECT COUNT(*) c, SUM(correct) k FROM quiz_questions WHERE quiz_id=? AND answered=1", (quiz_id,))
        total = self.db.one("SELECT COUNT(*) c FROM quiz_questions WHERE quiz_id=?", (quiz_id,))["c"]
        done = left["c"] >= total
        return {"correct": ok, "answer": q["answer"], "answer_text": q["options"][q["answer"]], "explanation": ex, "node_id": q["node_id"], "state": st,
                "verdict": tr(ui_lang, "correct" if ok else "incorrect"), "answer_label": tr(ui_lang, "answer_was"),
                "done": done, "score": int(left["k"] or 0), "total": total, "summary": tr(ui_lang, "quiz_done", k=int(left["k"] or 0), n=total) if done else None}

    # ------------------------------------------------------------- self-check
    def mark(self, user_id: str, doc_id: str, node_id: str, kind: str) -> dict[str, Any]:
        if kind not in ("self_understood", "self_confused", "viewed"):
            raise _err("Unknown action.")
        _d, _ix, nodes, _e = self._load(user_id, doc_id)
        if not any(n["id"] == node_id for n in nodes):
            raise _err("Unknown topic.", 404)
        mastery.record_evidence(self.db, user_id, doc_id, node_id, kind)
        return {"node_id": node_id, "state": mastery.node_states(self.db, user_id, doc_id).get(node_id)}

    def reset_progress(self, user_id: str, doc_id: str) -> None:
        self._doc(user_id, doc_id)
        for t in ("evidence", "doubts"):
            self.db.run(f"DELETE FROM {t} WHERE user_id=? AND doc_id=?", (user_id, doc_id))

    # ------------------------------------------------------------- motivation
    def me_stats(self, user_id: str) -> dict[str, Any]:
        """XP, level and daily streak - computed from the evidence log, so it can never drift from it."""
        import datetime as _dt

        rows = self.db.all("SELECT kind, ts FROM evidence WHERE user_id=?", (user_id,))
        xp = sum(XP_FOR.get(r["kind"], 0) for r in rows)
        tz = _dt.datetime.now().astimezone().tzinfo  # the server's local day decides the streak
        days = {_dt.datetime.fromtimestamp(r["ts"], tz).date() for r in rows if r["ts"]}
        today = _dt.datetime.fromtimestamp(now(), tz).date()
        streak, d = 0, today if today in days else today - _dt.timedelta(days=1)
        while d in days:
            streak += 1
            d -= _dt.timedelta(days=1)
        level = int((xp / 50) ** 0.5) + 1
        lo, hi = 50 * (level - 1) ** 2, 50 * level ** 2
        return {"xp": xp, "level": level, "level_floor": lo, "next_level_xp": hi, "streak": streak, "active_today": today in days,
                "actions": len(rows)}

    def info(self) -> dict[str, Any]:
        return {"languages": [{"code": c, "name": LANG_NAMES[c]} for c in SUPPORTED_LANGS], "models": self.models.status(), "trained": _neural_status(), "max_upload_mb": self.s.max_upload_mb,
                "allow_guest": self.s.allow_guest, "allow_registration": self.s.allow_registration, "samples": self.samples(), "extensions": sorted(ALLOWED_EXT),
                "speech": self._speech_status(), "ai": self._ai_status()}

    def _ai_status(self) -> dict[str, Any]:
        brain = getattr(self.models, "brain", None)
        ok = bool(brain is not None and brain.available)
        return {"available": ok, "offline_only": bool(ok and brain.first_is_local())}

    def _speech_status(self) -> dict[str, Any]:
        """What the browser should use for the mic / speaker buttons (server voices beat robotic device voices)."""
        from indiclearn.studio import neural, speech

        st = speech.status()
        brain = getattr(self.models, "brain", None)
        hosted_asr = st["asr"] or bool(brain and any(getattr(p, "audio", False) for p in brain.providers))
        local_asr = neural.models_dir().joinpath("whisper_small", "config.json").exists() or neural.models_dir().joinpath("whisper_te_small", "config.json").exists()
        return {"server_tts": bool(st["tts"]), "server_asr": bool(hosted_asr or local_asr), "hosted": st["provider"]}

    def transcribe(self, filename: str, data: bytes, lang: str = "", user_id: str | None = None, doc_id: str | None = None) -> dict[str, Any]:
        """Spoken question -> text + detected language.

        Hosted speech (Gemini) first when configured - it is far better at Telugu than Whisper-small -
        then the local Whisper model. 503 when neither is available.
        """
        from indiclearn.studio import neural

        if len(data) > 25 * 1024 * 1024:
            raise _err("That recording is too long - keep it under about 2 minutes.", 413)
        suffix = Path(filename or "clip.webm").suffix.lower() or ".webm"
        if suffix not in AUDIO_MIME:
            raise _err("Unsupported audio format.", 415)
        from indiclearn.studio import speech

        keyterms: list[str] = []
        if user_id and doc_id:  # the document's own concept names help the recogniser spell them right
            with contextlib.suppress(Exception):
                _d, _ix, nodes, _e = self._load(user_id, doc_id)
                keyterms = [n["label"] for n in nodes if n.get("type") in ("concept", "topic", "section")][:50]
        got = speech.stt(data, Path(filename or "clip.webm").name, AUDIO_MIME[suffix], keyterms)
        if got and got.get("text"):
            return got
        brain = getattr(self.models, "brain", None)
        if brain is not None and brain.available and data:
            got = brain.transcribe(data, AUDIO_MIME[suffix])
            if got and got.get("text"):
                return {"text": got["text"], "lang": got.get("lang", ""), "engine": got.get("engine", "")}
        rec = neural.load_recogniser((lang or "")[:5])
        if rec is None:
            raise _err("Voice input is not set up on this server yet. Either set INDICLEARN_GEMINI_API_KEY, or download the "
                       "offline speech models: python scripts/download_models.py --only whisper_te_small whisper_small", 503)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
            fh.write(data)
            tmp = fh.name
        try:
            text, lang = rec.transcribe(tmp)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
        if not text:
            raise _err("Could not make out any speech in that recording.", 422)
        return {"text": text, "lang": lang or ""}

    # ------------------------------------------------------------ translation
    def translate(self, fields: dict[str, Any], target: str, source: str = "") -> dict[str, Any]:
        """Convert one tutor reply (answer + reasoning + follow-ups ...) into any supported language.

        Chain: hosted/local LLM (keeps markdown, page tags and English technical terms)
        -> IndicTrans2 (local) -> NLLB (local) -> 503.
        """
        if target not in TARGET_LANGS:
            raise _err("Unsupported language.", 422)
        clean = {k: v for k, v in (fields or {}).items() if k in TRANSLATABLE and v}
        if not isinstance(clean.get("text"), str) or not clean["text"].strip():
            raise _err("Nothing to translate.")
        if len(json.dumps(clean, ensure_ascii=False)) > 12000:
            raise _err("That reply is too long to translate.", 413)
        key = hashlib.sha1((target + json.dumps(clean, sort_keys=True, ensure_ascii=False)).encode()).hexdigest()
        with self._lock:
            if key in self._tcache:
                return self._tcache[key]
        out: dict[str, Any] | None = None
        reasons: list[str] = []
        brain = getattr(self.models, "brain", None)
        if brain is not None and brain.available:
            got = brain.translate(clean, target)
            if got:
                out = {k: got.get(k, v) for k, v in clean.items()}
                out["engine"] = got.get("_engine", "llm")
        if out is None and brain is not None:
            reasons.append("AI engines - " + brain.why_unavailable())
        elif out is None:
            reasons.append("AI engines are switched off (INDICLEARN_BRAIN=off)")
        if out is None:
            out, why = self._translate_local(clean, target, source)
            if out is None:
                reasons.append("Offline translators - " + why)
        if out is None:
            raise _err("Could not translate. " + " | ".join(reasons) +
                       " || Fix: put a working key in .env, or install Ollama + run setup_offline.bat, or download IndicTrans2/NLLB.", 503)
        out["lang"] = target
        with self._lock:
            self._tcache[key] = out
            while len(self._tcache) > 256:
                self._tcache.popitem(last=False)
        return out

    def _translate_local(self, fields: dict[str, Any], target: str, source: str = "") -> tuple[dict[str, Any] | None, str]:
        """Offline machine translation (IndicTrans2, else NLLB). Returns (result, reason-if-it-failed).

        Indic <-> Indic (e.g. Telugu -> Hindi) goes through English automatically, so any two of the
        supported languages work as long as the two IndicTrans2 folders (or NLLB) are installed."""
        from indiclearn.studio import neural
        from indiclearn.studio.lang import detect_language
        from indiclearn.studio.llm import NLLB_CODES

        base = target.split("-")[0]
        if "-" in target:
            return None, "Romanized output (Telugu/Hindi in English letters) needs an LLM - add a key or install Ollama"
        src = (source or "").split("-")[0] or detect_language(fields["text"]).base
        if src == base:
            return None, f"the reply is already in {base}"
        it = neural.load_indictrans()
        nl = getattr(self.models, "translator", None)
        use_it = bool(it and it.can(src, base))
        use_nl = nl is not None and src in NLLB_CODES and base in NLLB_CODES
        if not (use_it or use_nl):
            why = []
            if it is None:
                why.append("IndicTrans2: " + neural.LOAD_ERRORS.get("indictrans2", "not installed"))
            else:
                why.append(f"IndicTrans2 has no {src}->{base} model")
            if nl is None:
                why.append("NLLB: not installed (python scripts/setup_offline.py --nllb)")
            return None, "; ".join(why)

        def tx(texts: list[str]) -> list[str]:
            if not texts:
                return texts
            if use_it:
                try:
                    return it.translate(texts, src, base, strict=True)  # type: ignore[union-attr]
                except Exception:
                    if not use_nl:
                        raise
            return [nl.translate(t_, src, base) for t_ in texts]  # type: ignore[union-attr]

        def tx_md(md_text: str) -> str:
            # translate line by line, keeping bullets; bold markers would confuse the MT model
            lines = md_text.split("\n")
            body = [re.sub(r"^\s*[-•*]\s+", "", ln).replace("**", "") for ln in lines]
            idx = [i for i, ln in enumerate(body) if ln.strip()]
            done = tx([body[i] for i in idx])
            for i, t_ in zip(idx, done, strict=False):
                lines[i] = ("- " if re.match(r"^\s*[-•*]\s+", lines[i]) else "") + t_
            return "\n".join(lines)

        try:
            out: dict[str, Any] = {"text": tx_md(fields["text"]), "engine": "indictrans2" if use_it else "nllb"}
            if src != "en" and base != "en" and use_it:
                out["engine"] = "indictrans2 (via English)"
            for k in ("reasoning", "follow_ups"):
                if isinstance(fields.get(k), list):
                    out[k] = tx([str(x) for x in fields[k]])
            if isinstance(fields.get("beyond_notes"), str):
                out["beyond_notes"] = tx([fields["beyond_notes"]])[0]
            if isinstance(fields.get("key_terms"), list):
                means = tx([str(x.get("meaning", "")) for x in fields["key_terms"]])
                out["key_terms"] = [{**x, "meaning": m} for x, m in zip(fields["key_terms"], means, strict=False)]
        except Exception as e:
            return None, f"offline translator crashed: {str(e)[:160]}"
        if out["text"].strip() == fields["text"].strip():
            return None, "the offline translator returned the text unchanged"
        return out, ""

    # ------------------------------------------------------------------ speech
    def speak(self, text: str, lang: str) -> bytes:
        """Read a reply aloud with local TTS. The browser's own voices are tried first, so this
        only runs when the device has no voice for the language (common for Telugu on Windows).

        A Telugu voice cannot pronounce English letters (it just skips them), and replies are full of
        English technical terms. So the text is split into runs: Latin runs are read by the English
        voice, everything else by the language's voice, and the pieces are joined into one WAV."""
        from indiclearn.studio import neural

        from indiclearn.studio import speech

        base = (lang or "en").split("-")[0]
        clean = re.sub(r"\[(?:p\.|page |slide |part )\s*\d+(?:\s*,\s*(?:p\.)?\s*\d+)*\]|[*_`#>]", "", text or "")
        clean = re.sub(r"\s+", " ", clean).strip()[:3000]
        hosted = speech.tts(clean, base)  # one natural voice for Telugu-English code-mix (Sarvam Bulbul)
        if hosted:
            return hosted[0] if len(hosted) == 1 else join_wavs(hosted, gap_ms=60)
        clean = clean[:1200]
        runs = split_script_runs(clean) if base != "en" else [("en", clean)]
        native = neural.load_speaker(base) if base != "en" else None
        parler = neural.load_parler() if base in neural.PARLER_LANGS else None
        english = neural.load_speaker("en")
        eng_parler = neural.load_parler() if "en" in neural.PARLER_LANGS else None
        if base != "en" and native is None and parler is None:
            raise _err(f"No voice installed for this language on the server. Download one with: "
                       f"python scripts/download_models.py --only {neural.TTS_FOLDERS.get(base, 'mms_tts_tel')}", 503)
        pieces: list[bytes] = []
        for kind, chunk in runs:
            if not re.search(r"\w", chunk):
                continue
            wav = b""
            for part in _chunks(chunk, 280):
                if kind == "en" or base == "en":
                    w = english.wav(part) if english is not None else b""
                    if not w and eng_parler is not None:
                        w = eng_parler.wav(part, "en")
                else:
                    w = parler.wav(part, base) if parler is not None else b""
                    if not w and native is not None:
                        w = native.wav(part)
                if w:
                    pieces.append(w)
            _ = wav
        if not pieces:
            raise _err("Could not synthesise speech for that text.", 422)
        return pieces[0] if len(pieces) == 1 else join_wavs(pieces)


_LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z0-9'\-.]*(?:[ ,]+[A-Za-z0-9][A-Za-z0-9'\-.]*)*")


def split_script_runs(text: str) -> list[tuple[str, str]]:
    """[("native", "కన్వల్యూషన్ అంటే "), ("en", "Graph Neural Networks"), ...] in reading order."""
    out: list[tuple[str, str]] = []
    pos = 0
    for m in _LATIN_RUN.finditer(text):
        if m.start() > pos:
            out.append(("native", text[pos : m.start()]))
        out.append(("en", m.group(0)))
        pos = m.end()
    if pos < len(text):
        out.append(("native", text[pos:]))
    return out


def _chunks(text: str, n: int) -> list[str]:
    text = text.strip()
    if len(text) <= n:
        return [text]
    parts, cur = [], ""
    for w in text.split(" "):
        if cur and len(cur) + len(w) + 1 > n:
            parts.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    return parts + ([cur] if cur else [])


def join_wavs(wavs: list[bytes], gap_ms: int = 120) -> bytes:
    """Concatenate mono 16-bit WAVs (resampling to the first one's rate) with a short pause between."""
    import io
    import wave

    import numpy as np

    rate, chunks = 0, []
    for b in wavs:
        with wave.open(io.BytesIO(b), "rb") as w:
            r = w.getframerate()
            x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32)
        rate = rate or r
        if r != rate and len(x):
            x = np.interp(np.linspace(0, len(x) - 1, int(len(x) * rate / r)), np.arange(len(x)), x)
        chunks += [x, np.zeros(int(rate * gap_ms / 1000), dtype=np.float32)]
    pcm = np.concatenate(chunks).clip(-32768, 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()
