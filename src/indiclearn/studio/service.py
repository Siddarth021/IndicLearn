"""Application service: orchestrates ingest -> graph -> reader -> tutor -> quiz -> progress.

All methods take the authenticated ``user_id`` and verify ownership, so users are isolated from one
another.  Heavy work (parsing / graph building) runs on a small thread pool and is polled via the
document ``status`` field - the same shape works unchanged if the pool is later replaced by a queue.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from indiclearn.studio import mastery, tutor
from indiclearn.studio import quiz as quizmod
from indiclearn.studio.auth import AuthError, AuthService
from indiclearn.studio.config import Settings
from indiclearn.studio.db import Database, jdump, jload, now
from indiclearn.studio.i18n import tr
from indiclearn.studio.kgraph import build_knowledge_graph
from indiclearn.studio.lang import LANG_NAMES, SUPPORTED_LANGS
from indiclearn.studio.llm import Models
from indiclearn.studio.parse import parse_file
from indiclearn.studio.retrieval import DocIndex

log = logging.getLogger("indiclearn.service")
ALLOWED_EXT = {".pdf", ".pptx", ".docx", ".md", ".txt", ".markdown"}
SAMPLES = [
    ("signals_and_systems_notes.md", "Signals & Systems (English)", "en"),
    ("signals_and_systems_te.md", "సిగ్నల్స్ & సిస్టమ్స్ (తెలుగు)", "te"),
    ("signals_and_systems_hi.md", "सिग्नल और सिस्टम (हिन्दी)", "hi"),
    ("signals_sanskrit.md", "संकेत-प्रणाली (संस्कृतम्)", "sa"),
]


def _err(msg: str, status: int = 400) -> AuthError:
    return AuthError(msg, status)


class StudioService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or Settings()
        self.s.ensure_dirs()
        self.db = Database(self.s.db_path)
        self.auth = AuthService(self.db, self.s.token_ttl_days)
        self.models = Models(self.s)
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ingest")
        self._cache: OrderedDict[str, tuple[DocIndex, list[dict[str, Any]], list[dict[str, Any]]]] = OrderedDict()
        self._lock = threading.Lock()
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
                "stats": jload(r["stats_json"], {}), "summary": r["summary"], "is_sample": bool(r["is_sample"])}

    # ------------------------------------------------------------------ ingest
    def list_docs(self, user_id: str) -> list[dict[str, Any]]:
        return [self._doc_public(r) for r in self.db.all("SELECT * FROM documents WHERE user_id=? ORDER BY created_at DESC", (user_id,))]

    def add_upload(self, user_id: str, filename: str, data: bytes, is_sample: bool = False) -> dict[str, Any]:
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            raise _err(f"Unsupported file type {ext or '(none)'}. Upload PDF, PPTX, DOCX, Markdown or TXT.", 415)
        if not data:
            raise _err("The file is empty.")
        if len(data) > self.s.max_upload_mb * 1024 * 1024:
            raise _err(f"File is larger than {self.s.max_upload_mb} MB.", 413)
        n = self.db.one("SELECT COUNT(*) c FROM documents WHERE user_id=?", (user_id,))["c"]
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

    def _build(self, doc_id: str, path: Path, filename: str) -> None:
        try:
            pd = parse_file(path, filename)
            kg = build_knowledge_graph(pd)
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
        toc = [{"id": n["id"], "label": n["label"], "level": n["level"], "page": n["page"], "type": n["type"]} for n in nodes if n["type"] in ("chapter", "section", "topic") and not n.get("is_term") or n["type"] in ("chapter", "section")]
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

    def chat(self, user_id: str, doc_id: str, message: str, ui_lang: str = "en", focus_node: str | None = None, auto_lang: bool = True) -> dict[str, Any]:
        message = (message or "").strip()
        if not message:
            raise _err("Please type a question.")
        if len(message) > 1000:
            raise _err("Please keep questions under 1000 characters.")
        if ui_lang not in SUPPORTED_LANGS:
            ui_lang = "en"
        d, ix, nodes, edges = self._load(user_id, doc_id)
        self.db.run("INSERT INTO chat_messages(user_id,doc_id,role,content,meta_json,ts) VALUES (?,?,?,?,?,?)", (user_id, doc_id, "user", message, "{}", now()))
        rep = tutor.answer(message, ui_lang, ix, nodes, edges, d["title"], d["lang"] or "en", d["unit_label"] or "Page", focus_node, auto_lang, self.models)
        out: dict[str, Any] = {"lang": rep.lang, "intent": rep.intent, "grounded": rep.grounded, "relevance": rep.relevance, "citations": rep.citations, "linked": rep.linked,
                               "focus_node": rep.focus_node, "related": rep.related}
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
                    (user_id, doc_id, "assistant", text, jdump({k: out[k] for k in ("lang", "intent", "grounded", "citations", "linked", "related", "touched", "new_doubt")}), now()))
        if touched or new_doubt:
            st = mastery.node_states(self.db, user_id, doc_id)
            out["states"] = {n: st[n] for n in touched if n in st}
        return out

    # ------------------------------------------------------------------- quiz
    def quiz_new(self, user_id: str, doc_id: str, scope: str | None, n: int, ui_lang: str) -> dict[str, Any]:
        d, _ix, nodes, _e = self._load(user_id, doc_id)
        n = max(1, min(int(n or 5), 10))
        if ui_lang not in SUPPORTED_LANGS:
            ui_lang = "en"
        states = mastery.node_states(self.db, user_id, doc_id)
        prio = {nid: (1.0 - s["score"]) * (1.5 if s["status"] in ("revise", "doubt") else 1.0) for nid, s in states.items() if s["status"] != "mastered"}
        qs = quizmod.generate(nodes, scope, n, ui_lang, seed=int(now() * 1000) % 100000, priority=prio)
        qid = uuid.uuid4().hex[:12]
        self.db.run("INSERT INTO quizzes(id,user_id,doc_id,created_at,scope_json) VALUES (?,?,?,?,?)", (qid, user_id, doc_id, now(), jdump({"scope": scope, "lang": ui_lang})))
        pub = []
        for i, q in enumerate(qs):
            qq = f"{qid}-{i}"
            q["id"], q["idx"] = qq, i
            self.db.run("INSERT INTO quiz_questions(id,quiz_id,idx,node_id,qtype,payload_json) VALUES (?,?,?,?,?,?)", (qq, qid, i, q["node_id"], q["qtype"], jdump(q)))
            pub.append(quizmod.public(q))
        scope_label = next((x["label"] for x in nodes if x["id"] == scope), d["title"]) if scope else d["title"]
        return {"quiz_id": qid, "scope": scope, "scope_label": scope_label, "questions": pub}

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

    def info(self) -> dict[str, Any]:
        return {"languages": [{"code": c, "name": LANG_NAMES[c]} for c in SUPPORTED_LANGS], "models": self.models.status(), "max_upload_mb": self.s.max_upload_mb,
                "allow_guest": self.s.allow_guest, "allow_registration": self.s.allow_registration, "samples": self.samples(), "extensions": sorted(ALLOWED_EXT)}
