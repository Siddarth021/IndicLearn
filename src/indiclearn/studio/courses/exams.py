"""Exam-paper service: notes coverage -> topics -> (fresh questions from the notes) -> paper -> stored -> PDF."""

from __future__ import annotations

import logging
import random
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any

from indiclearn.studio.courses import papers as P
from indiclearn.studio.courses import pdf, qgen
from indiclearn.studio.db import Database, jdump, jload, now

log = logging.getLogger("indiclearn.exams")
FRESH_TIMEOUT_S = 50  # never keep a student waiting longer than this for questions from their notes
MIN_TOPICS_FOR_SUBJECT = 3  # an upload that touches fewer OS topics than this is not OS notes


class ExamService:
    def __init__(self, db: Database, models: Any) -> None:
        self.db, self.models = db, models

    # ------------------------------------------------------------------ notes
    def _notes(self, user_id: str, units: list[str], doc_ids: list[str] | None, subject: str = "os") -> tuple[str, list[dict], dict]:
        """Text of the student's uploads that are about this subject (+ which docs were used). Course lessons are not uploads."""
        q = "SELECT id, title FROM documents WHERE user_id=? AND status='ready' AND course IS NULL"
        docs = self.db.all(q, (user_id,))
        if doc_ids is not None:
            docs = [d for d in docs if d["id"] in set(doc_ids)]
        used, texts = [], []
        for d in docs:
            text = " ".join(r["text"] for r in self.db.all("SELECT text FROM chunks WHERE doc_id=? ORDER BY idx", (d["id"],)))
            cov = P.coverage(text, [], subject)  # all units: is this upload about this subject at all?
            if len(cov["covered"]) >= MIN_TOPICS_FOR_SUBJECT:
                used.append({"id": d["id"], "title": d["title"]})
                texts.append(text)
        joined = "\n".join(texts)
        return joined, used, (P.coverage(joined, units, subject) if joined else {"covered": [], "missing": []})

    @staticmethod
    def _passages(text: str, topic: dict, limit: int = 3500) -> str:
        """Sentences of the notes that mention the topic's keywords, in order, up to `limit` characters."""
        sents = re.split(r"(?<=[.!?])\s+", text)
        kws = [k.lower() for k in topic["keywords"]]
        hits = [s for s in sents if any(k in s.lower() for k in kws)]
        out, n = [], 0
        for s in hits:
            if n + len(s) > limit:
                break
            out.append(s)
            n += len(s)
        return " ".join(out)

    def _fresh(self, notes: str, topic_ids: list[str], bp: dict) -> list[dict]:
        """A few new questions written from the student's own notes (checked like the bank), best effort and time-boxed."""
        brain = getattr(self.models, "brain", None)
        if not notes or brain is None or not getattr(brain, "available", False):
            return []
        kinds = {r["qtype"] for r in bp["rows"]}
        want = {"fill": int("fill" in kinds or "auto" in kinds), "mcq": int("mcq" in kinds or "auto" in kinds),
                "short": 1 if kinds & {"short", "auto"} else 0, "medium": 1 if kinds & {"long", "auto"} else 0, "long": 0}
        if not any(want.values()):
            return []
        low = notes.lower()
        scored = sorted(((sum(P._kw_hits(low, k) for k in t["keywords"]), u, t) for u, t in P.topics_of(None, bp["subject"]) if t["id"] in topic_ids
                         and not t.get("numerical")), key=lambda x: -x[0])
        picks = [(u, t) for s, u, t in scored[:3] if s >= 3]
        if not picks:
            return []

        def one(u: dict, t: dict) -> list[dict]:
            qs = qgen.gen_topic(brain, u, t, 1, notes=self._passages(notes, t), want=want, tries=1, subject=bp["subject"],
                                course=P.syllabus(bp["subject"])["title"])
            return [{**q, "source": "notes"} for q in qgen.check(brain, qs)] if qs else []

        ex = ThreadPoolExecutor(len(picks))
        futs = [ex.submit(one, u, t) for u, t in picks]
        done, _ = wait(futs, timeout=FRESH_TIMEOUT_S)
        ex.shutdown(wait=False, cancel_futures=True)
        out = []
        for f in done:
            try:
                out += f.result()
            except Exception as e:  # pragma: no cover - fresh questions are a bonus, never a failure
                log.warning("fresh questions failed: %s", e)
        return out

    # ------------------------------------------------------------------ papers
    def new(self, user_id: str, blueprint: dict[str, Any], doc_ids: list[str] | None = None) -> dict[str, Any]:
        bp = P.validate(blueprint)
        notes, used_docs, cov = self._notes(user_id, bp["units"], doc_ids, bp["subject"])
        ticked = [t["id"] for _u, t in P.topics_of(bp["units"], bp["subject"])]
        topics = [t for t in ticked if t in set(cov["covered"])] if used_docs and cov["covered"] else ticked
        fresh = self._fresh(notes, topics, bp) if used_docs else []
        seen: set[str] = set()
        for r in self.db.all("SELECT payload_json FROM papers WHERE user_id=? ORDER BY created_at DESC LIMIT 20", (user_id,)):
            seen |= {q["id"] for q in (jload(r["payload_json"], {}) or {}).get("questions", [])}
        paper = P.assemble(bp, topics, fresh, seen, seed=random.randrange(1 << 30))
        pid = uuid.uuid4().hex[:16]
        syl = P.syllabus(bp["subject"])
        paper.update(id=pid, blueprint=bp, course=bp["subject"], course_title=syl["title"], created_at=now(),
                     sources={"docs": used_docs, "topics": topics, "from_notes": bool(used_docs and cov["covered"]),
                              "fresh": sum(1 for q in paper["questions"] if q.get("source") == "notes"),
                              "missing_in_notes": [t for t in ticked if t not in set(cov["covered"])] if used_docs else []})
        self.db.run("INSERT INTO papers(id,user_id,subject,name,created_at,payload_json) VALUES (?,?,?,?,?,?)",
                    (pid, user_id, bp["subject"], bp["name"], paper["created_at"], jdump(paper)))
        return P.public(paper, answers=False)

    def _load(self, user_id: str, pid: str) -> dict[str, Any]:
        r = self.db.one("SELECT payload_json FROM papers WHERE id=? AND user_id=?", (pid, user_id))
        if not r:
            raise P.PaperError("Paper not found.")
        return jload(r["payload_json"], {})

    def get(self, user_id: str, pid: str, answers: bool = False) -> dict[str, Any]:
        return P.public(self._load(user_id, pid), answers)

    def pdf(self, user_id: str, pid: str, answers: bool = False) -> tuple[bytes, str]:
        paper = self._load(user_id, pid)
        short = (P.syllabus(paper.get("course") or "os").get("short") or "OS").replace(" ", "")
        name = re.sub(r"[^A-Za-z0-9]+", "_", f"{short}_{paper['blueprint']['name']}_{'answer_key' if answers else 'paper'}").strip("_") + ".pdf"
        return pdf.render(paper, answers), name

    def list(self, user_id: str) -> list[dict[str, Any]]:
        out = []
        for r in self.db.all("SELECT id, name, subject, created_at, payload_json FROM papers WHERE user_id=? ORDER BY created_at DESC LIMIT 30", (user_id,)):
            p = jload(r["payload_json"], {}) or {}
            out.append({"id": r["id"], "name": r["name"], "subject": r["subject"], "created_at": r["created_at"], "total": p.get("total"),
                        "questions": len(p.get("questions", [])), "minutes": (p.get("blueprint") or {}).get("minutes")})
        return out

    def delete(self, user_id: str, pid: str) -> None:
        self.db.run("DELETE FROM papers WHERE id=? AND user_id=?", (pid, user_id))
