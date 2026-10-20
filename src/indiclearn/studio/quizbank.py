"""AI-written quiz bank: built once per document in the background, served instantly.

Why: the rule-based quiz (quiz.py) can only blank out a word or shuffle definitions, and a student
waiting 8 seconds for an LLM every time they press "Start quiz" gets irritated.  So right after an
upload is ready, the brain writes 2 understanding-level MCQs per concept (in batches) and they are
stored in ``quiz_bank``.  A quiz is then just a pick from the bank, weighted towards the student's
weak concepts, with options re-shuffled every time.

Missing language / concept -> generated on demand (one call) and cached.  No LLM at all -> the
caller falls back to the trained M4 model / the rule-based generator.
"""

from __future__ import annotations

import logging
import random
import threading
import uuid
from typing import Any

from indiclearn.studio import brain as B
from indiclearn.studio.db import Database, jdump, jload, now

log = logging.getLogger("indiclearn.quizbank")

BATCH = 5            # concepts per LLM call
MAX_NODES = 30       # concepts per document in the background bank
QUIZ_TYPES = ("concept", "topic")
_building: set[tuple[str, str]] = set()
_attempted: dict[tuple[str, str], float] = {}  # (doc, lang) -> when the bank was last (re)built
RETRY_AFTER_S = 600
_lock = threading.Lock()


def quizzable(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Concepts first (by importance), then sections that have a real definition."""
    banked = [n for n in nodes if n.get("quiz")]  # course topics: they carry the checked question bank
    concepts = [n for n in nodes if n.get("type") == "concept"]
    topics = [n for n in nodes if n.get("type") == "topic" and len((n.get("definition") or "").strip()) >= 40 and not n.get("quiz")]
    concepts.sort(key=lambda n: -float(n.get("importance") or n.get("weight") or 0.5))
    return banked + (concepts + topics)[: max(0, MAX_NODES - len(banked))]


def scope_ids(nodes: list[dict[str, Any]], scope: str | None) -> set[str] | None:
    if not scope or scope == "root":
        return None
    kids: dict[str, list[str]] = {}
    for n in nodes:
        kids.setdefault(n.get("parent") or "", []).append(n["id"])
    out, stack = {scope}, [scope]
    while stack:
        for c in kids.get(stack.pop(), []):
            if c not in out:
                out.add(c)
                stack.append(c)
    by_id = {n["id"]: n for n in nodes}
    return {i for i in by_id if i in out or by_id[i].get("section_id") in out}


def notes_for(node: dict[str, Any], ix: Any, k: int = 2) -> list[tuple[int, str]]:
    chunks = []
    if ix is not None:
        for ci in getattr(ix, "concept_chunks", {}).get(node["id"], [])[:k]:
            chunks.append(ix.chunks[ci])
        if len(chunks) < k:
            chunks += [c for c in ix.chunks if c.get("section_id") == node["id"]][: k - len(chunks)]
    out = [(int(c.get("page") or 1), c["text"]) for c in chunks]
    if not out and node.get("definition"):
        out = [(int(node.get("page") or 1), node["definition"])]
    return out


def _payload(n: dict[str, Any], ix: Any) -> dict[str, Any]:
    return {"id": n["id"], "label": n.get("full_label") or n["label"], "definition": n.get("definition") or "", "notes": notes_for(n, ix)}


def generate(db: Database, brain: Any, doc_id: str, nodes: list[dict[str, Any]], ix: Any, lang: str, per: int = 2) -> int:
    """Write questions for *nodes* and store them. Returns how many were stored."""
    if brain is None or not getattr(brain, "available", False) or not nodes:
        return 0
    stored = 0
    local = bool(getattr(brain, "first_is_local", lambda: False)())
    step = 2 if local else BATCH  # a local model answers students in between short batches
    if local:
        nodes = nodes[:12]
    for i in range(0, len(nodes), step):
        batch = nodes[i : i + step]
        try:
            got = brain.quiz([_payload(n, ix) for n in batch], lang, per)
        except Exception as e:  # pragma: no cover - never let the bank break an upload
            log.warning("quiz bank batch failed: %s", e)
            got = None
        if not got:
            continue
        qs, engine = got
        notes_of = {n["id"]: " ".join(t for _p, t in notes_for(n, ix))[:700] for n in batch}
        verdict = brain.verify_mcqs([{"q": q["q"], "options": q["options"], "notes": notes_of.get(str(q["concept_id"]), "")} for q in qs])
        if verdict is not None:  # keep only questions an independent pass answers the same way
            kept = [q for q, v in zip(qs, verdict) if v == q["answer"]]
            log.info("quiz bank: %d/%d questions passed the answer-key check", len(kept), len(qs))
            qs = kept
        rows = []
        for q in qs:
            qid = uuid.uuid4().hex[:16]
            rows.append((qid, doc_id, str(q["concept_id"]), lang, qid, jdump({k: q[k] for k in ("q", "options", "answer", "why", "page")}), engine, now()))
        with db.conn() as c:
            c.executemany("INSERT OR IGNORE INTO quiz_bank(id,doc_id,node_id,lang,src_id,payload_json,engine,created_at) VALUES (?,?,?,?,?,?,?,?)", rows)
        stored += len(rows)
    return stored


def generate_m4(db: Database, doc_id: str, nodes: list[dict[str, Any]], ix: Any, limit: int = 12) -> int:
    """No LLM at all: the trained M4 model writes one MCQ per concept (slow on a CPU, so only in the background)."""
    from indiclearn.studio import neural
    from indiclearn.studio.quiz import _parse_generated

    m = neural.load_quizzer()
    if m is None:
        return 0
    stored = 0
    for n in nodes[:limit]:
        passage = " ".join(t for _p, t in notes_for(n, ix, 1))[:1200]
        if len(passage) < 80:
            continue
        prompt = f"question: {n.get('full_label') or n['label']} | passage: {passage}"
        parsed = _parse_generated(m.generate(prompt, 160))  # beam search first (best quality) ...
        if not parsed:  # ... then a few sampled tries; keep the first well-formed one (4 distinct options)
            parsed = next((p for p in map(_parse_generated, m.generate_many(prompt, 4, 160)) if p), None)
        if not parsed:
            continue
        q, opts, ans = parsed
        qid = uuid.uuid4().hex[:16]
        payload = {"q": q, "options": opts, "answer": ans, "why": n.get("definition") or "", "page": n.get("page", 1)}
        db.run("INSERT OR IGNORE INTO quiz_bank(id,doc_id,node_id,lang,src_id,payload_json,engine,created_at) VALUES (?,?,?,?,?,?,?,?)",
               (qid, doc_id, n["id"], "en", qid, jdump(payload), "m4", now()))
        stored += 1
    return stored


def build_in_background(db: Database, brain: Any, doc_id: str, nodes: list[dict[str, Any]], ix: Any, langs: list[str]) -> None:
    """Fire-and-forget bank build for each language (skips a doc/lang that is already being built)."""
    todo = quizzable(nodes)
    if brain is None or not getattr(brain, "available", False):
        key = (doc_id, "m4")
        with _lock:
            if key in _building or now() - _attempted.get(key, 0.0) < RETRY_AFTER_S:
                return
            _building.add(key)
            _attempted[key] = now()

        def work_m4() -> None:
            try:
                if not db.one("SELECT 1 FROM quiz_bank WHERE doc_id=? LIMIT 1", (doc_id,)):
                    log.info("quiz bank (M4): %d questions for %s", generate_m4(db, doc_id, todo, ix), doc_id)
            except Exception as e:  # pragma: no cover
                log.warning("M4 quiz bank failed for %s: %s", doc_id, e)
            finally:
                with _lock:
                    _building.discard(key)

        threading.Thread(target=work_m4, name=f"quizbank-m4-{doc_id}", daemon=True).start()
        return

    def work() -> None:
        for lang in langs:
            key = (doc_id, lang)
            with _lock:
                if key in _building or now() - _attempted.get(key, 0.0) < RETRY_AFTER_S:
                    continue
                _building.add(key)
                _attempted[key] = now()
            try:
                have = {r["node_id"] for r in db.all("SELECT DISTINCT node_id FROM quiz_bank WHERE doc_id=? AND lang=?", (doc_id, lang))}
                missing = [n for n in todo if n["id"] not in have]
                n = generate(db, brain, doc_id, missing, ix, lang)
                log.info("quiz bank: %d questions for %s [%s]", n, doc_id, lang)
            except Exception as e:  # pragma: no cover
                log.warning("quiz bank build failed for %s: %s", doc_id, e)
            finally:
                with _lock:
                    _building.discard(key)

    threading.Thread(target=work, name=f"quizbank-{doc_id}", daemon=True).start()


def pick(db: Database, brain: Any, user_id: str, doc_id: str, nodes: list[dict[str, Any]], ix: Any, scope: str | None, n: int,
         lang: str, priority: dict[str, float], rng: random.Random) -> list[dict[str, Any]]:
    """Up to *n* bank questions (generating the missing ones on demand), ready for quiz_questions."""
    by_id = {x["id"]: x for x in nodes}
    allowed = scope_ids(nodes, scope)
    pool_nodes = [x for x in quizzable(nodes) if allowed is None or x["id"] in allowed]
    if not pool_nodes:
        return []
    rows = [r for r in db.all("SELECT * FROM quiz_bank WHERE doc_id=? AND lang=?", (doc_id, lang)) if r["node_id"] in {x["id"] for x in pool_nodes}]
    if not rows and lang != "en" and not (brain is not None and getattr(brain, "available", False)):
        # offline with no LLM: the M4 questions are English - better than none
        rows = [r for r in db.all("SELECT * FROM quiz_bank WHERE doc_id=? AND lang='en'", (doc_id,)) if r["node_id"] in {x["id"] for x in pool_nodes}]
    covered = {r["node_id"] for r in rows}
    if len(covered) < min(n, len(pool_nodes)) and brain is not None and getattr(brain, "available", False):
        need = [x for x in sorted(pool_nodes, key=lambda x: -priority.get(x["id"], 0.5)) if x["id"] not in covered][: max(1, n - len(covered))]
        if generate(db, brain, doc_id, need[:6], ix, lang, per=1):
            rows = [r for r in db.all("SELECT * FROM quiz_bank WHERE doc_id=? AND lang=?", (doc_id, lang)) if r["node_id"] in {x["id"] for x in pool_nodes}]
    if not rows:
        return []
    # questions this student already saw recently go to the back of the queue
    seen = {jload(r["payload_json"], {}).get("bank_id") for r in db.all(
        "SELECT qq.payload_json FROM quiz_questions qq JOIN quizzes z ON z.id=qq.quiz_id WHERE z.user_id=? AND z.doc_id=? ORDER BY z.created_at DESC LIMIT 60",
        (user_id, doc_id))}

    def weight(r: Any) -> float:
        return priority.get(r["node_id"], 0.5) + rng.random() * 0.6 - (1.0 if r["id"] in seen else 0.0)

    rows.sort(key=weight, reverse=True)
    out, used_nodes = [], set()
    for second_round in (False, True):  # one question per concept first, then a second one if needed
        for r in rows:
            if len(out) >= n:
                break
            if (r["node_id"] in used_nodes) != second_round or any(o["bank_id"] == r["id"] for o in out):
                continue
            q = jload(r["payload_json"], {})
            node = by_id.get(r["node_id"])
            if not node or not q.get("options"):
                continue
            order = list(range(len(q["options"])))
            rng.shuffle(order)
            opts = [q["options"][i] for i in order]
            out.append({"qtype": "mcq", "node_id": node["id"], "bank_id": r["id"], "stem": q["q"], "options": opts,
                        "answer": order.index(int(q["answer"])), "engine": r["engine"],
                        "explain": {"definition": B.fix_option_refs(q.get("why") or "", q["options"]) or node.get("definition") or "", "label": node.get("full_label") or node["label"],
                                    "page": q.get("page") or node.get("page", 1)}})
            used_nodes.add(r["node_id"])
    return out
