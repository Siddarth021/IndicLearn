"""OS course lessons: index, one lesson (English or Telugu) with a live solved example and a quick check, progress."""

from __future__ import annotations

import json
import random
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from indiclearn.studio.courses import oscalc
from indiclearn.studio.courses import papers as P
from indiclearn.studio.db import Database, jdump, jload, now

HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=2)
def _load(mtime: float) -> dict[str, dict]:
    path = HERE / "os_lessons.jsonl"
    return {d["topic"]: d for d in (json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip())}


def lessons() -> dict[str, dict]:
    path = HERE / "os_lessons.jsonl"
    return _load(path.stat().st_mtime) if path.exists() else {}


def index(db: Database, user_id: str) -> dict[str, Any]:
    syl = P.syllabus("os")
    done = {r["topic"]: r["status"] for r in db.all("SELECT topic, status FROM lesson_progress WHERE user_id=? AND subject='os'", (user_id,))}
    ls = lessons()
    units = []
    for u in syl["units"]:
        units.append({"id": u["id"], "title": u["title"], "topics": [
            {"id": t["id"], "label": t["label"], "has_lesson": t["id"] in ls, "telugu": bool((ls.get(t["id"]) or {}).get("te")),
             "summary": ((ls.get(t["id"]) or {}).get("en") or {}).get("summary", ""), "numerical": bool(t.get("numerical")),
             "status": done.get(t["id"], "new")} for t in u["topics"]]})
    total = sum(len(u["topics"]) for u in units)
    return {"id": "os", "title": syl["title"], "units": units, "done": sum(1 for s in done.values() if s == "learned"), "total": total}


def _telugu(db: Database, d: dict, brain: Any) -> dict | None:
    """Telugu lesson: built in at build time, else translated once (by whichever AI engine answers) and cached for everyone."""
    if d.get("te"):
        return d["te"]
    r = db.one("SELECT payload_json FROM lesson_i18n WHERE subject='os' AND topic=? AND lang='te'", (d["topic"],))
    if r:
        return jload(r["payload_json"])
    if brain is None or not getattr(brain, "available", False):
        return None
    with _TE_LOCKS.setdefault(d["topic"], threading.Lock()):  # two students opening it at once: translate once
        r = db.one("SELECT payload_json FROM lesson_i18n WHERE subject='os' AND topic=? AND lang='te'", (d["topic"],))
        if r:
            return jload(r["payload_json"])
        te = translate(brain, d["en"], "te")
        if te:
            db.run("INSERT OR REPLACE INTO lesson_i18n(subject,topic,lang,payload_json,created_at) VALUES ('os',?,?,?,?)", (d["topic"], "te", jdump(te), now()))
        return te


_TE_LOCKS: dict[str, threading.Lock] = {}


# The exact rules the solver (oscalc.py) uses - written by hand, so the lesson can never contradict the worked example.
RULES = {
    "cpu": ["Completion time (CT) = the moment the process finishes. Turnaround time TAT = CT - AT. Waiting time WT = TAT - BT. "
            "Response time = first time on the CPU - AT.",
            "Ties: the process that arrived earlier goes first; if they arrived together, the lower process number (P1 before P2).",
            "Round Robin: a process that arrives exactly when a time slice ends joins the ready queue BEFORE the preempted process.",
            "If no process is ready, the CPU is idle - show the idle gap in the Gantt chart; averages are over all processes."],
    "pages": ["A page fault happens whenever the referenced page is not in a frame - the first loads into empty frames are faults too.",
              "FIFO replaces the page that was loaded earliest; LRU the page used least recently; Optimal the page whose next use is "
              "farthest in the future (a page never used again is replaced first).",
              "Belady's anomaly: with FIFO, giving more frames can sometimes cause MORE page faults. LRU and Optimal never show it."],
    "bankers": ["Need = Max - Allocation. A process can finish if its Need <= Work (Work starts as Available).",
                "When it finishes it releases its allocation: Work = Work + Allocation. Repeat, each time taking the lowest-numbered "
                "process that can finish.",
                "If every process can finish, the state is SAFE and the order is a safe sequence; otherwise it is unsafe."],
    "disk": ["Total head movement = sum of |next cylinder - current cylinder| along the path.",
             "SCAN travels to the last cylinder before reversing; LOOK reverses at the last request in that direction.",
             "C-SCAN and C-LOOK jump back to the other end and continue in the same direction - say in your answer whether you "
             "count the jump (this app counts it)."],
    "fit": ["First fit: the first hole big enough. Best fit: the smallest hole that fits. Worst fit: the largest hole.",
            "The unused part of a hole stays as a smaller hole for the next process; a process that fits nowhere must wait."],
    "paging": ["Page number = logical address div page size; offset = logical address mod page size.",
               "Physical address = frame number x page size + offset.",
               "Offset bits = log2(page size); page-number bits = log2(number of pages)."],
    "eat": ["TLB hit: t + m (TLB lookup, then the data). TLB miss: t + 2m (TLB, page table in memory, then the data).",
            "EAT = h x (t + m) + (1 - h) x (t + 2m), where h is the hit ratio."],
}
RULES_HEADING = {"en": "Exact rules for numericals", "te": "Numericals కి ఖచ్చితమైన నియమాలు"}


def _with_rules(body: dict, kind: str | None, lang: str) -> dict:
    """Numerical topics: drop any model-written 'conventions' section and put the solver's exact rules in its place."""
    if not kind or kind not in RULES:
        return body
    secs = [x for x in body["sections"] if not re.search(r"tie|convention|rule", x.get("heading", ""), re.IGNORECASE)]
    rules = "\n".join("- " + r for r in RULES[kind])
    return {**body, "sections": secs + [{"heading": RULES_HEADING.get(lang, RULES_HEADING["en"]), "body": rules, "exact": True}]}


def lesson(db: Database, user_id: str, topic_id: str, lang: str = "en", seed: int | None = None, brain: Any = None) -> dict[str, Any]:
    order = [(u, t) for u, t in P.topics_of(None)]
    pos = next((i for i, (_u, t) in enumerate(order) if t["id"] == topic_id), None)
    if pos is None:
        raise P.PaperError("Lesson not found.")
    u, t = order[pos]
    d = lessons().get(topic_id)
    if not d:
        raise P.PaperError("This lesson is not ready yet.")
    te = _telugu(db, d, brain) if lang == "te" else None
    body = _with_rules(te if te else d["en"], t.get("numerical"), "te" if te else "en")
    rng = random.Random(seed)
    example = None
    if t.get("numerical"):  # a fresh, exactly solved problem every time ("New example")
        g = oscalc.generate(t["numerical"], rng, 4, "medium", topic_id)
        example = {"blocks": g["blocks"], "answer": g["answer"]}
    bank = [q for q in P.bank("os") if q["topic"] == topic_id and q["qtype"] == "mcq"]
    rng.shuffle(bank)
    check = [{"q": q["q"], "options": q["options"], "correct": q["correct"], "why": q["answer"]} for q in bank[:3]]
    st = db.one("SELECT status FROM lesson_progress WHERE user_id=? AND subject='os' AND topic=?", (user_id, topic_id))
    if not st:
        db.run("INSERT OR IGNORE INTO lesson_progress(user_id,subject,topic,status,updated_at) VALUES (?,?,?,?,?)", (user_id, "os", topic_id, "opened", now()))
    nav = lambda i: {"id": order[i][1]["id"], "label": order[i][1]["label"]} if 0 <= i < len(order) else None  # noqa: E731
    return {"topic": topic_id, "unit": {"id": u["id"], "title": u["title"]}, "label": t["label"], "lang": "te" if te else "en",
            "has_telugu": bool(d.get("te")), "lesson": body, "example": example, "check": check, "status": (st["status"] if st else "opened"),
            "prev": nav(pos - 1), "next": nav(pos + 1)}


def mark(db: Database, user_id: str, topic_id: str, status: str) -> dict[str, Any]:
    if status not in ("learned", "opened"):
        raise P.PaperError("Unknown status.")
    db.run("INSERT INTO lesson_progress(user_id,subject,topic,status,updated_at) VALUES (?,?,?,?,?) "
           "ON CONFLICT(user_id,subject,topic) DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at", (user_id, "os", topic_id, status, now()))
    return {"ok": True, "status": status}


# ---------------------------------------------------------------- Telugu (shared with training/make_os_lessons.py)


def flatten(lesson: dict) -> dict[str, str]:
    f = {"title": lesson.get("title", ""), "summary": lesson["summary"], "diagram": lesson.get("diagram", "")}
    for i, s in enumerate(lesson["sections"]):
        f[f"s{i}h"], f[f"s{i}b"] = s.get("heading", ""), s.get("body", "")
    for i, k in enumerate(lesson.get("key_terms", [])):
        f[f"k{i}"] = k.get("meaning", "")
    ex = lesson.get("example") or {}
    f["exh"], f["exb"] = ex.get("title", ""), ex.get("body", "")
    for i, t in enumerate(lesson.get("exam_tips", [])):
        f[f"t{i}"] = t
    for i, t in enumerate(lesson.get("mistakes", [])):
        f[f"m{i}"] = t
    return {k: v for k, v in f.items() if v}


def unflatten(lesson: dict, f: dict[str, str]) -> dict:
    g = lambda k, d="": f.get(k, d)  # noqa: E731
    return {"title": g("title", lesson.get("title", "")), "summary": g("summary"), "diagram": g("diagram"),
            "sections": [{"heading": g(f"s{i}h", s.get("heading", "")), "body": g(f"s{i}b", s.get("body", ""))} for i, s in enumerate(lesson["sections"])],
            "key_terms": [{"term": k["term"], "meaning": g(f"k{i}", k.get("meaning", ""))} for i, k in enumerate(lesson.get("key_terms", []))],
            "example": {"title": g("exh"), "body": g("exb")},
            "exam_tips": [g(f"t{i}", t) for i, t in enumerate(lesson.get("exam_tips", []))],
            "mistakes": [g(f"m{i}", t) for i, t in enumerate(lesson.get("mistakes", []))]}


SYSTEM_TE = """Translate the JSON values into Telugu the way an Andhra/Telangana engineering lecturer explains in class:
Telugu sentences written in TELUGU SCRIPT (తెలుగు లిపి), with every technical term kept in ENGLISH LETTERS exactly as in the
source - e.g. process, CPU, scheduling, time quantum, preemptive, context switch, ready queue, page fault, semaphore, deadlock,
kernel, thread, Gantt chart, turnaround time. Never spell an English technical term in Telugu letters (write "scheduling",
not "షెడ్యూలింగ్"), and never write Telugu in English letters. Keep **bold**, "- " bullets, numbers and symbols.
Return ONLY JSON with exactly the same keys."""


def _te_ok(src: str, out: str) -> bool:
    te = len(re.findall(r"[\u0C00-\u0C7F]", out))
    letters = len(re.findall(r"[A-Za-z\u0C00-\u0C7F]", out)) or 1
    return te / letters >= 0.2 or len(src) < 25  # technical Telugu is English-term heavy; short labels may stay English


SYSTEM_TE_TEXT = SYSTEM_TE.replace("Translate the JSON values", "Translate the text").replace(
    "Return ONLY JSON with exactly the same keys.", "Output ONLY the Telugu translation, nothing else.")


def _one(brain: Any, src: str, heading: bool) -> str | None:
    for _ in range(3):
        r = brain._run(SYSTEM_TE_TEXT, src, False, 2500)
        text = (r[0] if r else "").strip().strip('"')
        if text and (heading or _te_ok(src, text)):
            return text
    return src if heading else None


def translate(brain: Any, lesson: dict, lang: str) -> dict | None:
    """Whole lesson in one plain-text call with ### markers (fast), then field-by-field retries for anything that came back wrong."""
    f = flatten(lesson)
    body = "\n\n".join(f"### {k}\n{v}" for k, v in f.items())
    sys_all = SYSTEM_TE_TEXT + "\nThe text has sections that start with lines like '### s0b'. Keep every '### key' line exactly as it is and translate only the text under it."
    r = brain._run(sys_all, body, False, 12000)
    got: dict[str, str] = {}
    if r:
        for part in re.split(r"(?m)^###\s*", r[0]):
            key, _, text = part.partition("\n")
            if key.strip() in f and text.strip():
                got[key.strip()] = text.strip()
    out: dict[str, str] = {}
    for k, src in f.items():
        heading = k in ("title", "exh") or (k[0] == "s" and k.endswith("h"))
        if k in got and (heading or _te_ok(src, got[k])):
            out[k] = got[k]
        else:
            one = _one(brain, src, heading)
            if one is None:
                return None
            out[k] = one
    return unflatten(lesson, out)
