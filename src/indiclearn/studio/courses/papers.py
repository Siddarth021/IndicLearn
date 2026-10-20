"""Printable exam papers for a subject course (Operating Systems first).

The student describes the paper's skeleton (total marks, time, units, rows such as "5 x 1 mark fill-in-the-blank"),
and gets a question paper plus a separate answer key - both printable PDFs; answers are only sent when asked for.

Where questions come from:
- topics = the syllabus units the student ticked, intersected with the topics their uploaded notes cover
  (when they have uploads that touch the subject; otherwise the ticked syllabus alone);
- numericals = exact generators in oscalc.py (answers computed, never guessed);
- theory = the checked question bank (os_bank.jsonl), plus fresh questions written from the student's own notes
  when an AI engine is available (checked the same way before use).
"""

from __future__ import annotations

import json
import random
import re
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from indiclearn.studio.courses import dsmacalc, oscalc

HERE = Path(__file__).resolve().parent
QTYPES = {"auto": "Any type", "fill": "Fill in the blank", "mcq": "Multiple choice", "short": "Short answer",
          "long": "Long answer / explain", "numerical": "Numerical problem", "msq": "Multiple select (MSQ)", "nat": "Numerical answer (NAT)"}
DIFFICULTY = {"easy": "Easy", "college": "College level", "tough": "Tough"}

PRESETS = [
    {"id": "mid", "name": "Mid-semester", "total": 25, "minutes": 90,
     "rows": [{"count": 3, "marks": 3, "qtype": "short"}, {"count": 2, "marks": 4, "qtype": "long"}, {"count": 2, "marks": 4, "qtype": "numerical"}]},
    {"id": "end", "name": "End-semester", "total": 50, "minutes": 180,
     "rows": [{"count": 5, "marks": 1, "qtype": "fill"}, {"count": 5, "marks": 1, "qtype": "mcq"}, {"count": 5, "marks": 2, "qtype": "short"},
              {"count": 3, "marks": 5, "qtype": "long"}, {"count": 3, "marks": 5, "qtype": "numerical"}]},
    {"id": "unit", "name": "Unit test", "total": 15, "minutes": 40,
     "rows": [{"count": 5, "marks": 1, "qtype": "mcq"}, {"count": 2, "marks": 2, "qtype": "short"}, {"count": 1, "marks": 6, "qtype": "numerical"}]},
    {"id": "gate", "name": "GATE-style (OS)", "total": 15, "minutes": 30, "style": "gate",
     "rows": [{"count": 3, "marks": 1, "qtype": "mcq"}, {"count": 1, "marks": 1, "qtype": "msq"}, {"count": 1, "marks": 1, "qtype": "nat"},
              {"count": 2, "marks": 2, "qtype": "mcq"}, {"count": 1, "marks": 2, "qtype": "msq"}, {"count": 2, "marks": 2, "qtype": "nat"}]},
    {"id": "custom", "name": "Custom", "total": 20, "minutes": 60, "rows": [{"count": 4, "marks": 5, "qtype": "auto"}]},
]


class PaperError(ValueError):
    pass


# --------------------------------------------------------------------------- course data


SUBJECTS = ("os", "dsma")  # the order courses are listed in
CALC = {"os": oscalc, "dsma": dsmacalc}  # exact numerical generators per subject


def calc(subject: str) -> Any:
    return CALC[subject] if subject in CALC else CALC[syllabus(subject)["subject"]]


@lru_cache(maxsize=8)
def syllabus(subject: str = "os") -> dict[str, Any]:
    if subject not in SUBJECTS:
        raise PaperError("Course not found.")
    return json.loads((HERE / f"{subject}_syllabus.json").read_text(encoding="utf-8"))


def topics_of(units: list[str] | None, subject: str = "os") -> list[tuple[dict, dict]]:
    syl = syllabus(subject)
    return [(u, t) for u in syl["units"] if not units or u["id"] in units for t in u["topics"]]


@lru_cache(maxsize=4)
def _bank_cached(subject: str, mtime: float) -> tuple[dict, ...]:
    path = HERE / f"{subject}_bank.jsonl"
    return tuple(json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip())


def bank(subject: str = "os") -> list[dict]:
    """The checked question bank plus the institute's previous exam questions (with worked answers)."""
    from indiclearn.studio.courses import pyq

    path = HERE / f"{subject}_bank.jsonl"
    return (list(_bank_cached(subject, path.stat().st_mtime)) if path.exists() else []) + pyq.bank_items(subject)


def presets(subject: str) -> list[dict[str, Any]]:
    short = syllabus(subject).get("short") or ("OS" if subject == "os" else subject.upper())
    return [{**p, "name": f"GATE-style ({short})"} if p["id"] == "gate" else p for p in PRESETS]


def courses() -> list[dict[str, Any]]:
    out = []
    for subject in SUBJECTS:
        syl = syllabus(subject)
        out.append({"id": subject, "title": syl["title"], "short": syl.get("short", syl["title"]), "questions": len(bank(subject)),
                    "units": [{"id": u["id"], "title": u["title"], "topics": [{"id": t["id"], "label": t["label"], "numerical": bool(t.get("numerical"))} for t in u["topics"]]}
                              for u in syl["units"]],
                    "presets": presets(subject), "qtypes": QTYPES, "difficulty": DIFFICULTY})
    return out


# --------------------------------------------------------------------------- blueprint


def validate(bp: dict[str, Any]) -> dict[str, Any]:
    try:
        total = int(bp.get("total") or 0)
        minutes = int(bp.get("minutes") or 0)
        rows = [{"count": int(r.get("count") or 0), "marks": int(r.get("marks") or 0), "qtype": str(r.get("qtype") or "auto")} for r in bp.get("rows") or []]
    except (TypeError, ValueError, AttributeError):
        raise PaperError("Marks, counts and time must be whole numbers.") from None
    rows = [r for r in rows if r["count"] > 0]
    if not rows:
        raise PaperError("Add at least one row of questions.")
    for r in rows:
        if r["qtype"] not in QTYPES:
            raise PaperError(f"Unknown question type '{r['qtype']}'.")
        if not 1 <= r["marks"] <= 20 or r["count"] > 30:
            raise PaperError("Each question can carry 1-20 marks, with at most 30 questions in a row.")
        if r["qtype"] in ("fill", "mcq") and r["marks"] > 2:
            raise PaperError(f"{QTYPES[r['qtype']]} questions carry 1 or 2 marks.")
        if r["qtype"] == "numerical" and r["marks"] < 2:
            raise PaperError("A numerical problem needs at least 2 marks.")
        if r["qtype"] == "msq" and r["marks"] > 2:
            raise PaperError("MSQ questions carry 1 or 2 marks (as in GATE).")
        if r["qtype"] == "nat" and r["marks"] > 5:
            raise PaperError("NAT questions carry 1-5 marks.")
    got = sum(r["count"] * r["marks"] for r in rows)
    if total and got != total:
        raise PaperError(f"The rows add up to {got} marks, but the paper is set to {total}. Change a row or the total.")
    if sum(r["count"] for r in rows) > 60:
        raise PaperError("At most 60 questions in one paper.")
    units = [u for u in (bp.get("units") or []) if isinstance(u, str)]
    subject = bp.get("subject") or "os"
    known = {u["id"] for u in syllabus(subject)["units"]}
    units = [u for u in units if u in known] or sorted(known)
    style = "gate" if bp.get("style") == "gate" or any(r["qtype"] in ("msq", "nat") for r in rows) else "college"
    return {"subject": subject, "style": style, "name": str(bp.get("name") or "Practice paper")[:60], "total": got, "minutes": minutes or max(30, got * 3),
            "units": units, "difficulty": bp.get("difficulty") if bp.get("difficulty") in DIFFICULTY else "college", "rows": rows}


# --------------------------------------------------------------------------- what the student's notes cover


def _kw_hits(text: str, kw: str) -> int:
    return len(re.findall(r"(?<![a-z0-9])" + re.escape(kw.lower()) + r"(?![a-z0-9])", text))


def coverage(notes_text: str, units: list[str], subject: str = "os", min_hits: int = 2) -> dict[str, Any]:
    """Which ticked syllabus topics the uploaded notes actually teach (>= 2 keyword mentions)."""
    low = notes_text.lower()
    covered, missing = [], []
    for u, t in topics_of(units, subject):
        score = sum(_kw_hits(low, k) for k in t["keywords"])
        (covered if score >= min_hits else missing).append(t["id"])
    return {"covered": covered, "missing": missing}


# --------------------------------------------------------------------------- assembling a paper


def _slot_kind(qtype: str, marks: int, rng: random.Random, can_num: bool, style: str = "college") -> str:
    if qtype != "auto":
        return qtype
    if style == "gate":
        return rng.choice(["mcq", "mcq", "msq", "nat"] if can_num else ["mcq", "msq"])
    if marks == 1:
        return rng.choice(["fill", "mcq"])
    if marks <= 3:
        return "short"
    return "numerical" if can_num and rng.random() < 0.45 else "long"


def _bank_type(kind: str, marks: int) -> list[str]:
    if kind in ("fill", "mcq", "msq"):
        return [kind]
    if kind == "short":
        return ["short", "medium"] if marks >= 3 else ["short"]
    return ["long", "medium"] if marks >= 7 else ["medium", "long", "short"] if marks <= 4 else ["medium", "long"]


_ASKS_TO_COMPUTE = re.compile(r"\b(calculate|compute|find|determine|draw the gantt|how many)\b", re.I)


def numericish(q: dict) -> bool:
    """A bank question that is really a numerical (it gives data and asks for a computed result): its answer was worked
    out by a language model, not computed. Numericals come only from the exact generators, so these never go into a paper."""
    text = q.get("q", "")
    if q.get("source") == "pyq":
        return False  # previous exam questions carry answers computed and checked when they were added
    return q.get("qtype") in ("short", "medium", "long") and bool(_ASKS_TO_COMPUTE.search(text)) and len(re.findall(r"\b\d+\b", text)) >= 3


def _diff_weight(q: dict, difficulty: str) -> float:
    d = q.get("difficulty", "medium")
    table = {"easy": {"easy": 3, "medium": 2, "hard": 0.3}, "college": {"easy": 1, "medium": 3, "hard": 1.5}, "tough": {"easy": 0.3, "medium": 1.5, "hard": 3}}
    return table[difficulty].get(d, 1)


def _scale_scheme(scheme: list[dict], marks: int) -> list[dict]:
    """Re-split a mark scheme onto the marks this paper gives the question (whole marks, same proportions)."""
    if not scheme:
        return []
    tot = sum(max(0, s["marks"]) for s in scheme) or 1
    raw = [s["marks"] * marks / tot for s in scheme]
    out = [int(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - out[i], reverse=True)[: marks - sum(out)]:
        out[i] += 1
    return [{"point": s["point"], "marks": m} for s, m in zip(scheme, out) if m > 0]


def assemble(bp: dict[str, Any], topic_ids: list[str], fresh: list[dict], used: set[str], seed: int | None = None) -> dict[str, Any]:
    """Fill every row of the blueprint. Returns the paper with questions AND answers (the API strips answers)."""
    rng = random.Random(seed)
    subject = bp.get("subject") or "os"
    oc = calc(subject)
    syl_topics = {t["id"]: (u, t) for u, t in topics_of(None, subject)}
    tset = set(topic_ids)
    num_topics = [t for t in topic_ids if syl_topics[t][1].get("numerical") in oc.GENERATORS]
    pool = [q for q in bank(subject) + fresh if q["topic"] in tset and not numericish(q)]
    taken: set[str] = set()
    topic_count: dict[str, int] = {}
    questions, notes = [], []
    num_kinds_used: dict[str, int] = {}
    for row in bp["rows"]:
        for _ in range(row["count"]):
            kind = _slot_kind(row["qtype"], row["marks"], rng, bool(num_topics), bp.get("style", "college"))
            diff = {"easy": "easy", "college": "medium", "tough": "hard"}[bp["difficulty"]]
            val_topics = [t for t in num_topics if syl_topics[t][1]["numerical"] in oc.VALUE_GENERATORS]
            msq_topics = [t for t in num_topics if syl_topics[t][1]["numerical"] in oc.MSQ_KINDS]
            if kind == "nat" and not val_topics:
                kind = "mcq"
            if kind == "numerical" and not num_topics:
                kind = "long"
                if "no_numerical" not in notes:
                    notes.append("no_numerical")
            q = None
            pick_t = lambda ts: min(ts, key=lambda x: (num_kinds_used.get(syl_topics[x][1]["numerical"], 0), topic_count.get(x, 0), rng.random()))  # noqa: E731
            if kind == "nat":
                t = pick_t(val_topics)
                num_kinds_used[syl_topics[t][1]["numerical"]] = num_kinds_used.get(syl_topics[t][1]["numerical"], 0) + 1
                g = oc.gen_nat(syl_topics[t][1]["numerical"], rng, row["marks"], diff, t)
                q = {"id": "nat-" + uuid.uuid4().hex[:8], "kind": "nat", "topic": t, "blocks": g["blocks"], "answer_blocks": g["answer"],
                     "value": g["value"], "tolerance": g["tolerance"]}
            elif kind == "msq" and msq_topics and (not [c for c in pool if c["qtype"] == "msq" and c["id"] not in taken] or rng.random() < 0.5):
                t = pick_t(msq_topics)
                g = None
                for _ in range(8):
                    g = oc.gen_msq_numeric(syl_topics[t][1]["numerical"], rng, row["marks"], diff, t)
                    if g:
                        break
                if g:
                    num_kinds_used[syl_topics[t][1]["numerical"]] = num_kinds_used.get(syl_topics[t][1]["numerical"], 0) + 1
                    q = {"id": "msq-" + uuid.uuid4().hex[:8], "kind": "msq", "topic": t, "blocks": g["blocks"], "options": g["options"],
                         "correct": g["correct"], "answer_blocks": g["answer"]}
            elif kind == "mcq" and bp.get("style") == "gate" and row["marks"] >= 2 and val_topics and rng.random() < 0.6:
                t = pick_t(val_topics)
                num_kinds_used[syl_topics[t][1]["numerical"]] = num_kinds_used.get(syl_topics[t][1]["numerical"], 0) + 1
                g = oc.gen_mcq_numeric(syl_topics[t][1]["numerical"], rng, row["marks"], diff, t)
                q = {"id": "mcqn-" + uuid.uuid4().hex[:8], "kind": "mcq", "topic": t, "blocks": g["blocks"], "options": g["options"],
                     "correct": g["correct"], "answer_blocks": g["answer"]}
            if q is not None:
                pass
            elif kind == "numerical":
                # spread numericals over different problem kinds (scheduling, paging, deadlock, disk...)
                t = pick_t(num_topics)
                nk = syl_topics[t][1]["numerical"]
                num_kinds_used[nk] = num_kinds_used.get(nk, 0) + 1
                g = oc.generate(nk, rng, row["marks"], diff, t)
                q = {"id": "num-" + uuid.uuid4().hex[:8], "kind": "numerical", "topic": t, "blocks": g["blocks"], "answer_blocks": g["answer"]}
            else:
                types = _bank_type(kind, row["marks"])
                cands = [c for c in pool if c["qtype"] in types and c["id"] not in taken]
                if not cands:
                    cands = [c for c in bank(subject) if c["qtype"] in types and c["id"] not in taken and not numericish(c)]
                if cands:
                    def w(c: dict) -> float:
                        fit = 1.0 if c["qtype"] == types[0] else 0.4
                        spread = 1 / (1 + 2 * topic_count.get(c["topic"], 0))
                        fresh_bonus = 2.0 if c.get("source") == "notes" else 1.7 if c.get("source") == "pyq" else 1.0  # own notes, then real past papers
                        seen = 0.15 if c["id"] in used else 1.0
                        return fit * spread * fresh_bonus * seen * _diff_weight(c, bp["difficulty"]) * (0.5 + rng.random())
                    c = max(cands, key=w)
                    taken.add(c["id"])
                    q = {"id": c["id"], "kind": c["qtype"], "topic": c["topic"], "blocks": [{"p": c["q"]}],
                         "options": c.get("options"), "correct": c.get("correct"), "answer_text": c["answer"],
                         "scheme": _scale_scheme(c.get("scheme") or [], row["marks"]), "diagram": c.get("diagram") or "",
                         "source": c.get("source", "bank")}
                    if c.get("source") == "pyq":
                        q.update(answer_blocks=c["answer_blocks"], paper_title=c["paper_title"], answer_text="")
            if q is None:
                notes.append("short_of_questions")
                continue
            topic_count[q["topic"]] = topic_count.get(q["topic"], 0) + 1
            q["marks"] = row["marks"]
            if bp.get("style") == "gate":  # GATE marking: wrong MCQ costs 1/3 of its marks; MSQ and NAT have no negative marks
                q["negative"] = round(row["marks"] / 3, 2) if q["kind"] == "mcq" else 0
            q["topic_label"] = syl_topics[q["topic"]][1]["label"]
            q["unit"] = syl_topics[q["topic"]][0]["id"]
            questions.append(q)
    for i, q in enumerate(questions, 1):
        q["n"] = i
    return {"questions": questions, "notes": sorted(set(notes)), "total": sum(q["marks"] for q in questions)}


def public(paper: dict[str, Any], answers: bool) -> dict[str, Any]:
    """Questions only, unless the student asked for the answers."""
    keep = ("n", "id", "kind", "marks", "negative", "topic", "topic_label", "unit", "blocks", "options", "source", "paper_title")
    secret = ("correct", "answer_text", "answer_blocks", "scheme", "diagram", "value", "tolerance")
    qs = [{k: q.get(k) for k in keep} | ({k: q.get(k) for k in secret} if answers else {})
          for q in paper["questions"]]
    return {**{k: v for k, v in paper.items() if k != "questions"}, "questions": qs, "answers_included": answers}
