"""Previous question papers of the institute (IIIT Sri City), with worked answers.

<subject>_pyq.json is built by training/build_dsma_pyq.py. The questions are used everywhere:
- "Previous papers" in the course (read online, show answers, print the paper or its answer key);
- the lesson page lists the questions of that topic that were asked in exams;
- exam papers and quizzes draw on them like on the checked bank (and prefer them: they are what the institute asks);
- the course chat can find them (they are part of the course document).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=8)
def _load(subject: str, mtime: float) -> dict[str, Any]:
    return json.loads((HERE / f"{subject}_pyq.json").read_text(encoding="utf-8"))


def data(subject: str) -> dict[str, Any]:
    path = HERE / f"{subject}_pyq.json"
    return _load(subject, path.stat().st_mtime) if path.exists() else {"papers": []}


def papers(subject: str) -> list[dict[str, Any]]:
    return data(subject)["papers"]


def _public_q(q: dict[str, Any], answers: bool) -> dict[str, Any]:
    keep = {k: q[k] for k in ("n", "q", "marks", "topic", "qtype", "options") if k in q}
    if answers:
        keep.update({k: q[k] for k in ("answer_blocks", "correct", "key_note", "paper_issue") if k in q})
    return keep


def listing(subject: str) -> list[dict[str, Any]]:
    return [{k: p.get(k) for k in ("id", "title", "exam", "year", "date", "minutes", "marks", "note")} | {"questions": len(p["questions"])} for p in papers(subject)]


def paper(subject: str, pid: str, answers: bool = False) -> dict[str, Any]:
    from indiclearn.studio.courses.papers import PaperError, syllabus

    p = next((x for x in papers(subject) if x["id"] == pid), None)
    if p is None:
        raise PaperError("Paper not found.")
    labels = {t["id"]: t["label"] for u in syllabus(subject)["units"] for t in u["topics"]}
    qs = [{**_public_q(q, answers), "topic_label": labels.get(q["topic"], q["topic"])} for q in p["questions"]]
    return {**{k: v for k, v in p.items() if k != "questions"}, "questions": qs, "answers_included": answers}


def by_topic(subject: str, topic: str) -> list[dict[str, Any]]:
    """Exam questions asked on one topic (newest paper first) - shown on the lesson page."""
    out = []
    for p in sorted(papers(subject), key=lambda x: -int(x.get("year") or 0)):
        for q in p["questions"]:
            if q["topic"] == topic:
                out.append({"paper": p["id"], "paper_title": p["title"], **_public_q(q, True)})
    return out


def _text(blocks: list[dict]) -> str:
    out = []
    for b in blocks or []:
        if "p" in b:
            out.append(b["p"])
        elif "math" in b:
            out.append(f"$${b['math']}$$")
    return "\n".join(out)


def bank_items(subject: str) -> list[dict[str, Any]]:
    """The questions in the format of the checked bank (papers.bank merges them in)."""
    from indiclearn.studio.courses.papers import topics_of

    unit_of = {t["id"]: u["id"] for u, t in topics_of(None, subject)}
    out = []
    for p in papers(subject):
        for q in p["questions"]:
            if q["qtype"] in ("mcq", "msq") and q.get("correct") is None:
                continue  # a printed question with no correct option is shown in Previous papers only
            if q["topic"] not in unit_of:
                continue
            out.append({"id": f"pyq-{p['id']}-{q['n']}", "subject": subject, "unit": unit_of[q["topic"]], "topic": q["topic"], "qtype": q["qtype"],
                        "marks": q["marks"], "difficulty": "medium", "q": q["q"], "options": q.get("options"), "correct": q.get("correct"),
                        "answer": _text(q["answer_blocks"]), "answer_blocks": q["answer_blocks"], "source": "pyq", "paper_title": p["title"]})
    return out


def chunks(subject: str) -> list[tuple[str, str]]:
    """(topic, text) for the course document, so the chat can find and explain previous exam questions."""
    out = []
    for p in papers(subject):
        for q in p["questions"]:
            opts = (" Options: " + " | ".join(q["options"])) if q.get("options") else ""
            out.append((q["topic"], f"IIIT Sri City {p['title']}, question {q['n']} [{q['marks']} marks]: {q['q']}{opts}\nWorked answer: {_text(q['answer_blocks'])}"))
    return out


def _split_matrices(text: str) -> list[dict]:
    """Question text with $...matrix...$ inside -> text and display blocks (the PDF then draws real matrices)."""
    import re

    out: list[dict] = []
    pos = 0
    for m in re.finditer(r"\$([^$]*\\begin\{[bpv]?matrix\}[^$]*)\$", text):
        if text[pos:m.start()].strip():
            out.append({"p": text[pos:m.start()].strip()})
        out.append({"math": m.group(1)})
        pos = m.end()
    if text[pos:].strip(" ."):
        out.append({"p": text[pos:].strip().lstrip(".,; ").capitalize() if pos else text[pos:].strip()})
    return out or [{"p": text}]


def as_exam_paper(subject: str, pid: str) -> dict[str, Any]:
    """The previous paper in the shape pdf.render() prints."""
    from indiclearn.studio.courses.papers import syllabus

    p = next(x for x in papers(subject) if x["id"] == pid)
    kind = {"mcq": "mcq", "msq": "msq"}
    qs = [{"n": q["n"], "id": f"{pid}-{q['n']}", "kind": kind.get(q["qtype"], q["qtype"]), "marks": q["marks"], "blocks": _split_matrices(q["q"]),
           "options": q.get("options"), "correct": q.get("correct"), "answer_blocks": q["answer_blocks"]} for q in p["questions"]]
    return {"course_title": f"{syllabus(subject)['title']} - IIIT Sri City", "blueprint": {"name": p["title"], "minutes": p.get("minutes") or 90, "style": "college"},
            "total": p["marks"], "questions": qs}
