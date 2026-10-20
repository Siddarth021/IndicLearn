"""Read a past question paper (PDF / Word / text / phone photo) and turn it into a paper blueprint.

Works without AI: marks markers like "[3 Marks]", "(4M)", "Max. Marks: 25" and "Duration: 90 Mins" give the rows; tables of
arrival/burst times, reference strings or cylinder requests mark numericals. With an AI engine, the question types are
read properly from the text, and kept only when their marks add up to what the paper says.
"""

from __future__ import annotations

import io
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from indiclearn.studio.courses import papers as P

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
DOC_EXT = {".pdf", ".docx", ".txt", ".md", ".pptx"}
_MARK = re.compile(r"[\[(]\s*(\d{1,2})\s*(?:marks?|m)\s*[\])]", re.IGNORECASE)
_TOTAL = re.compile(r"(?:max(?:imum)?|total)?\W{0,3}\w?\w?arks?\s*[:\-=]\s*(\d{1,3})", re.IGNORECASE)
_MIN = re.compile(r"(\d{2,3})\s*min", re.IGNORECASE)
_HRS = re.compile(r"(\d(?:\.\d)?)\s*(?:hours?|hrs?)\b", re.IGNORECASE)
_NUM_CUE = re.compile(r"arrival\s*time|burst\s*time|reference\s*string|page\s*frames|cylinders?|head\s*(?:is|at|position)|"
                      r"\ballocation\b.*\bmax\b|available\s*=|logical\s*address|hit\s*ratio|\bfirst\s*fit\b", re.IGNORECASE)

SYSTEM_PATTERN = """You read an exam question paper (the text may be OCR with errors and out of order). List every question in
order with its marks and its type: "fill" (fill in the blank), "mcq" (choose an option), "short" (definition / short
explanation), "long" (explain / describe / compare / with diagram), "numerical" (a problem with given data to compute).
Return ONLY JSON {"title": "<exam name>", "total": <max marks>, "minutes": <duration>, "questions": [{"marks": 3, "qtype": "short"}]}"""


def text_of(data: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in IMAGE_EXT:
        from PIL import Image

        from indiclearn.studio import ocr

        img = Image.open(io.BytesIO(data))
        img.load()
        lines = ocr.ocr_image(img.convert("RGB"))
        if not lines:
            raise P.PaperError("Could not read any text in that photo - try a sharper, straight photo in good light.")
        lines = sorted(lines, key=lambda ln: (round(ln.top / 12), ln.left))
        return "\n".join(ln.text for ln in lines)
    if ext in DOC_EXT:
        from indiclearn.studio.parse import ParseError, parse_file

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / ("paper" + ext)
            p.write_bytes(data)
            try:
                return parse_file(p, filename).full_text
            except ParseError as e:
                raise P.PaperError(str(e)) from e
    raise P.PaperError("Upload the paper as a PDF, Word file or a photo (JPG/PNG).")


def _heuristic(text: str) -> dict[str, Any]:
    marks = [int(m) for m in _MARK.findall(text) if 0 < int(m) <= 20]
    tot = next((int(m) for m in _TOTAL.findall(text) if 5 <= int(m) <= 200), None)
    minutes = next((int(m) for m in _MIN.findall(text) if 10 <= int(m) <= 240), None)
    if minutes is None:
        h = next((float(x) for x in _HRS.findall(text) if 0.5 <= float(x) <= 4), None)
        minutes = int(h * 60) if h else None
    if not marks:
        raise P.PaperError("Couldn't find the marks for each question (like \"[3 Marks]\"). You can still type the pattern yourself.")
    if tot and sum(marks) != tot:
        # OCR may drop or duplicate a marker: trust the per-question marks only when they are close to the stated total
        if abs(sum(marks) - tot) > max(2, tot * 0.15):
            tot = sum(marks)
    tot = tot or sum(marks)
    # numericals: one per data table (arrival/burst headers, reference strings...), given to the biggest questions first
    tables = len(re.findall(r"arrival\s*time\W{0,12}burst\s*time", text, re.IGNORECASE))  # one header per scheduling table
    others = len(re.findall(r"reference\s*string|cylinders?\b.*\bhead|safe\s*(?:state|sequence)|logical\s*address|hit\s*ratio", text, re.IGNORECASE))
    n_num = min(tables + others or (1 if _NUM_CUE.search(text) else 0), sum(1 for m in marks if m >= 3))
    types = ["short" if m <= 3 else "long" for m in marks]
    for i in sorted(range(len(marks)), key=lambda i: (-marks[i], -i))[:n_num]:
        types[i] = "numerical"
    for i, m in enumerate(marks):
        if m == 1:
            types[i] = "mcq"
    return {"total": tot, "minutes": minutes, "questions": [{"marks": m, "qtype": t} for m, t in zip(marks, types)]}


def _rows(questions: list[dict]) -> list[dict]:
    c = Counter((q["marks"], q["qtype"]) for q in questions)
    order = {"fill": 0, "mcq": 1, "short": 2, "long": 3, "numerical": 4, "msq": 5, "nat": 6}
    return [{"count": n, "marks": m, "qtype": t} for (m, t), n in sorted(c.items(), key=lambda x: (order.get(x[0][1], 9), x[0][0]))]


def detect_subject(text: str) -> str | None:
    """Which course a past paper belongs to (the one whose syllabus topics it mentions most)."""
    best = max(((len(P.coverage(text, [], s, min_hits=1)["covered"]), s) for s in P.SUBJECTS), default=(0, None))
    return best[1] if best[0] >= 2 else None


def blueprint_from(data: bytes, filename: str, brain: Any = None, subject: str | None = None) -> dict[str, Any]:
    text = text_of(data, filename)
    found = _heuristic(text)
    engine = "rules"
    if brain is not None and getattr(brain, "available", False):
        def ok(v: Any) -> Any:
            qs = v.get("questions") if isinstance(v, dict) else None
            if not isinstance(qs, list) or not qs:
                return None
            clean = []
            for q in qs:
                try:
                    m, t = int(q.get("marks")), str(q.get("qtype"))
                except (TypeError, ValueError, AttributeError):
                    return None
                if not 0 < m <= 20 or t not in ("fill", "mcq", "short", "long", "numerical"):
                    return None
                if t == "long" and m <= 3:  # a 3-mark question is a short answer, whatever the model calls it
                    t = "short"
                elif t == "short" and m >= 6:
                    t = "long"
                clean.append({"marks": m, "qtype": t})
            return {**v, "questions": clean}

        r = brain._run(SYSTEM_PATTERN, text[:6000], True, 1500, ok)
        if r and sum(q["marks"] for q in r[0]["questions"]) == found["total"]:
            found = {**found, "questions": r[0]["questions"], "title": r[0].get("title")}
            engine = r[1]
    subject = subject if subject in P.SUBJECTS else (detect_subject(text) or "os")
    cov = P.coverage(text, [], subject, min_hits=1)  # a paper mentions each topic once or twice
    units = sorted({u["id"] for u, t in P.topics_of(None, subject) if t["id"] in set(cov["covered"])})
    name = str(found.get("title") or "My college pattern")[:60]
    bp = {"name": name, "subject": subject, "total": found["total"], "minutes": found["minutes"] or max(30, found["total"] * 3), "difficulty": "college",
          "units": units or [u["id"] for u in P.syllabus(subject)["units"]], "rows": _rows(found["questions"]), "style": "college"}
    P.validate(bp)  # never hand the student a blueprint the builder would reject
    return {"blueprint": bp, "engine": engine, "questions_found": len(found["questions"]), "units_from_paper": bool(units)}
