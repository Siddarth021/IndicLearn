"""Printable PDFs for exam papers: the question paper and (only when asked) a separate answer key."""

from __future__ import annotations

import io
import os
import re
import time
from html import escape
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import HRFlowable, Indenter, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from indiclearn.studio.courses import texplain

_HERE = os.path.dirname(os.path.abspath(__file__))

_FONT: tuple[str, str] | None = None
_CANDIDATES = [  # a Unicode serif (logic and maths symbols print correctly): the bundled DejaVu Serif, else the machine's
    ("PaperSerif", os.path.join(_HERE, "fonts", "DejaVuSerif.ttf"), os.path.join(_HERE, "fonts", "DejaVuSerif-Bold.ttf")),
    ("PaperSerif", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
    ("PaperSerif", "C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/timesbd.ttf"),
    ("PaperSerif", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"),
    ("PaperSerif", "/System/Library/Fonts/Supplemental/Times New Roman.ttf", "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"),
]
_ASCII = str.maketrans({"→": "->", "←": "<-", "≤": "<=", "≥": ">=", "×": "x", "–": "-", "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"', "…": "...", "≠": "!=", "λ": "lambda"})


def _fonts() -> tuple[str, str]:
    global _FONT
    if _FONT is None:
        _FONT = ("Times-Roman", "Times-Bold")
        for name, reg, bold in _CANDIDATES:
            if os.path.exists(reg) and os.path.exists(bold):
                try:
                    pdfmetrics.registerFont(TTFont(name, reg))
                    pdfmetrics.registerFont(TTFont(name + "-Bold", bold))
                    pdfmetrics.registerFontFamily(name, normal=name, bold=name + "-Bold")
                    _FONT = (name, name + "-Bold")
                    break
                except Exception:  # pragma: no cover - a broken font file: keep the built-in one
                    continue
    return _FONT


def _txt(s: Any) -> str:
    s = texplain.text(str(s or ""))  # $\\frac{1}{2}$ -> 1/2, $p \\rightarrow q$ -> p → q
    if _fonts()[0].startswith("Times"):
        s = s.translate(_ASCII).encode("latin-1", "replace").decode("latin-1")
    s = escape(s, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    return s.replace("$", "")


def _styles() -> dict[str, ParagraphStyle]:
    reg, bold = _fonts()
    return {
        "title": ParagraphStyle("t", fontName=bold, fontSize=14, leading=18, alignment=TA_CENTER),
        "sub": ParagraphStyle("s", fontName=reg, fontSize=11, leading=14, alignment=TA_CENTER),
        "body": ParagraphStyle("b", fontName=reg, fontSize=10.5, leading=14),
        "small": ParagraphStyle("sm", fontName=reg, fontSize=9.5, leading=12),
        "bold": ParagraphStyle("bo", fontName=bold, fontSize=10.5, leading=14),
        "head": ParagraphStyle("h", fontName=bold, fontSize=10.5, leading=14, alignment=TA_CENTER),
        "bullet": ParagraphStyle("bu", fontName=reg, fontSize=10.5, leading=14, leftIndent=12, bulletIndent=2),
        "mcell": ParagraphStyle("mc", fontName=reg, fontSize=9.5, leading=11, alignment=TA_CENTER),
    }


def _table(rows: list[list[str]], st: dict[str, ParagraphStyle], width: float) -> Table:
    cols = max(len(r) for r in rows)
    data = [[Paragraph(_txt(c), st["small"]) for c in r] + [""] * (cols - len(r)) for r in rows]
    t = Table(data, colWidths=[width / cols] * cols, hAlign="LEFT")
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEEEEE")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return t


def _blocks(blocks: list[dict], st: dict[str, ParagraphStyle], width: float) -> list[Any]:
    out: list[Any] = []
    for b in blocks:
        if "table" in b:
            out += [Spacer(1, 3), _table(b["table"], st, width), Spacer(1, 3)]
        elif "math" in b:
            out += _math(b["math"], st, width)
        else:
            out += _rich(b.get("p", ""), st)
    return out


def _matrix(rows: list[list[str]], st: dict[str, ParagraphStyle]) -> Table:
    """A matrix drawn with bracket lines on both sides."""
    ncol = max(len(r) for r in rows)
    w = max(6.5 * mm, max(len(c) for r in rows for c in r) * 1.9 * mm + 2.5 * mm)
    data = [[Paragraph(escape(c), st["mcell"]) for c in r] + [""] * (ncol - len(r)) for r in rows]
    t = Table(data, colWidths=[w] * ncol, hAlign="LEFT")
    t.setStyle(TableStyle([("LINEBEFORE", (0, 0), (0, -1), 0.8, colors.black), ("LINEAFTER", (-1, 0), (-1, -1), 0.8, colors.black),
                           ("LINEABOVE", (0, 0), (0, 0), 0.8, colors.black), ("LINEBELOW", (0, -1), (0, -1), 0.8, colors.black),
                           ("LINEABOVE", (-1, 0), (-1, 0), 0.8, colors.black), ("LINEBELOW", (-1, -1), (-1, -1), 0.8, colors.black),
                           ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1), ("LEFTPADDING", (0, 0), (-1, -1), 2),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 2), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    t._w = w * ncol  # noqa: SLF001 - remembered for line wrapping below
    return t


def _lines(latex: str) -> list[str]:
    """Split a display formula at its top-level '\\\\' line breaks (never inside a matrix)."""
    out, depth, start, i = [], 0, 0, 0
    while i < len(latex):
        if latex.startswith("\\begin{", i):
            depth += 1
        elif latex.startswith("\\end{", i):
            depth -= 1
        elif latex.startswith("\\\\", i) and depth == 0:
            out.append(latex[start:i])
            start = i + 2
            i += 2
            continue
        i += 1
    out.append(latex[start:])
    return out


def _math(latex: str, st: dict[str, ParagraphStyle], width: float) -> list[Any]:
    """A display formula: one line per '\\\\'-separated step; matrices drawn as grids, wrapped onto new rows when wide."""
    out: list[Any] = []
    for line in _lines(latex):
        if not line.strip():
            continue
        parts = texplain.split_matrices(line)
        if not any(k == "matrix" for k, _v in parts):
            out.append(Paragraph(_txt("$" + line + "$"), st["body"]))
            continue
        rows, cur, used = [], [], 0.0
        for k, v in parts:
            if k == "matrix":
                cell = _matrix(v, st)  # type: ignore[arg-type]
                w = cell._w + 4 * mm  # noqa: SLF001
            else:
                txt_ = str(v).strip().strip(",").strip()
                if not txt_:
                    continue
                w = max(10 * mm, len(txt_) * 2.3 * mm + 4 * mm)
                cell = Paragraph(escape(txt_), st["body"])
            if cur and used + w > width:
                rows.append(cur)
                cur, used = [], 0.0
            cur.append((cell, w))
            used += w
        if cur:
            rows.append(cur)
        for r in rows:
            t = Table([[c for c, _w in r]], colWidths=[w for _c, w in r], hAlign="LEFT")
            t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 1), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
            out += [t, Spacer(1, 2)]
    return out


def _rich(text: str, st: dict[str, ParagraphStyle]) -> list[Any]:
    """Model-answer text: paragraphs, '- ' bullets and '1.' lists, **bold**."""
    out: list[Any] = []
    for line in str(text).split("\n"):
        line = line.rstrip()
        if not line.strip():
            continue
        m = re.match(r"^\s*(?:[-*\u2022]|\d+[.)])\s+(.*)", line)
        if m:
            out.append(Paragraph(_txt(m.group(1)), st["bullet"], bulletText="\u2022" if not _fonts()[0].startswith("Times") else "-"))
        else:
            out.append(Paragraph(_txt(line), st["body"]))
    return out


def _header(paper: dict[str, Any], st: dict[str, ParagraphStyle], key: bool) -> list[Any]:
    bp = paper["blueprint"]
    head = [Paragraph(_txt(f"{paper['course_title']} - {bp['name']}" + (" - ANSWER KEY" if key else "")), st["title"]), Spacer(1, 4),
            Paragraph(_txt(f"Max. Marks: {paper['total']}") + "&nbsp;" * 8 + _txt(f"Duration: {bp['minutes']} minutes"), st["sub"]),
            Spacer(1, 8)]
    if not key and bp.get("style") == "gate":
        head += [Paragraph("<b>Instructions (GATE pattern):</b>", st["body"]),
                 Paragraph(_txt("MCQ - exactly one correct option; a wrong answer costs 1/3 of the question's marks. "
                                "MSQ - one or more options are correct; no negative marks and no partial marks. "
                                "NAT - write the numerical value; no negative marks. A virtual calculator is allowed."), st["small"]), Spacer(1, 6)]
    elif not key:
        head += [Paragraph("<b>Instructions:</b>", st["body"]),
                 Paragraph(_txt("1. Answer all the questions. 2. Marks for each question are shown on the right. "
                                "3. Draw neat diagrams wherever necessary. 4. Calculators are allowed."), st["small"]), Spacer(1, 6)]
    return head


def render(paper: dict[str, Any], answers: bool = False) -> bytes:
    st = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
                            title=f"{paper['course_title']} - {paper['blueprint']['name']}" + (" (answer key)" if answers else ""),
                            author="IndicLearn", creator="IndicLearn")
    width = A4[0] - 32 * mm
    qw = width - 10 * mm - 16 * mm
    story: list[Any] = _header(paper, st, answers)
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.black, spaceAfter=6))
    for q in paper["questions"]:
        first, rest = q["blocks"][0], q["blocks"][1:]
        lead = _blocks([first], st, qw)
        tag = {"mcq": "MCQ", "msq": "MSQ", "nat": "NAT"}.get(q["kind"], "") if paper["blueprint"].get("style") == "gate" else ""
        head = Table([[Paragraph(f"<b>{q['n']}.</b>", st["body"]), lead[0] if len(lead) == 1 else lead,
                       Paragraph(f"<b>[{q['marks']}]</b>" + (f"<br/><font size=8>{tag}</font>" if tag else ""), st["body"])]],
                     colWidths=[10 * mm, qw, 16 * mm])
        head.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("ALIGN", (2, 0), (2, 0), "RIGHT")]))
        body: list[Any] = _blocks(rest, st, qw)
        if q.get("options"):
            body += [Paragraph(_txt(f"({'abcde'[i]}) {o}"), st["bullet"]) for i, o in enumerate(q["options"])]
        if q["kind"] == "nat" and not answers:
            body.append(Paragraph(_txt("Answer: ______________"), st["body"]))
        if answers:
            body.append(Spacer(1, 4))
            body += _answer(q, st, qw)
        keep = 1 if answers else 3  # keep a question's start with its first lines; long answers may flow on
        story.append(KeepTogether([head, Indenter(left=10 * mm), *body[:keep], Indenter(left=-10 * mm)]))
        if body[keep:]:
            story += [Indenter(left=10 * mm), *body[keep:], Indenter(left=-10 * mm)]
        story.append(HRFlowable(width="100%", thickness=0.3, color=colors.grey, spaceBefore=6, spaceAfter=6))
    story.append(Spacer(1, 10))
    story.append(Paragraph(_txt(f"Generated {time.strftime('%d %b %Y')}. Numerical answers are computed exactly; theory answers were checked by a second AI model."
                                if answers else "*** All the best ***"), st["small"]))
    doc.build(story)
    return buf.getvalue()


def _answer(q: dict[str, Any], st: dict[str, ParagraphStyle], width: float) -> list[Any]:
    out: list[Any] = [Paragraph("<b>Answer:</b>", st["body"])]
    if q["kind"] == "msq" and q.get("correct") is not None:
        out.append(Paragraph(_txt("Correct options: " + ", ".join(f"({'abcde'[i]})" for i in q["correct"])), st["bold"]))
    if q.get("answer_blocks") and q["kind"] == "mcq" and q.get("correct") is not None:
        out.append(Paragraph(_txt(f"({'abcde'[q['correct']]}) {q['options'][q['correct']]}"), st["bold"]))
    if q.get("answer_blocks"):
        out += _blocks(q["answer_blocks"], st, width)
        return out
    if q["kind"] == "numerical":
        out += _blocks(q.get("answer_blocks") or [], st, width)
        return out
    if q["kind"] == "mcq" and q.get("correct") is not None:
        out.append(Paragraph(_txt(f"({'abcde'[q['correct']]}) {q['options'][q['correct']]}"), st["bold"]))
    out += _rich(q.get("answer_text", ""), st)
    if q.get("diagram"):
        out.append(Paragraph(_txt(f"Diagram expected: {q['diagram']}"), st["small"]))
    if q.get("scheme"):
        out.append(Paragraph("<b>Marking scheme:</b>", st["small"]))
        out += [Paragraph(_txt(f"{s['point']} - {s['marks']} mark{'s' if s['marks'] != 1 else ''}"), st["bullet"]) for s in q["scheme"]]
    return out
