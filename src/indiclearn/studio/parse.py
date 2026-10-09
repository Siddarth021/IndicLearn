"""Structure-preserving document parsing (PDF, PPTX, DOCX, TXT, MD).

Unlike a plain text extractor this keeps *structure*: headings (with level), paragraphs, bullets
and equation-like lines, each tagged with its page / slide.  The headings are what lets the
default knowledge map follow the real topic hierarchy of the material instead of guessing.

PDF heading detection uses font sizes (via pypdf's text visitor) relative to the body-text size,
numbering patterns (``2.1 Method``, ``II. RELATED WORK``, ``A. Dataset``), and bold short lines.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from indiclearn.studio.lang import detect_document_language, normalize

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".docx", ".txt", ".md", ".markdown"}
UNSUPPORTED_HINT = {".ppt": "Old .ppt files are not supported - save as .pptx.", ".doc": "Old .doc files are not supported - save as .docx."}
VIRTUAL_PAGE_CHARS = 2400
MAX_PAGES = 600


@dataclass
class Block:
    kind: str  # title | heading | para | bullet | math | ref
    text: str
    page: int
    level: int = 0  # heading level (1 = chapter)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDoc:
    title: str
    unit_label: str  # Page | Slide | Part
    blocks: list[Block]
    n_pages: int
    language: str = "en"
    script: str = "Latin"
    warnings: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n".join(b.text for b in self.blocks)

    def pages(self) -> dict[int, list[Block]]:
        out: dict[int, list[Block]] = {}
        for b in self.blocks:
            out.setdefault(b.page, []).append(b)
        return out


class ParseError(Exception):
    """Raised with a student-friendly message when a file cannot be read."""


# ---------------------------------------------------------------------------
# Heading heuristics shared by text-ish formats
# ---------------------------------------------------------------------------
_NUMBERED = re.compile(r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,3})[.)]?\s+(?P<t>\S.{1,110})$")
_CHAPTER = re.compile(r"^(?:chapter|unit|module|lesson|topic|section|అధ్యాయం|యూనిట్|పాఠం|अध्याय|इकाई|पाठ|अध्यायः)\s*[\d IVXivx]+[:.\-–]?\s*(?P<t>.*)$", re.IGNORECASE)
_ROMAN = re.compile(r"^(?P<num>[IVX]{1,5})\.\s+(?P<t>[A-Z][A-Za-z0-9 \-:&,/]{2,90})$")
_LETTERED = re.compile(r"^(?P<num>[A-Z])\.\s+(?P<t>[A-Z][^.]{2,80})$")
_KEYWORD_HEADINGS = {
    "abstract", "introduction", "background", "related work", "conclusion", "conclusions", "references",
    "acknowledgment", "acknowledgments", "acknowledgements", "methodology", "methods", "results", "discussion",
    "experiments", "summary", "overview", "preface", "contents", "bibliography", "appendix",
}
_REF_HEADINGS = {"references", "bibliography", "works cited"}
_BULLET = re.compile(r"^\s*(?:[•●▪■◦○\-–—*]|\(?\d{1,2}[.)]|\(?[a-z][.)])\s+(?P<t>\S.*)$")
_PAGE_NUM = re.compile(r"^\s*(?:page\s*)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?\s*$", re.IGNORECASE)
_MATHY = re.compile(r"[=∑∫∏√≤≥≈∞±×÷∂∇αβγδεθλμπσφωΩ∈∀∃→⇒]|\\frac|\\sum")


_CAPTION = re.compile(r"^(?:table|fig\.?|figure|chart|algorithm|listing)\s+[IVXivx\d]+[.:)]?(?:\s|$)", re.IGNORECASE)
_TITLE_SMALL = {"and", "or", "of", "the", "a", "an", "in", "on", "for", "to", "with", "vs", "by", "at", "from"}


def smart_title(text: str) -> str:
    """Title-case an ALL-CAPS heading while keeping short acronyms (ECG, LSTM, PTB-XL)."""
    if not text.isupper():
        return text
    out = []
    for i, w in enumerate(text.split()):
        lw = w.lower()
        if lw in _TITLE_SMALL and i > 0:
            out.append(lw)
        elif len(re.sub(r"[^A-Za-z]", "", w)) <= 4 and re.search(r"\d|-", w) or (len(w) <= 3 and w.isalpha()):
            out.append(w)
        else:
            out.append(w.capitalize())
    return " ".join(out)


def _plausible_heading_text(t: str) -> bool:
    t = t.strip()
    if not t or len(t) > 110:
        return False
    if not (t[0].isupper() or t[0].isdigit() or not t[0].isascii()):
        return False
    last = re.sub(r"[^\w]+$", "", t.split()[-1].lower()) if t.split() else ""
    if last in ("the", "a", "an", "of", "and", "or", "to", "in", "for", "with", "using", "is", "are", "by", "that", "as", "on"):
        return False
    if _CAPTION.match(t):
        return False
    return True


def _looks_like_math(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 3:
        return False
    letters = sum(c.isalpha() for c in stripped)
    symbols = sum((not c.isalnum()) and (not c.isspace()) for c in stripped)
    has_eq_no = bool(re.search(r"\(\s*\d{1,3}\s*\)\s*$", stripped))
    if _MATHY.search(stripped) and (symbols / max(len(stripped), 1) > 0.18 or has_eq_no) and letters / max(len(stripped), 1) < 0.65:
        return True
    return False


def _heading_level_from_text(line: str) -> tuple[int, str, bool] | None:
    """Return (level, clean_title, strong) if the line looks like a heading by text pattern alone.

    ``strong`` patterns (roman numerals, decimal numbering, keywords, "Chapter N") are accepted
    regardless of font size; weak ones (``A. Dataset``) need typographic support.
    """
    s = line.strip()
    if not s or len(s) > 130 or _CAPTION.match(s):
        return None
    if s.endswith((".", ",", ";")) and not _CHAPTER.match(s):
        return None
    low = re.sub(r"^(?:[IVX]{1,5}\.|\d{1,2}(?:\.\d{1,2}){0,3}[.)]?)\s*", "", s).strip().lower().rstrip(":")
    if low in _KEYWORD_HEADINGS and len(s) < 48:
        title = re.sub(r"^(?:[IVX]{1,5}\.|\d{1,2}(?:\.\d{1,2}){0,3}[.)]?)\s*", "", s).strip().rstrip(":")
        return 1, smart_title(title), True
    m = _CHAPTER.match(s)
    if m and len(s) < 100:
        return 1, smart_title(s), True
    m = _ROMAN.match(s)
    if m and _plausible_heading_text(m.group("t")):
        return 1, smart_title(m.group("t").strip()), True
    m = _NUMBERED.match(s)
    if m:
        title = m.group("t").strip()
        if 1 <= len(title.split()) <= 14 and _plausible_heading_text(title) and not re.match(r"^\d+[.)]\s+\S+(\s+[a-z]+){4,}", s):
            return min(1 + m.group("num").count("."), 4), smart_title(title), True
    m = _LETTERED.match(s)
    if m and len(s.split()) <= 10 and _plausible_heading_text(m.group("t")):
        return 2, smart_title(m.group("t").strip()), False
    return None


# ---------------------------------------------------------------------------
# TXT / MD
# ---------------------------------------------------------------------------
def _parse_markdown_like(text: str, filename: str) -> list[Block]:
    blocks: list[Block] = []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    para: list[str] = []
    in_code = False

    emph_re = re.compile(r"\*\*([^*]{2,60})\*\*|__([^_]{2,60})__")

    def clean_md(t: str) -> str:
        return re.sub(r"[*_`]", "", t)

    def flush() -> None:
        if para:
            raw_joined = " ".join(para)
            emph = [a or b for a, b in emph_re.findall(raw_joined)]
            joined = normalize(clean_md(raw_joined))
            if joined:
                blk = Block("math" if _looks_like_math(joined) and len(joined) < 160 else "para", joined, 1)
                if emph:
                    blk.meta["emph"] = [normalize(e) for e in emph]
                blocks.append(blk)
            para.clear()

    for i, raw in enumerate(lines):
        line = raw.rstrip()
        if line.strip().startswith("```"):
            flush()
            in_code = not in_code
            continue
        if in_code:
            if line.strip():
                blocks.append(Block("math", line.strip(), 1, meta={"code": True}))
            continue
        if not line.strip():
            flush()
            continue
        m = re.match(r"^(#{1,6})\s+(.*?)\s*#*$", line)
        if m:
            flush()
            blocks.append(Block("heading", normalize(re.sub(r"[*_`]", "", m.group(2))), 1, level=len(m.group(1))))
            continue
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if nxt and re.fullmatch(r"=+|-+", nxt) and len(nxt) >= 3 and not para:
            flush()
            blocks.append(Block("heading", normalize(line), 1, level=1 if nxt[0] == "=" else 2))
            lines[i + 1] = ""
            continue
        if re.fullmatch(r"[=\-_*]{3,}", line.strip()):
            flush()
            continue
        b = _BULLET.match(line)
        if b and not _NUMBERED.match(line.strip()):
            flush()
            emph = [a or c for a, c in emph_re.findall(b.group("t"))]
            blk = Block("bullet", normalize(clean_md(b.group("t"))), 1)
            if emph:
                blk.meta["emph"] = [normalize(e) for e in emph]
            blocks.append(blk)
            continue
        h = _heading_level_from_text(line.strip()) if not para else None
        if h and h[2]:
            flush()
            blocks.append(Block("heading", normalize(h[1]), 1, level=h[0]))
            continue
        para.append(line.strip())
    flush()
    return blocks


def _paginate(blocks: list[Block], target: int = VIRTUAL_PAGE_CHARS) -> int:
    """Assign virtual page numbers (parts) to blocks without natural pages."""
    page, used = 1, 0
    for b in blocks:
        if used and used + len(b.text) > target and b.kind != "heading":
            page += 1
            used = 0
        if b.kind == "heading" and used > target * 0.6:
            page += 1
            used = 0
        b.page = page
        used += len(b.text)
    return page


def _parse_text(path: Path) -> ParsedDoc:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")
    blocks = _parse_markdown_like(text, path.name)
    title = next((b.text for b in blocks if b.kind == "heading" and b.level == 1), path.stem.replace("_", " ").strip())
    n = _paginate(blocks)
    return ParsedDoc(title=title, unit_label="Part", blocks=blocks, n_pages=n)


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------
def _parse_docx(path: Path) -> ParsedDoc:
    try:
        import docx  # python-docx
    except ImportError as exc:  # pragma: no cover
        raise ParseError("python-docx is not installed (pip install python-docx).") from exc
    try:
        d = docx.Document(str(path))
    except Exception as exc:
        raise ParseError(f"Could not open the Word file: {exc}") from exc
    blocks: list[Block] = []
    for p in d.paragraphs:
        t = normalize(p.text)
        if not t:
            continue
        style = (p.style.name or "").lower() if p.style is not None else ""
        m = re.match(r"heading\s*(\d)", style)
        if style == "title":
            blocks.append(Block("title", t, 1))
        elif m:
            blocks.append(Block("heading", t, 1, level=int(m.group(1))))
        elif "list" in style:
            blocks.append(Block("bullet", t, 1))
        else:
            h = _heading_level_from_text(t) if len(t) < 90 else None
            blocks.append(Block("heading", h[1], 1, level=h[0]) if h and h[2] else Block("para", t, 1))
    for tbl in d.tables:
        for row in tbl.rows:
            cells = [normalize(c.text) for c in row.cells if normalize(c.text)]
            if cells:
                blocks.append(Block("para", " | ".join(cells), 1, meta={"table": True}))
    title = next((b.text for b in blocks if b.kind in ("title", "heading")), path.stem)
    n = _paginate(blocks)
    return ParsedDoc(title=title, unit_label="Part", blocks=blocks, n_pages=n)


# ---------------------------------------------------------------------------
# PPTX
# ---------------------------------------------------------------------------
def _parse_pptx(path: Path) -> ParsedDoc:
    try:
        from pptx import Presentation
    except ImportError as exc:  # pragma: no cover
        raise ParseError("python-pptx is not installed (pip install python-pptx).") from exc
    try:
        prs = Presentation(str(path))
    except Exception as exc:
        raise ParseError(f"Could not open the PowerPoint file: {exc}") from exc
    blocks: list[Block] = []
    for idx, slide in enumerate(prs.slides, start=1):
        title_shape = slide.shapes.title if slide.shapes.title is not None else None
        title_text = normalize(title_shape.text_frame.text) if title_shape is not None and title_shape.has_text_frame else ""
        slide_blocks: list[Block] = []
        for shape in slide.shapes:
            if shape is title_shape:
                continue
            if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    t = normalize("".join(r.text for r in para.runs))
                    if t:
                        slide_blocks.append(Block("bullet" if para.level > 0 or len(t) < 140 else "para", t, idx, meta={"indent": para.level}))
            elif getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    cells = [normalize(c.text) for c in row.cells if normalize(c.text)]
                    if cells:
                        slide_blocks.append(Block("para", " | ".join(cells), idx, meta={"table": True}))
        if not title_text and slide_blocks:
            first = slide_blocks[0]
            if len(first.text) < 90:
                title_text = first.text
                slide_blocks = slide_blocks[1:]
        if title_text:
            blocks.append(Block("heading", title_text, idx, level=1, meta={"slide": True}))
        blocks.extend(slide_blocks)
    if not blocks:
        raise ParseError("No text found in the presentation (it may contain only images).")
    title = Path(path).stem.replace("_", " ")
    try:
        cp = prs.core_properties.title
        if cp and len(cp.strip()) > 3:
            title = cp.strip()
    except Exception:  # pragma: no cover
        pass
    return ParsedDoc(title=title, unit_label="Slide", blocks=blocks, n_pages=len(prs.slides))


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def _pdf_fragments(page: Any) -> list[dict[str, Any]]:
    frags: list[dict[str, Any]] = []

    def visitor(text: str, cm: list[float], tm: list[float], font: Any, size: float) -> None:
        if not text or not text.strip():
            return
        try:
            y = tm[4] * cm[1] + tm[5] * cm[3] + cm[5]
            eff = abs(size * (tm[3] if tm[3] else 1.0) * (cm[3] if cm[3] else 1.0))
        except Exception:
            y, eff = 0.0, size or 0.0
        base = ""
        try:
            base = str(font.get("/BaseFont", "")) if font else ""
        except Exception:
            base = ""
        frags.append({"t": text, "y": y, "s": eff or size or 0.0, "b": bool(re.search(r"bold|black|heavy|semibold|demi", base, re.IGNORECASE))})

    try:
        page.extract_text(visitor_text=visitor)
    except Exception as exc:
        logger.debug("visitor extraction failed: %s", exc)
    return frags


def _fragment_lines(frags: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Merge fragments that share a baseline; map whitespace-free text -> style info."""
    lines: list[dict[str, Any]] = []
    for f in frags:
        if lines and abs(lines[-1]["y"] - f["y"]) < max(1.0, 0.3 * (f["s"] or 8)):
            cur = lines[-1]
            cur["t"] += f["t"]
            cur["chars"] += len(f["t"].strip())
            cur["bold_chars"] += len(f["t"].strip()) if f["b"] else 0
            cur["size_chars"][round(f["s"], 1)] += len(f["t"].strip())
        else:
            lines.append({"t": f["t"], "y": f["y"], "chars": len(f["t"].strip()), "bold_chars": len(f["t"].strip()) if f["b"] else 0, "size_chars": Counter({round(f["s"], 1): len(f["t"].strip())})})
    out: dict[str, dict[str, Any]] = {}
    prev_y = None
    for ln in lines:
        key = re.sub(r"\s+", "", ln["t"])
        if not key:
            continue
        size = ln["size_chars"].most_common(1)[0][0] if ln["size_chars"] else 0.0
        gap = abs(prev_y - ln["y"]) if prev_y is not None else 0.0
        prev_y = ln["y"]
        out.setdefault(key, {"size": size, "bold": ln["bold_chars"] / max(ln["chars"], 1) > 0.6, "gap": gap})
    return out


def _dehyphenate(lines: list[str]) -> list[str]:
    """Join lines; soft hyphens vanish (classi-/fication), real ones stay (Self-/Supervised)."""
    out: list[str] = []
    for ln in lines:
        if out and out[-1].endswith("-") and len(out[-1]) > 2 and out[-1][-2].isalpha():
            if ln[:1].islower():
                out[-1] = out[-1][:-1] + ln
            else:
                out[-1] = out[-1] + ln
        else:
            out.append(ln)
    return out


def _is_tabular(text: str) -> bool:
    """Numeric-heavy paragraphs (result tables flattened by PDF extraction)."""
    toks = text.split()
    if len(toks) < 8:
        return False
    num = sum(1 for t in toks if re.search(r"\d", t) and re.fullmatch(r"[\d.,%±+\-/()]+[A-Za-z%]{0,2}", t))
    return num / len(toks) > 0.22


def _parse_pdf(path: Path) -> ParsedDoc:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ParseError("pypdf is not installed (pip install pypdf).") from exc
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise ParseError("This PDF is password protected.") from exc
        n_pages = len(reader.pages)
    except ParseError:
        raise
    except Exception as exc:
        raise ParseError(f"Could not open the PDF: {exc}") from exc
    if n_pages > MAX_PAGES:
        raise ParseError(f"PDF has {n_pages} pages; the limit is {MAX_PAGES}. Split it into smaller files.")

    page_lines: list[list[tuple[str, dict[str, Any] | None]]] = []
    size_votes: Counter[float] = Counter()
    for page in reader.pages:
        try:
            plain = page.extract_text() or ""
        except Exception:
            plain = ""
        fl = _fragment_lines(_pdf_fragments(page))
        rows: list[tuple[str, dict[str, Any] | None]] = []
        for raw in plain.split("\n"):
            ln = normalize(raw)
            if not ln:
                continue
            info = fl.get(re.sub(r"\s+", "", ln))
            rows.append((ln, info))
            if info:
                size_votes[info["size"]] += len(ln)
        page_lines.append(rows)

    total_chars = sum(len(t) for rows in page_lines for t, _ in rows)
    if total_chars < 40:
        raise ParseError(
            "No selectable text found - this looks like a scanned PDF. Run OCR on it first "
            "(e.g. OCRmyPDF with Telugu/Hindi language packs) and upload the searchable PDF."
        )
    body = size_votes.most_common(1)[0][0] if size_votes else 0.0

    # repeated header/footer detection
    sig: Counter[str] = Counter()
    for rows in page_lines:
        seen = set()
        for t, _ in rows[:2] + rows[-2:]:
            k = re.sub(r"\d+", "#", t.lower())
            if k not in seen:
                sig[k] += 1
                seen.add(k)
    repeated = {k for k, c in sig.items() if n_pages >= 4 and c >= max(3, int(0.4 * n_pages))}

    blocks: list[Block] = []
    title_text = ""
    in_refs = False
    in_caption = False
    for pno, rows in enumerate(page_lines, start=1):
        para: list[str] = []
        para_page = pno

        def flush(_para: list[str] = para, _pno: int = pno) -> None:
            nonlocal in_refs
            if _para:
                text = " ".join(_dehyphenate(_para))
                text = normalize(text)
                if text:
                    kind = "ref" if in_refs else ("math" if _looks_like_math(text) and len(text) < 180 else "para")
                    blk = Block(kind, text, _pno)
                    if kind == "para" and _is_tabular(text):
                        blk.meta["table"] = True
                    blocks.append(blk)
                _para.clear()

        # first page title: biggest consecutive lines
        if pno == 1 and body:
            big = [(t, i) for i, (t, inf) in enumerate(rows[:12]) if inf and inf["size"] >= body * 1.5]
            if big:
                top_size = max(inf["size"] for _, inf in rows[:12] if inf)
                tl = [t for t, inf in rows[:12] if inf and abs(inf["size"] - top_size) < 0.6 and len(t) < 160]
                if tl:
                    title_text = normalize(" ".join(tl))
        for i, (ln, info) in enumerate(rows):
            if _PAGE_NUM.match(ln) or re.sub(r"\d+", "#", ln.lower()) in repeated:
                continue
            if title_text and pno == 1 and ln in title_text:
                continue
            size = info["size"] if info else body
            bold = bool(info and info["bold"])
            level = 0
            clean = ln
            if _CAPTION.match(ln):
                flush()
                blocks.append(Block("para", ln, pno, meta={"caption": True}))
                in_caption = True
                continue
            h = _heading_level_from_text(ln)
            if in_caption and ln.isupper() and not (h and h[2]) and blocks and blocks[-1].text.upper().startswith("TABLE"):
                blocks[-1].text += " " + ln
                continue
            in_caption = False
            if h and (h[2] or not info or size >= body * 0.94 or bold):
                level, clean = h[0], h[1]
            if not level and body and size >= body * 1.18 and len(ln) <= 110 and not ln.endswith((".", ",")) and _plausible_heading_text(ln):
                level = 1 if size >= body * 1.45 else 2
                clean = smart_title(ln)
            if not level and bold and len(ln) <= 70 and not ln.endswith((".", ",", ":")) and len(ln.split()) <= 9 and body and size >= body * 0.94 and not _BULLET.match(ln) and _plausible_heading_text(ln):
                level = 3
                clean = smart_title(ln)
            if not level and ln.isupper() and 2 <= len(ln.split()) <= 8 and len(ln) <= 60 and re.search(r"[A-Z]{3}", ln) and not _looks_like_math(ln) and _plausible_heading_text(ln):
                level = 2
                clean = smart_title(ln)
            if level:
                flush()
                low = re.sub(r"^[\dIVX.\s]+", "", clean, flags=re.IGNORECASE).strip().lower().rstrip(":")
                in_refs = low in _REF_HEADINGS
                blocks.append(Block("heading", clean.strip(), pno, level=level))
                continue
            b = _BULLET.match(ln)
            if b and para:
                flush()
            if b and not in_refs:
                flush()
                blocks.append(Block("bullet", normalize(b.group("t")), pno))
                continue
            if info and info["gap"] and body and info["gap"] > 1.9 * (size or body) and para:
                flush()
            para.append(ln)
            if ln.endswith((".", "!", "?", ":")) and len(ln) < 0.55 * 95 and para:
                flush()
        flush()
        _ = para_page

    meta_title = ""
    try:
        mt = (reader.metadata.title or "").strip() if reader.metadata else ""
        if len(mt) > 8 and not re.search(r"microsoft word|untitled|\.docx?$|\.tex$", mt, re.IGNORECASE):
            meta_title = mt
    except Exception:  # pragma: no cover
        pass
    title = title_text or meta_title or path.stem.replace("_", " ")
    if len(title) > 160:
        title = title[:157].rsplit(" ", 1)[0] + "..."
    warnings: list[str] = []
    if not any(b.kind == "heading" for b in blocks):
        warnings.append("No headings detected - topics were inferred from the text itself.")
    return ParsedDoc(title=title, unit_label="Page", blocks=blocks, n_pages=n_pages, warnings=warnings)


# ---------------------------------------------------------------------------
def parse_file(path: str | Path, filename: str | None = None) -> ParsedDoc:
    """Parse *path* into a :class:`ParsedDoc` or raise :class:`ParseError`."""
    path = Path(path)
    ext = Path(filename or path.name).suffix.lower()
    if ext in UNSUPPORTED_HINT:
        raise ParseError(UNSUPPORTED_HINT[ext])
    if ext not in SUPPORTED_EXTENSIONS:
        raise ParseError(f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.")
    if ext == ".pdf":
        doc = _parse_pdf(path)
    elif ext == ".pptx":
        doc = _parse_pptx(path)
    elif ext == ".docx":
        doc = _parse_docx(path)
    else:
        doc = _parse_text(path)
    # drop empty / trivial blocks
    doc.blocks = [b for b in doc.blocks if b.text and len(b.text.strip()) > 1]
    if sum(len(b.text) for b in doc.blocks) < 40:
        raise ParseError("The file has almost no readable text.")
    if not doc.title.strip():
        doc.title = Path(filename or path.name).stem
    info = detect_document_language(doc.full_text)
    doc.language = info.base if info.code != "unknown" else "en"
    doc.script = info.script
    return doc
