"""Structure-preserving document parsing (PDF, PPTX, DOCX, TXT, MD).

Unlike a plain text extractor this keeps *structure*: headings (with level), paragraphs, bullets
and equation-like lines, each tagged with its page / slide.  The headings are what lets the
default knowledge map follow the real topic hierarchy of the material instead of guessing.

PDF heading detection uses font sizes (via pypdf's text visitor) relative to the body-text size,
numbering patterns (``2.1 Method``, ``II. RELATED WORK``, ``A. Dataset``), and bold short lines.
"""

from __future__ import annotations

import logging
import os
import re
import threading
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
    # chart axes, years, table cells and citations that OCR/slide text turns into fake titles
    # ("40 60 80", "2009 2010 2011", "From Gelfand et al. (2010, pg.5)", "100% x p 3.00")
    chars = re.sub(r"\s", "", t)
    letters = len(re.findall(r"[^\W\d_]", chars))
    if (letters < 4 and not chars.isupper()) or letters / max(1, len(chars)) < 0.55 or len(re.findall(r"\d", chars)) / max(1, len(chars)) > 0.25:
        return False
    if _CITATION.search(t) or re.search(r"[%×$≈±]", t):
        return False
    words = re.findall(r"[^\W\d_]+", t)
    if words and sum(len(w) == 1 for w in words) / len(words) > 0.5:
        return False
    if _FIGURE.match(t):  # "Figure4O: ...", "FIGURE 1.1" (OCR garbles the number, so _CAPTION misses it)
        return False
    low = [w.lower() for w in words]
    if len(low) >= 3 and len(set(low)) / len(low) < 0.6:  # "Latitude Latitude Latitude"
        return False
    toks = t.split()
    nums = [i for i, w in enumerate(toks) if re.fullmatch(r"\d{1,4}", w)]
    if any(0 < i < len(toks) - 1 and toks[i - 1].lower().strip(":") not in _NUMBER_WORDS for i in nums):  # "Ashok Vihar 22 Narela": map/chart labels
        return False
    if nums and nums[0] == 0 and len(toks) <= 3 and not re.match(r"^\d{1,2}[.)]", t):  # "400 Elevation"
        return False
    return True


_NUMBER_WORDS = {"chapter", "unit", "part", "section", "lecture", "module", "week", "raid", "level", "phase", "step", "type", "class",
             "round", "version", "ipv", "rule", "case", "example", "problem", "question", "lab", "experiment", "day"}
_FIGURE = re.compile(r"^(fig(ure)?|table)\s*[\dlIO]", re.IGNORECASE)


_CITATION = re.compile(r"\bet\s*al\b|\bpg\.\s*\d|\(\s*(?:19|20)\d\d\b|^(?:from|source|adapted from)\b\s*[A-Z]", re.IGNORECASE)


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
def _iter_shapes(shapes: Any):
    """Every shape on a slide, including the ones inside groups (python-pptx does not recurse)."""
    for sh in shapes:
        yield sh
        if getattr(sh, "shape_type", None) == 6 and hasattr(sh, "shapes"):  # MSO_SHAPE_TYPE.GROUP
            yield from _iter_shapes(sh.shapes)


def _shape_picture(sh: Any) -> bytes | None:
    """Image bytes of a picture shape / picture placeholder / picture-filled shape, else None."""
    try:
        if sh.shape_type == 13 or hasattr(sh, "image"):  # PICTURE
            return sh.image.blob
    except Exception:
        pass
    try:  # a shape filled with a picture
        blip = sh._element.xpath(".//a:blip/@r:embed")
        if blip:
            return sh.part.related_part(blip[0]).blob
    except Exception:
        pass
    return None


def _ocr_blocks(lines: list[Any], page: int) -> list[Block]:
    """OCR lines -> blocks; the tallest first line becomes the slide title."""
    out: list[Block] = []
    if not lines:
        return out
    heights = sorted(ln.height for ln in lines)
    body_h = heights[len(heights) // 2] or 1.0
    for i, ln in enumerate(lines):
        t = normalize(ln.text)
        if not t:
            continue
        big = ln.height >= 1.35 * body_h and len(t) <= 90 and i < 3 and _plausible_heading_text(t)
        out.append(Block("heading" if big else ("bullet" if len(t) < 140 else "para"), t, page,
                         level=1 if big else 0, meta={"ocr": True}))
    return out


def _parse_pptx(path: Path) -> ParsedDoc:
    try:
        from pptx import Presentation
    except ImportError as exc:  # pragma: no cover
        raise ParseError("python-pptx is not installed (pip install python-pptx).") from exc
    from indiclearn.studio import ocr

    try:
        prs = Presentation(str(path))
    except Exception as exc:
        raise ParseError(f"Could not open the PowerPoint file: {exc}") from exc
    blocks: list[Block] = []
    warnings: list[str] = []
    pics_seen = pics_read = 0
    max_imgs = int(os.environ.get("INDICLEARN_OCR_MAX_IMAGES", "150"))
    n_slides = len(prs.slides)
    for idx, slide in enumerate(prs.slides, start=1):
        _report(idx - 1, n_slides, "reading")
        title_shape = slide.shapes.title if slide.shapes.title is not None else None
        title_text = normalize(title_shape.text_frame.text) if title_shape is not None and title_shape.has_text_frame else ""
        slide_blocks: list[Block] = []
        pictures: list[bytes] = []
        for shape in _iter_shapes(slide.shapes):
            if title_shape is not None and shape.shape_id == title_shape.shape_id:
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
            else:
                blob = _shape_picture(shape)
                if blob:
                    pictures.append(blob)
        # text that is stored as a picture (screenshots, scanned figures, exported slides)
        ocr_blocks: list[Block] = []
        for blob in pictures:
            pics_seen += 1
            if pics_read >= max_imgs:
                continue
            pics_read += 1
            ocr_blocks.extend(_ocr_blocks(ocr.ocr_bytes(blob), idx))
        if not title_text:
            first_head = next((b for b in ocr_blocks if b.kind == "heading"), None)
            if first_head is not None and not any(b.kind == "heading" for b in slide_blocks):
                title_text = first_head.text
                ocr_blocks.remove(first_head)
        for b in ocr_blocks:
            if b.kind == "heading":  # keep only one title per slide; the rest are plain lines
                b.kind, b.level = "bullet", 0
        slide_blocks.extend(ocr_blocks)
        if not title_text and slide_blocks:
            first = slide_blocks[0]
            if len(first.text) < 90:
                title_text = first.text
                slide_blocks = slide_blocks[1:]
        if title_text:
            blocks.append(Block("heading", title_text, idx, level=1, meta={"slide": True}))
        blocks.extend(slide_blocks)
        try:  # speaker notes often hold the real explanation
            if slide.has_notes_slide:
                nt = normalize(slide.notes_slide.notes_text_frame.text)
                if len(nt) > 30:
                    blocks.append(Block("para", nt, idx, meta={"notes": True}))
        except Exception:  # pragma: no cover
            pass
    if pics_seen and ocr.get_engine() is None:
        warnings.append(f"{pics_seen} picture(s) on the slides may contain text that could not be read. {ocr.INSTALL_HINT}")
    elif pics_read:
        warnings.append(f"Read text from {pics_read} picture(s) with OCR ({ocr.get_engine().name}); please double-check technical terms.")
    if not blocks:
        raise ParseError("No text found in the presentation. It looks like it contains only pictures. " + ocr.INSTALL_HINT)
    title = Path(path).stem.replace("_", " ")
    try:
        cp = prs.core_properties.title
        if cp and len(cp.strip()) > 3:
            title = cp.strip()
    except Exception:  # pragma: no cover
        pass
    return ParsedDoc(title=title, unit_label="Slide", blocks=blocks, n_pages=len(prs.slides), warnings=warnings)


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


_LIG_FILL = ("fi", "fl", "ff", "ffi", "ffl")
# Words that commonly lose an fi/fl/ff ligature in engineering PDFs and may never appear
# correctly anywhere in the same document, so the document's own vocabulary cannot repair them.
_LIG_WORDS = frozenset(["artificial", "classification", "identification", "specification", "verification", "amplification", "simplification", "notification", "modification", "justification", "quantification", "qualification", "stratification", "efficient", "efficiency", "sufficient", "coefficient", "deficient", "proficient", "significant", "significance", "magnificent", "beneficial", "artificially", "specifically", "scientific", "specific", "pacific", "identifier", "classifier", "amplifier", "verifier", "modifier", "notifier", "certificate", "certification", "define", "defined", "defines", "defining", "definition", "definitions", "finite", "infinite", "confine", "confined", "refine", "refined", "confidence", "confident", "configuration", "confirm", "confirmed", "first", "firstly", "fifth", "fifty", "filter", "filtered", "filtering", "filters", "final", "finally", "finance", "financial", "find", "finding", "findings", "fine", "finish", "fixed", "fitness", "flow", "flows", "flowing", "float", "floating", "floor", "fluid", "flux", "reflect", "reflected", "reflection", "reflects", "affine", "affinity", "benefit", "benefits", "profile", "profiles", "profiling", "difficult", "difficulty", "different", "difference", "differences", "differential", "diffusion", "diffuse", "offline", "offset", "office", "officer", "effort", "efforts", "effective", "effectively", "effect", "effects", "affect", "affected", "affects", "buffer", "buffers", "staff", "stiff", "stiffness", "off", "offer", "offers", "suffix", "suffer", "suffering", "traffic", "trafficking", "classifications", "identifications", "amplifications", "verifications", "certificates", "coefficients", "filtration", "infiltration", "ratification", "ramification", "unification", "clarification"])


def _repair_ligature_gaps(blocks: list[Block]) -> None:
    """Rejoin words a PDF split where an fi/fl/ff ligature was dropped ("Arti cial").

    A join is accepted only when the joined word is real: either it appears intact elsewhere in
    the same document, or it is in the bundled list above. Unknown pairs are left exactly as
    they are, so ordinary prose is never mangled.
    """
    vocab: set[str] = set()
    for b in blocks:
        for w in re.findall(r"[A-Za-z]{4,}", b.text):
            vocab.add(w.lower())

    def join(a: str, b_: str) -> str | None:
        for lig in _LIG_FILL:
            cand = a + lig + b_
            if cand.lower() in vocab or cand.lower() in _LIG_WORDS:
                return cand
        return None

    pat = re.compile(r"\b([A-Za-z]{2,}) ([a-z]{2,})\b")
    for b in blocks:
        text = b.text
        if " " not in text:
            continue
        out: list[str] = []
        i = 0
        while True:
            m = pat.search(text, i)
            if not m:
                break
            merged = join(m.group(1), m.group(2))
            if merged:
                out.append(text[i : m.start()])
                out.append(merged)
                i = m.end()
            else:
                # Not a real word: step to the second word so IT can start the next pair.
                # ("and classi cation" must still find "classi"+"cation".)
                out.append(text[i : m.start(2)])
                i = m.start(2)
        if out:
            out.append(text[i:])
            b.text = "".join(out)


OCR_MIN_CHARS = 60
_HEADWORDS = frozenset(["abstract", "introduction", "background", "related", "work", "works", "preliminaries", "methodology", "method", "methods", "approach", "overview", "experiments", "experiment", "evaluation", "results", "discussion", "conclusion", "conclusions", "references", "acknowledgment", "acknowledgments", "acknowledgements", "appendix", "dataset", "datasets", "analysis", "applications", "application", "architecture", "implementation", "motivation", "problem", "formulation", "framework", "model", "models", "theory", "summary", "contents", "outline", "objectives", "objective"])
_NOJOIN_A = frozenset(["a", "i"])


def _build_vocab(texts: list[str]) -> set[str]:
    """Lower-case words that appear intact (not ALL-CAPS) somewhere in the document."""
    v: set[str] = set()
    for t in texts:
        for w in re.findall(r"[A-Za-z]{3,}", t):
            if not w.isupper():
                v.add(w.lower())
    return v


_SMALLCAPS = re.compile(r"\b([A-Z]) ([A-Z]{2,})\b")
_SMALLCAPS_TC = re.compile(r"\b([A-Z]) ([A-Z][a-z]{2,})\b")
_SPLIT_WORD = re.compile(r"\b([A-Za-z]{1,2}) ([a-z]{3,})\b")
_CITE_RESIDUE = re.compile(r"(?:\s*,){2,}")


def _fix_line(line: str, vocab: set[str]) -> str:
    """Repair typical PDF text damage on one line:

    * small-caps headings    "I NTRODUCTION" -> "INTRODUCTION"
    * dropped first letters   "e ffectively" -> "effectively"
    * citation numbers lost   "years , , , ,"  -> "years"
    A join happens only when the joined word really exists in this document (or is a known heading word).
    """
    if " " not in line and "," not in line:
        return line

    def ok(word: str) -> bool:
        w = word.lower()
        return w in vocab or w in _HEADWORDS or w in _LIG_WORDS

    line = _SMALLCAPS.sub(lambda m: m.group(1) + m.group(2) if ok(m.group(1) + m.group(2)) else m.group(0), line)
    line = _SMALLCAPS_TC.sub(lambda m: (m.group(1) + m.group(2).lower()).capitalize() if ok(m.group(1) + m.group(2)) else m.group(0), line)

    def split_fix(m: re.Match[str]) -> str:
        a, b = m.group(1), m.group(2)
        if a.lower() in _NOJOIN_A and not b.startswith(("ff", "fi", "fl")):
            return m.group(0)
        if len(a) == 2 and a.lower() in _STOP2:
            return m.group(0)
        return a + b if ok(a + b) else m.group(0)

    line = _SPLIT_WORD.sub(split_fix, line)
    if "," in line:
        line = _CITE_RESIDUE.sub("", line)
        line = re.sub(r"\s+([,.;:])", r"\1", line)
        line = re.sub(r"\(\s*\)|\[\s*\]", "", line)
    return line


_STOP2 = frozenset(["an", "as", "at", "be", "by", "do", "go", "he", "if", "in", "is", "it", "me", "my", "no", "of", "on", "or", "so", "to", "up", "us", "we"])


class _PageRenderer:
    """Turns a PDF page into a PIL image for OCR: pypdfium2 if present, else the page's embedded pictures."""

    def __init__(self, path: Path, reader: Any) -> None:
        self.path, self.reader = path, reader
        self._pdf: Any = None
        self._tried = False

    def _open(self) -> Any:
        if not self._tried:
            self._tried = True
            try:
                import pypdfium2 as pdfium

                self._pdf = pdfium.PdfDocument(str(self.path))
            except Exception as exc:
                logger.info("pypdfium2 unavailable (%s) - using embedded images for OCR", exc)
        return self._pdf

    def has_images(self, pno: int) -> bool:
        try:
            return len(self.reader.pages[pno].images) > 0
        except Exception:
            return False

    def image(self, pno: int) -> Any | None:
        pdf = self._open()
        if pdf is not None:
            try:
                return pdf[pno].render(scale=2.5).to_pil()
            except Exception as exc:
                logger.info("page render failed: %s", exc)
        try:
            import io

            from PIL import Image

            ims = [Image.open(io.BytesIO(i.data)) for i in self.reader.pages[pno].images]
            ims = [i for i in ims if min(i.size) >= 80]
            return max(ims, key=lambda i: i.size[0] * i.size[1]) if ims else None
        except Exception:
            return None


def _is_tabular(text: str) -> bool:
    """Numeric-heavy paragraphs (result tables flattened by PDF extraction)."""
    toks = text.split()
    if len(toks) < 4:
        return False
    plain = sum(1 for t in toks if re.fullmatch(r"\d+(?:[.,]\d+)?%?", t))
    if plain >= 3 and plain / len(toks) > 0.28:
        return True
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

    from indiclearn.studio import ocr

    raw_pages: list[dict[str, Any]] = []
    ocr_done = ocr_skipped = 0
    max_ocr = int(os.environ.get("INDICLEARN_OCR_MAX_PAGES", "80"))
    renderer = _PageRenderer(path, reader)
    for pno, page in enumerate(reader.pages):
        _report(pno, n_pages, "reading")
        try:
            plain = page.extract_text() or ""
        except Exception:
            plain = ""
        entry: dict[str, Any] = {"plain": plain, "frags": None, "ocr": None}
        if len(re.sub(r"\s+", "", plain)) < OCR_MIN_CHARS:
            # (nearly) no selectable text: a scan, a slide exported as a picture, or a page of screenshots
            img = renderer.image(pno) if ocr_done < max_ocr else None
            if img is not None and ocr.get_engine() is not None:
                _report(pno, n_pages, "ocr")
                lines = ocr.ocr_image(img)
                ocr_done += 1
                if lines:
                    entry["ocr"] = lines
            elif img is not None or renderer.has_images(pno):
                ocr_skipped += 1
        if entry["ocr"] is None:
            entry["frags"] = _fragment_lines(_pdf_fragments(page))
        raw_pages.append(entry)

    vocab = _build_vocab([e["plain"] for e in raw_pages] + [ln.text for e in raw_pages for ln in (e["ocr"] or [])])
    page_lines: list[list[tuple[str, dict[str, Any] | None]]] = []
    size_votes: Counter[float] = Counter()
    for e in raw_pages:
        rows: list[tuple[str, dict[str, Any] | None]] = []
        if e["ocr"]:
            hs = sorted(ln.height for ln in e["ocr"])
            med = hs[len(hs) // 2] or 1.0
            prev = None
            slide_like = sorted(len(ln.text) for ln in e["ocr"])[len(e["ocr"]) // 2] < 55
            for ln in e["ocr"]:
                t = normalize(_fix_line(ln.text, vocab))
                if not t:
                    continue
                info = {"size": round(10.0 * ln.height / med * 2) / 2, "bold": False,
                        "gap": 999.0 if slide_like else (0.0 if prev is None else (ln.top - prev) / med * 10.0)}
                prev = ln.top
                rows.append((t, info))
                size_votes[info["size"]] += len(t)
        else:
            fl = e["frags"] or {}
            for raw in e["plain"].split("\n"):
                ln_ = normalize(_fix_line(raw, vocab))
                if not ln_:
                    continue
                info = fl.get(re.sub(r"\s+", "", normalize(raw))) or fl.get(re.sub(r"\s+", "", ln_))
                rows.append((ln_, info))
                if info:
                    size_votes[info["size"]] += len(ln_)
        page_lines.append(rows)

    total_chars = sum(len(t) for rows in page_lines for t, _ in rows)
    if total_chars < 40:
        if ocr.get_engine() is None:
            raise ParseError("No selectable text found - this looks like a scanned PDF. " + ocr.INSTALL_HINT)
        raise ParseError("No readable text found - the pages look empty, or the scan is too blurry for OCR.")
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
            if level and in_refs:
                level = 0  # inside References nothing is a heading: entries are not topics
            if level:
                flush()
                low = re.sub(r"^[\dIVX.\s]+", "", clean, flags=re.IGNORECASE).strip().lower().rstrip(":")
                in_refs = in_refs or low in _REF_HEADINGS
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

    _repair_ligature_gaps(blocks)
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
    if ocr_done and any(e["ocr"] for e in raw_pages):
        warnings.append(f"{sum(1 for e in raw_pages if e['ocr'])} page(s) had no selectable text and were read with OCR "
                        f"({ocr.get_engine().name}); please double-check technical terms.")
    if ocr_skipped:
        warnings.append(f"{ocr_skipped} page(s) are pictures and were not read. {ocr.INSTALL_HINT}")
    if not any(b.kind == "heading" for b in blocks):
        warnings.append("No headings detected - topics were inferred from the text itself.")
    return ParsedDoc(title=title, unit_label="Page", blocks=blocks, n_pages=n_pages, warnings=warnings)


# ---------------------------------------------------------------------------
_PROGRESS = threading.local()


def _report(done: int, total: int, stage: str = "reading") -> None:
    cb = getattr(_PROGRESS, "cb", None)
    if cb is not None:
        try:
            cb(done, total, stage)
        except Exception:  # pragma: no cover - progress is cosmetic
            pass


def parse_file(path: str | Path, filename: str | None = None, progress: Any = None) -> ParsedDoc:
    """Parse *path* into a :class:`ParsedDoc` or raise :class:`ParseError`.
    progress(done, total, stage) is called per page/slide so the UI can say "Reading page 12 of 41"."""
    _PROGRESS.cb = progress
    try:
        return _parse_file(path, filename)
    finally:
        _PROGRESS.cb = None


def _parse_file(path: str | Path, filename: str | None = None) -> ParsedDoc:
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
    display = re.sub(r"(\.(pptx?|docx?|pdf|md|txt))+$", "", Path(filename or path.name).stem, flags=re.IGNORECASE)  # "Lecture2.pptx.pdf"
    display = re.sub(r"\s+", " ", display.replace("_", " ")).strip()
    bad = not doc.title.strip() or doc.title.strip() == path.stem or bool(re.fullmatch(r"[0-9a-f]{12,}", doc.title.strip()))
    if bad:
        doc.title = display or path.stem
    # titles copied from file metadata often end in the source file's extension ("Lecture2.pptx")
    doc.title = re.sub(r"\s+", " ", re.sub(r"(\.(pptx?|docx?|pdf|key|odp))+$", "", doc.title, flags=re.IGNORECASE).replace("_", " ")).strip() or doc.title
    info = detect_document_language(doc.full_text)
    doc.language = info.base if info.code != "unknown" else "en"
    doc.script = info.script
    return doc
