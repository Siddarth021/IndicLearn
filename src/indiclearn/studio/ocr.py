"""Read text that is stored as a PICTURE (scanned PDF pages, screenshots on slides, photographed notes).

Engines, all free and local (first one that is installed wins; ``INDICLEARN_OCR`` forces one):

  rapidocr   pip install rapidocr-onnxruntime        small, no PyTorch, Windows-friendly   (English/Latin)
  easyocr    pip install easyocr                      uses PyTorch; Telugu, Hindi, ... too
  tesseract  Tesseract program + pytesseract         add ``tel``/``hin`` language data for Indic text

``INDICLEARN_OCR_LANGS`` (default ``en``) lists the languages to expect, e.g. ``en,te``. With an Indic
language listed, EasyOCR / Tesseract are preferred because RapidOCR only reads Latin text.
"""

from __future__ import annotations

import io
import logging
import os
import re
import shutil
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

MIN_SIDE = 40          # ignore icons / bullets smaller than this (pixels)
MAX_SIDE = 3200        # shrink huge scans before OCR
TARGET_WIDTH = 1400    # enlarge small screenshots so thin fonts become readable
_EASY_CODES = {"en": "en", "te": "te", "hi": "hi", "ta": "ta", "kn": "kn", "mr": "mr", "bn": "bn", "ne": "ne", "sa": "hi"}
_TESS_CODES = {"en": "eng", "te": "tel", "hi": "hin", "ta": "tam", "kn": "kan", "mr": "mar", "bn": "ben", "sa": "san"}


@dataclass
class OcrLine:
    text: str
    top: float       # pixels from the top of the image
    height: float    # text height in pixels (used to spot titles)
    left: float = 0.0


def wanted_langs() -> list[str]:
    raw = os.environ.get("INDICLEARN_OCR_LANGS", "en")
    out = [x.strip().lower() for x in raw.split(",") if x.strip()]
    return out or ["en"]


def _tesseract_cmd() -> str | None:
    exe = shutil.which("tesseract")
    if exe:
        return exe
    for p in (r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
        if os.path.exists(p):
            return p
    return None


def _prep(img: Any) -> Any:
    from PIL import Image

    img = img.convert("RGB")
    w, h = img.size
    if w < TARGET_WIDTH and w >= MIN_SIDE:
        s = TARGET_WIDTH / w
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    elif max(w, h) > MAX_SIDE:
        s = MAX_SIDE / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return img


def _merge_row(items: list[tuple[str, float, float, float]]) -> list[OcrLine]:
    """Boxes (text, left, top, height) -> reading-order lines; boxes on the same row are joined."""
    items = sorted(items, key=lambda t: (t[2] + t[3] / 2, t[1]))
    rows: list[list[tuple[str, float, float, float]]] = []
    for it in items:
        cy = it[2] + it[3] / 2
        if rows and abs(cy - (rows[-1][0][2] + rows[-1][0][3] / 2)) < 0.5 * max(it[3], rows[-1][0][3]):
            rows[-1].append(it)
        else:
            rows.append([it])
    out: list[OcrLine] = []
    for r in rows:
        r.sort(key=lambda t: t[1])
        text = " ".join(t[0].strip() for t in r if t[0].strip())
        if text:
            out.append(OcrLine(text, min(t[2] for t in r), max(t[3] for t in r), r[0][1]))
    return out


class _Rapid:
    name = "rapidocr"
    latin_only = True

    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR

        self.eng = RapidOCR()

    def read(self, img: Any) -> list[OcrLine]:
        import numpy as np

        res, _ = self.eng(np.array(img)[:, :, ::-1])
        items = []
        for box, text, score in res or []:
            if float(score) < 0.5:
                continue
            ys = [p[1] for p in box]
            xs = [p[0] for p in box]
            items.append((str(text), min(xs), min(ys), max(ys) - min(ys)))
        return _merge_row(items)


class _Easy:
    name = "easyocr"
    latin_only = False

    def __init__(self, langs: list[str]) -> None:
        import easyocr

        codes = [c for c in dict.fromkeys(_EASY_CODES.get(x, "") for x in langs) if c]
        # EasyOCR can combine English with ONE other script at a time
        non_en = [c for c in codes if c != "en"][:1]
        use_gpu = os.environ.get("INDICLEARN_OCR_GPU", "0").lower() in ("1", "true", "yes")
        self.reader = easyocr.Reader(["en", *non_en], gpu=use_gpu, verbose=False)

    def read(self, img: Any) -> list[OcrLine]:
        import numpy as np

        items = []
        for box, text, conf in self.reader.readtext(np.array(img)):
            if float(conf) < 0.35:
                continue
            ys = [p[1] for p in box]
            xs = [p[0] for p in box]
            items.append((str(text), min(xs), min(ys), max(ys) - min(ys)))
        return _merge_row(items)


class _Tess:
    name = "tesseract"
    latin_only = False

    def __init__(self, langs: list[str]) -> None:
        import pytesseract

        exe = _tesseract_cmd()
        if not exe:
            raise RuntimeError("tesseract program not found")
        pytesseract.pytesseract.tesseract_cmd = exe
        have = set(pytesseract.get_languages(config=""))
        want = [_TESS_CODES[x] for x in langs if x in _TESS_CODES and _TESS_CODES[x] in have]
        self.lang = "+".join(dict.fromkeys(want or (["eng"] if "eng" in have else list(have)[:1])))
        self.pt = pytesseract

    def read(self, img: Any) -> list[OcrLine]:
        d = self.pt.image_to_data(img, lang=self.lang, output_type=self.pt.Output.DICT)
        items = []
        for i, t in enumerate(d["text"]):
            if t.strip() and float(d["conf"][i]) >= 40:
                items.append((t, float(d["left"][i]), float(d["top"][i]), float(d["height"][i])))
        # Tesseract gives words: join words that share a row
        return _merge_row(items)


_lock = threading.Lock()
_cache: dict[str, Any] = {}
_errors: dict[str, str] = {}


def _build(kind: str, langs: list[str]) -> Any | None:
    key = kind + ":" + ",".join(langs)
    with _lock:
        if key in _cache:
            return _cache[key]
        try:
            eng = _Rapid() if kind == "rapidocr" else _Easy(langs) if kind == "easyocr" else _Tess(langs)
        except Exception as e:
            _errors[kind] = f"{type(e).__name__}: {str(e)[:160]}"
            logger.info("OCR engine %s unavailable: %s", kind, e)
            eng = None
        _cache[key] = eng
        return eng


def get_engine(langs: list[str] | None = None) -> Any | None:
    langs = langs or wanted_langs()
    forced = os.environ.get("INDICLEARN_OCR", "auto").strip().lower()
    if forced in ("off", "none", "0", "false"):
        return None
    indic = any(x != "en" for x in langs)
    order = ["rapidocr", "tesseract", "easyocr"] if not indic else ["easyocr", "tesseract", "rapidocr"]
    if forced in ("rapidocr", "easyocr", "tesseract"):
        order = [forced]
    for kind in order:
        eng = _build(kind, langs)
        if eng is not None:
            return eng
    return None


def status() -> dict[str, Any]:
    eng = get_engine()
    return {"engine": eng.name if eng else None, "langs": wanted_langs(), "errors": dict(_errors)}


INSTALL_HINT = ("Install a free OCR engine: pip install rapidocr-onnxruntime   (English, light)   or   pip install easyocr   "
                "(also Telugu/Hindi). Then restart the app.")


def ocr_image(img: Any, langs: list[str] | None = None) -> list[OcrLine]:
    """OCR a PIL image. Returns [] when there is no engine or no text."""
    eng = get_engine(langs)
    if eng is None:
        return []
    w, h = img.size
    if w < MIN_SIDE or h < MIN_SIDE:
        return []
    try:
        lines = eng.read(_prep(img))
    except Exception as e:  # never let a bad image break the whole upload
        logger.warning("OCR failed on one image: %s", e)
        return []
    # real text has letters; drop speckle such as "|" or "..."
    return [ln for ln in lines if len(re.findall(r"\w", ln.text)) >= 2]


def ocr_bytes(data: bytes, langs: list[str] | None = None) -> list[OcrLine]:
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:
        logger.info("cannot decode embedded image: %s", e)
        return []
    return ocr_image(img, langs)
