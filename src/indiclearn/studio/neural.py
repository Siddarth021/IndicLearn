"""Optional trained models, loaded lazily with a rule-based fallback.

Nothing here is required. Every loader returns ``None`` when the model folder is missing or
PyTorch is not installed, and every call site keeps working on the hand-written rules it used
before. That is deliberate: a student who has not trained anything still gets a working app,
and a missing or broken model can never take the site down.

Layout (all inside the project folder, created by the phase-2 scripts)::

    models/trained/m1_concept_filter/   is this span a real concept?
    models/trained/m2_describe/         write a one-line description
    models/trained/m3_relations/        prereq | none
    models/trained/m4_quiz/             passage -> question + options
    models/pretrained/whisper_small/    speech -> text, and which language
    models/pretrained/indictrans2_*/    translation (needs: pip install indictranstoolkit; Linux/macOS)
    models/pretrained/mms_tts_tel|hin|eng/   text -> speech (read answers aloud)

Switches (all default to "use it if it is there")::

    INDICLEARN_MODELS_DIR=D:\\Indic_NLP\\models
    INDICLEARN_USE_M1=0        # same for M2, M3, M4, ASR, MT
    INDICLEARN_M1_THRESHOLD=0.5
"""

from __future__ import annotations

import logging
import os
import re
import threading
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_LOCK = threading.Lock()


def models_dir() -> Path:
    env = os.environ.get("INDICLEARN_MODELS_DIR")
    if env:
        return Path(env)
    # src/indiclearn/studio/neural.py -> repo root
    return Path(__file__).resolve().parents[3] / "models"


def _enabled(name: str) -> bool:
    return os.environ.get(f"INDICLEARN_USE_{name.upper()}", "1").strip().lower() not in ("0", "false", "no", "off")


def _folder(*parts: str) -> Path | None:
    p = models_dir().joinpath(*parts)
    return p if (p / "config.json").exists() else None


def _meta(folder: Path) -> dict[str, Any]:
    """training metadata written by training/train.py (threshold, prompt format, base model)."""
    import json

    try:
        return json.loads((folder / "indiclearn.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _torch() -> Any | None:
    try:
        import torch
    except Exception:
        return None
    return torch


_AUX_DEVICE = os.environ.get("INDICLEARN_AUX_DEVICE", "auto").strip().lower()  # auto | cpu | cuda


def set_aux_device(dev: str) -> None:
    global _AUX_DEVICE
    _AUX_DEVICE = dev


def small_gpu(limit_gb: float = 7.5) -> bool:
    """True when the GPU is too small to share with a local LLM (e.g. a 4 GB RTX 3050)."""
    t = _torch()
    try:
        return bool(t and t.cuda.is_available() and t.cuda.get_device_properties(0).total_memory / 2**30 < limit_gb)
    except Exception:
        return False


def _device() -> str:
    t = _torch()
    if t is None:
        return "cpu"
    if os.environ.get("INDICLEARN_FORCE_CPU", "").strip() in ("1", "true") or _AUX_DEVICE == "cpu":
        return "cpu"
    return "cuda" if t.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- M1
@dataclass
class ConceptFilter:
    tok: Any
    model: Any
    threshold: float = 0.5

    def scores(self, spans: list[tuple[str, str]]) -> list[float]:
        """P(real concept) for each (span, context) pair. Never raises."""
        if not spans:
            return []
        t = _torch()
        if t is None:
            return [1.0] * len(spans)
        out: list[float] = []
        try:
            with t.no_grad():
                for i in range(0, len(spans), 32):
                    batch = spans[i : i + 32]
                    enc = self.tok([s for s, _ in batch], [c[:400] for _, c in batch],
                                   truncation=True, max_length=128, padding=True, return_tensors="pt")
                    enc = {k: v.to(self.model.device) for k, v in enc.items()}
                    probs = t.softmax(self.model(**enc).logits, dim=-1)[:, 1]
                    out.extend(float(x) for x in probs.cpu())
        except Exception as e:
            log.warning("concept filter failed, keeping every span: %s", e)
            return [1.0] * len(spans)
        return out

    def keep(self, spans: list[tuple[str, str]]) -> list[bool]:
        return [s >= self.threshold for s in self.scores(spans)]


# --------------------------------------------------------------------------- M2 / M4
@dataclass
class Seq2SeqModel:
    tok: Any
    model: Any
    max_new_tokens: int = 96

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        t = _torch()
        if t is None:
            return ""
        try:
            with t.no_grad():
                enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=512)
                enc = {k: v.to(self.model.device) for k, v in enc.items()}
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens or self.max_new_tokens,
                                          num_beams=4, no_repeat_ngram_size=3)
            return self.tok.decode(gen[0], skip_special_tokens=True).strip()
        except Exception as e:
            log.warning("seq2seq generation failed: %s", e)
            return ""


    def generate_many(self, prompt: str, n: int = 4, max_new_tokens: int | None = None) -> list[str]:
        """n different candidates (sampled) - the caller keeps the first one that passes its checks."""
        t = _torch()
        if t is None:
            return []
        try:
            with t.no_grad():
                enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=512)
                enc = {k: v.to(self.model.device) for k, v in enc.items()}
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens or self.max_new_tokens, do_sample=True, top_p=0.9,
                                          temperature=0.8, num_return_sequences=n, no_repeat_ngram_size=3)
            return [self.tok.decode(g, skip_special_tokens=True).strip() for g in gen]
        except Exception as e:
            log.warning("seq2seq sampling failed: %s", e)
            return []


_LANG_NAME = {"en": "English", "hi": "Hindi", "te": "Telugu", "sa": "Sanskrit"}


def describe(term: str, lang: str = "en", context: str = "") -> str:
    """One-line description for a concept the document never defined. '' if no model."""
    m = load_describer()
    if m is None or not term.strip():
        return ""
    # same prompt as training/build_datasets.py; the model writes English (Flan-T5)
    prompt = f"define: {term} | context: {context[:300]}"
    text = m.generate(prompt)
    if text and lang not in ("en", "") and len(text) >= 25:
        tr = load_indictrans()
        if tr is not None:
            try:
                text = tr.translate([text], "en", lang)[0] or text
            except Exception:  # pragma: no cover
                pass
    # A description that only echoes the term is worse than none at all.
    return "" if len(text) < 25 or text.lower().strip(" .") == term.lower().strip(" .") else text


# --------------------------------------------------------------------------- M3
@dataclass
class RelationModel:
    tok: Any
    model: Any
    labels: list[str]

    def classify(self, a: str, b: str, context: str = "") -> tuple[str, float]:
        t = _torch()
        if t is None:
            return "none", 0.0
        try:
            with t.no_grad():
                enc = self.tok(f"{a} [SEP] {b}", context[:300], truncation=True,
                               max_length=128, return_tensors="pt")
                enc = {k: v.to(self.model.device) for k, v in enc.items()}
                probs = t.softmax(self.model(**enc).logits, dim=-1)[0]
            i = int(probs.argmax())
            return self.labels[i] if i < len(self.labels) else "none", float(probs[i])
        except Exception as e:
            log.warning("relation model failed: %s", e)
            return "none", 0.0

    def classify_batch(self, pairs: list[tuple[str, str, str]]) -> list[tuple[str, float]]:
        """[(a, b, context)] -> [(label, prob)]; label is about "a before b". Never raises."""
        t = _torch()
        if t is None or not pairs:
            return [("none", 0.0)] * len(pairs)
        out: list[tuple[str, float]] = []
        try:
            with t.no_grad():
                for i in range(0, len(pairs), 32):
                    b = pairs[i : i + 32]
                    enc = self.tok([f"{a} [SEP] {c}" for a, c, _ in b], [(x or "-")[:300] for _, _, x in b],
                                   truncation=True, max_length=128, padding=True, return_tensors="pt")
                    enc = {k: v.to(self.model.device) for k, v in enc.items()}
                    probs = t.softmax(self.model(**enc).logits, dim=-1)
                    for row in probs.cpu():
                        j = int(row.argmax())
                        out.append((self.labels[j] if j < len(self.labels) else "none", float(row[j])))
        except Exception as e:
            log.warning("relation model failed: %s", e)
            return [("none", 0.0)] * len(pairs)
        return out


# --------------------------------------------------------------------------- speech
@dataclass
class Recogniser:
    pipe: Any
    fixed_lang: str = ""  # a language-specific fine-tune always reports its language

    def transcribe(self, audio_path: str) -> tuple[str, str]:
        """Return (text, language code). ('', '') when it cannot be done."""
        try:
            out = self.pipe(audio_path, return_timestamps=False)
            if self.fixed_lang:
                text = (out.get("text") or "").strip() if isinstance(out, dict) else str(out).strip()
                return text, self.fixed_lang
            text = (out.get("text") or "").strip() if isinstance(out, dict) else str(out).strip()
            lang = ""
            chunks = out.get("chunks") if isinstance(out, dict) else None
            if isinstance(chunks, list) and chunks:
                lang = str(chunks[0].get("language") or "")
            return text, (lang or "")[:2]
        except Exception as e:
            log.warning("speech recognition failed: %s", e)
            return "", ""


# --------------------------------------------------------------------------- loaders
@lru_cache(maxsize=1)
def load_concept_filter() -> ConceptFilter | None:
    if not _enabled("m1"):
        return None
    folder = _folder("trained", "m1_concept_filter")
    if folder is None:
        return None
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        with _LOCK:
            tok = AutoTokenizer.from_pretrained(str(folder))
            model = AutoModelForSequenceClassification.from_pretrained(str(folder), dtype=_torch().float32).to(_device()).eval()  # weights ship in fp16
        thr = float(os.environ.get("INDICLEARN_M1_THRESHOLD") or _meta(folder).get("threshold", 0.5))
        log.info("M1 concept filter loaded from %s (threshold %.2f)", folder, thr)
        return ConceptFilter(tok, model, thr)
    except Exception as e:
        log.warning("could not load M1 (%s) - keeping the rule-based concept miner", e)
        return None


def _load_seq2seq(sub: str, flag: str, max_new: int) -> Seq2SeqModel | None:
    if not _enabled(flag):
        return None
    folder = _folder("trained", sub)
    if folder is None:
        return None
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        with _LOCK:
            tok = AutoTokenizer.from_pretrained(str(folder))
            model = AutoModelForSeq2SeqLM.from_pretrained(str(folder), dtype=_torch().float32).to(_device()).eval()  # weights ship in fp16
        log.info("%s loaded from %s", flag.upper(), folder)
        return Seq2SeqModel(tok, model, max_new)
    except Exception as e:
        log.warning("could not load %s (%s) - using the rule-based path", flag.upper(), e)
        return None


@lru_cache(maxsize=1)
def load_describer() -> Seq2SeqModel | None:
    return _load_seq2seq("m2_describe", "m2", 96)


@lru_cache(maxsize=1)
def load_quizzer() -> Seq2SeqModel | None:
    return _load_seq2seq("m4_quiz", "m4", 128)


@lru_cache(maxsize=1)
def load_relations() -> RelationModel | None:
    if not _enabled("m3"):
        return None
    folder = _folder("trained", "m3_relations")
    if folder is None:
        return None
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        with _LOCK:
            tok = AutoTokenizer.from_pretrained(str(folder))
            model = AutoModelForSequenceClassification.from_pretrained(str(folder), dtype=_torch().float32).to(_device()).eval()  # weights ship in fp16
        id2 = getattr(model.config, "id2label", None) or {}
        labels = [str(id2[k]) for k in sorted(id2, key=lambda x: int(x))] if id2 else ["none", "prereq"]
        log.info("M3 relations loaded from %s (%s)", folder, labels)
        return RelationModel(tok, model, labels)
    except Exception as e:
        log.warning("could not load M3 (%s) - using co-occurrence rules", e)
        return None


def _hf_pipeline(task: str, model: str, device: int) -> Any:
    from transformers import pipeline

    return pipeline(task, model=model, device=device)


# language-specific speech models, best first (all Apache-2.0, fine-tuned Whisper)
ASR_FOLDERS = {"te": ("whisper_te_medium", "whisper_te_small")}


@lru_cache(maxsize=4)
def load_recogniser(lang: str = "") -> Recogniser | None:
    """Speech -> text. A Telugu-fine-tuned Whisper for Telugu speech, generic Whisper otherwise."""
    if not _enabled("asr"):
        return None
    base = (lang or "").split("-")[0]
    picks = [(f, base) for f in ASR_FOLDERS.get(base, ())] + [("whisper_small", "")]
    for name, fixed in picks:
        folder = _folder("pretrained", name)
        if folder is None:
            continue
        try:
            with _LOCK:
                pipe = _hf_pipeline("automatic-speech-recognition", str(folder), 0 if _device() == "cuda" else -1)
                if fixed:
                    pipe.model.config.forced_decoder_ids = None
                    pipe.model.generation_config.forced_decoder_ids = None
                    pipe.model.generation_config.language = fixed
                    pipe.model.generation_config.task = "transcribe"
            log.info("speech recognition (%s) loaded from %s", fixed or "auto-language", folder)
            return Recogniser(pipe, fixed)
        except Exception as e:
            log.warning("could not load speech model %s (%s)", name, e)
    return None


# --------------------------------------------------------------------------- translation (IndicTrans2)
FLORES = {"en": "eng_Latn", "te": "tel_Telu", "hi": "hin_Deva", "ta": "tam_Taml", "kn": "kan_Knda", "ml": "mal_Mlym",
          "mr": "mar_Deva", "bn": "ben_Beng", "gu": "guj_Gujr", "pa": "pan_Guru", "or": "ory_Orya", "ur": "urd_Arab",
          "sa": "san_Deva"}


# --- a Windows-friendly stand-in for IndicTransToolkit's IndicProcessor -----------------------------
# IndicTrans2 was trained on Indic text transliterated to Devanagari ("script unification") with
# "<src> <tgt>" tags in front. The official toolkit does that with Cython code its authors do not
# support on Windows. The steps are simple enough to reproduce: all Brahmic Unicode blocks share one
# 128-code-point layout, so moving between them is an offset (the same trick indic-nlp-library uses).
_BRAHMIC = {"Deva": 0x0900, "Beng": 0x0980, "Guru": 0x0A00, "Gujr": 0x0A80, "Orya": 0x0B00, "Taml": 0x0B80,
            "Telu": 0x0C00, "Knda": 0x0C80, "Mlym": 0x0D00}
_KEEP_OFFSETS = {0x64, 0x65}  # danda / double danda live only in the Devanagari block
_SENT_SPLIT = re.compile(r"(?<=[.!?।॥])\s+")


def brahmic_convert(text: str, src_script: str, tgt_script: str) -> str:
    a, b = _BRAHMIC.get(src_script), _BRAHMIC.get(tgt_script)
    if a is None or b is None or a == b:
        return text
    out = []
    for ch in text:
        o = ord(ch) - a
        out.append(chr(b + o) if 0 <= o < 0x80 and o not in _KEEP_OFFSETS else ch)
    return "".join(out)


class LightIndicProcessor:
    """Pre/post-processing for IndicTrans2 without IndicTransToolkit (tags, script unification, spacing)."""

    def __init__(self) -> None:
        try:
            from sacremoses import MosesDetokenizer, MosesTokenizer  # optional, pure Python

            self._tok, self._detok = MosesTokenizer(lang="en"), MosesDetokenizer(lang="en")
        except Exception:
            self._tok = self._detok = None

    @staticmethod
    def _script(flores: str) -> str:
        return flores.split("_", 1)[1]

    def _en_tok(self, s: str) -> str:
        if self._tok is not None:
            return self._tok.tokenize(s, return_str=True, escape=False)
        return re.sub(r"\s+", " ", re.sub(r"([.,!?;:()\[\]\"])", r" \1 ", s)).strip()

    def preprocess_batch(self, sents: list[str], src_lang: str, tgt_lang: str, **_: Any) -> list[str]:
        out = []
        for s in sents:
            s = unicodedata.normalize("NFC", re.sub(r"\s+", " ", s)).strip()
            if src_lang == "eng_Latn":
                body = self._en_tok(s)
            else:
                body = re.sub(r"\s+", " ", re.sub(r"([.,!?;:()\[\]\"।])", r" \1 ", s)).strip()
                body = brahmic_convert(body, self._script(src_lang), "Deva")
            out.append(f"{src_lang} {tgt_lang} {body}")
        return out

    def postprocess_batch(self, outs: list[str], lang: str = "eng_Latn", **_: Any) -> list[str]:
        res = []
        for s in outs:
            if lang == "eng_Latn":
                s = self._detok.detokenize(s.split(" ")) if self._detok is not None else re.sub(r"\s+([.,!?;:)\]])", r"\1", s).replace("( ", "(")
            else:
                s = brahmic_convert(s, "Deva", self._script(lang))
                s = re.sub(r"\s+([.,!?;:)\]।])", r"\1", s).replace("( ", "(")
            res.append(s.strip())
        return res


@dataclass
class IndicTranslator:
    """AI4Bharat IndicTrans2 (distilled 200M), local.  Indic <-> Indic goes through English."""

    en_indic: tuple[Any, Any] | None
    indic_en: tuple[Any, Any] | None
    ip: Any

    def can(self, src: str, tgt: str) -> bool:
        if src not in FLORES or tgt not in FLORES or src == tgt:
            return False
        if src == "en":
            return self.en_indic is not None
        if tgt == "en":
            return self.indic_en is not None
        return self.en_indic is not None and self.indic_en is not None

    def _run(self, pair: tuple[Any, Any], texts: list[str], src: str, tgt: str) -> list[str]:
        t = _torch()
        tok, model = pair
        batch = self.ip.preprocess_batch(texts, src_lang=FLORES[src], tgt_lang=FLORES[tgt])
        enc = tok(batch, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with t.no_grad():
            gen = model.generate(**enc, use_cache=True, min_length=0, max_length=256, num_beams=4, num_return_sequences=1)
        out = tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return list(self.ip.postprocess_batch(out, lang=FLORES[tgt]))

    def _pair(self, texts: list[str], src: str, tgt: str) -> list[str]:
        if src == "en":
            return self._batched(self.en_indic, texts, src, tgt)  # type: ignore[arg-type]
        if tgt == "en":
            return self._batched(self.indic_en, texts, src, tgt)  # type: ignore[arg-type]
        mid = self._batched(self.indic_en, texts, src, "en")  # type: ignore[arg-type]
        return self._batched(self.en_indic, mid, "en", tgt)  # type: ignore[arg-type]

    def _batched(self, pair: tuple[Any, Any], texts: list[str], src: str, tgt: str) -> list[str]:
        out: list[str] = []
        for i in range(0, len(texts), 8):
            out += self._run(pair, texts[i : i + 8], src, tgt)
        return out

    def translate(self, texts: list[str], src: str, tgt: str, strict: bool = False) -> list[str]:
        """Translate paragraphs; each is split into sentences (the model is trained on sentences).

        strict=True raises on failure instead of silently returning the untranslated text."""
        if not texts or not self.can(src, tgt):
            if strict:
                raise RuntimeError(f"IndicTrans2 has no model for {src} -> {tgt} (download both indictrans2 folders)")
            return texts
        try:
            pieces: list[str] = []
            spans: list[tuple[int, int]] = []
            for t_ in texts:
                sents = [x for x in _SENT_SPLIT.split(t_.strip()) if x.strip()] or [t_]
                spans.append((len(pieces), len(pieces) + len(sents)))
                pieces += sents
            done = self._pair(pieces, src, tgt)
            return [" ".join(done[a:b]) for a, b in spans]
        except Exception as e:
            log.warning("IndicTrans2 failed: %s", e)
            LOAD_ERRORS["indictrans2"] = f"translation failed: {str(e)[:150]}"
            if strict:
                raise
            return texts


LOAD_ERRORS: dict[str, str] = {}  # model name -> why it could not be used (shown by /api/info and the UI)


@lru_cache(maxsize=1)
def load_indictrans() -> IndicTranslator | None:
    """Needs the two gated IndicTrans2 folders and ``pip install IndicTransToolkit``."""
    if not _enabled("mt"):
        return None
    a, b = _folder("pretrained", "indictrans2_en_indic"), _folder("pretrained", "indictrans2_indic_en")
    if a is None and b is None:
        LOAD_ERRORS["indictrans2"] = "model folders not downloaded - run setup_offline.bat (accept the licence for ai4bharat/indictrans2-* on huggingface.co first)"
        return None
    if _torch() is None:
        LOAD_ERRORS["indictrans2"] = "PyTorch is not installed in this Python environment - run: pip install torch transformers sentencepiece sacremoses"
        return None
    try:
        IndicProcessor = None
        for mod in ("IndicTransToolkit", "IndicTransToolkit.IndicTransToolkit", "IndicTransToolkit.processor"):
            try:
                IndicProcessor = __import__(mod, fromlist=["IndicProcessor"]).IndicProcessor
                break
            except (ImportError, AttributeError):
                continue
        if IndicProcessor is None:
            raise ImportError("IndicProcessor")
    except ImportError:
        # The official toolkit does not support Windows; our light processor does the same core steps.
        log.info("IndicTransToolkit not installed - using the built-in light processor for IndicTrans2")
        IndicProcessor = None
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        def ld(folder: Path | None) -> tuple[Any, Any] | None:
            if folder is None:
                return None
            tok = AutoTokenizer.from_pretrained(str(folder), trust_remote_code=True)
            m = AutoModelForSeq2SeqLM.from_pretrained(str(folder), trust_remote_code=True).to(_device()).eval()
            return tok, m

        with _LOCK:
            tr = IndicTranslator(ld(a), ld(b), IndicProcessor(inference=True) if IndicProcessor else LightIndicProcessor())
        log.info("IndicTrans2 loaded (en->indic: %s, indic->en: %s)", tr.en_indic is not None, tr.indic_en is not None)
        return tr
    except Exception as e:
        log.warning("could not load IndicTrans2 (%s)", e)
        LOAD_ERRORS["indictrans2"] = f"could not load: {str(e)[:160]}"
        return None


# --------------------------------------------------------------------------- text to speech (MMS-TTS)
TTS_FOLDERS = {"te": "mms_tts_tel", "hi": "mms_tts_hin", "en": "mms_tts_eng"}


@dataclass
class Speaker:
    tok: Any
    model: Any

    def wav(self, text: str) -> bytes:
        """16-bit mono WAV bytes; b'' on failure."""
        import io
        import wave

        t = _torch()
        if t is None or not text.strip():
            return b""
        try:
            with t.no_grad():
                enc = self.tok(text, return_tensors="pt")
                enc = {k: v.to(self.model.device) for k, v in enc.items()}
                wave_t = self.model(**enc).waveform[0].float().cpu().clamp(-1, 1)
            pcm = (wave_t.numpy() * 32767).astype("<i2").tobytes()
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(int(getattr(self.model.config, "sampling_rate", 16000)))
                w.writeframes(pcm)
            return buf.getvalue()
        except Exception as e:
            log.warning("TTS failed: %s", e)
            return b""


# Indic Parler-TTS (AI4Bharat, Apache-2.0): natural voices for 20+ Indian languages, fine for a commercial
# product. Gated on Hugging Face (accept the terms once) and needs: pip install git+https://github.com/huggingface/parler-tts.git
PARLER_LANGS = {"te", "hi", "en", "ta", "kn", "ml", "mr", "bn", "gu", "pa", "or", "ur", "sa", "as", "ne"}
PARLER_VOICE = {"te": "Lalitha speaks at a moderate pace with a clear, close-sounding recording and very little background noise."}
PARLER_DEFAULT_VOICE = "A female speaker delivers clear, expressive speech at a moderate pace, with a close recording and very little background noise."


@dataclass
class ParlerSpeaker:
    model: Any
    prompt_tok: Any
    desc_tok: Any

    def wav(self, text: str, lang: str = "te") -> bytes:
        import io
        import wave

        t = _torch()
        if t is None or not text.strip():
            return b""
        try:
            dev = self.model.device
            d = self.desc_tok(PARLER_VOICE.get(lang, PARLER_DEFAULT_VOICE), return_tensors="pt").to(dev)
            p = self.prompt_tok(text, return_tensors="pt").to(dev)
            with t.no_grad():
                audio = self.model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                            prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
            pcm = (audio.float().cpu().numpy().squeeze().clip(-1, 1) * 32767).astype("<i2").tobytes()
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(int(self.model.config.sampling_rate))
                w.writeframes(pcm)
            return buf.getvalue()
        except Exception as e:
            log.warning("Parler TTS failed: %s", e)
            return b""


@lru_cache(maxsize=1)
def load_parler() -> ParlerSpeaker | None:
    if not _enabled("tts"):
        return None
    folder = _folder("pretrained", "indic_parler_tts")
    if folder is None:
        return None
    try:
        from parler_tts import ParlerTTSForConditionalGeneration  # type: ignore
        from transformers import AutoTokenizer

        with _LOCK:
            model = ParlerTTSForConditionalGeneration.from_pretrained(str(folder)).to(_device()).eval()
            prompt_tok = AutoTokenizer.from_pretrained(str(folder))
            desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
        log.info("Indic Parler-TTS loaded from %s", folder)
        return ParlerSpeaker(model, prompt_tok, desc_tok)
    except Exception as e:
        log.info("Indic Parler-TTS not usable (%s) - falling back to MMS-TTS", e)
        return None


@lru_cache(maxsize=4)
def load_speaker(lang: str) -> Speaker | None:
    if not _enabled("tts") or lang not in TTS_FOLDERS:
        return None
    folder = _folder("pretrained", TTS_FOLDERS[lang])
    if folder is None:
        return None
    try:
        from transformers import AutoTokenizer, VitsModel

        with _LOCK:
            tok = AutoTokenizer.from_pretrained(str(folder))
            model = VitsModel.from_pretrained(str(folder)).to(_device()).eval()
        log.info("TTS (%s) loaded from %s", lang, folder)
        return Speaker(tok, model)
    except Exception as e:
        log.warning("could not load TTS for %s (%s)", lang, e)
        return None


def status() -> dict[str, Any]:
    """What is actually loadable right now - shown on the /api/info endpoint."""
    root = models_dir()
    return {
        "models_dir": str(root),
        "m1_concept_filter": bool(_folder("trained", "m1_concept_filter")) and _enabled("m1"),
        "m2_describe": bool(_folder("trained", "m2_describe")) and _enabled("m2"),
        "m3_relations": bool(_folder("trained", "m3_relations")) and _enabled("m3"),
        "m4_quiz": bool(_folder("trained", "m4_quiz")) and _enabled("m4"),
        "asr": bool(_folder("pretrained", "whisper_small")) and _enabled("asr"),
        "asr_te": next((f for f in ASR_FOLDERS["te"] if _folder("pretrained", f)), None) if _enabled("asr") else None,
        "parler_tts": bool(_folder("pretrained", "indic_parler_tts")) and _enabled("tts"),
        "indictrans2": bool(_folder("pretrained", "indictrans2_en_indic") or _folder("pretrained", "indictrans2_indic_en")) and _enabled("mt"),
        "tts": [k for k, f in TTS_FOLDERS.items() if _folder("pretrained", f)] if _enabled("tts") else [],
        "torch": _torch() is not None,
        "device": _device(),
        "load_errors": dict(LOAD_ERRORS),
    }


def reset_cache() -> None:
    """Forget everything that was loaded - used by the tests and after training a model."""
    for fn in (load_concept_filter, load_describer, load_quizzer, load_relations, load_recogniser, load_indictrans, load_speaker, load_parler):
        fn.cache_clear()
