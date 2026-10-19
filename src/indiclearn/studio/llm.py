"""Optional local model backends.  Every one degrades gracefully; the app never *requires* them.

* ``Embedder``   - multilingual-e5 dense vectors (sentence-transformers) for semantic retrieval.
* ``Translator`` - NLLB-200 (local) to render a source passage in the student's chat language.
* ``LocalLLM``   - Ollama (HTTP on localhost) or a transformers pipeline to *rephrase* the grounded
                   answer in the chat language.  Never used without retrieved context.

No hosted API is ever called: Ollama is a local daemon and the others run in-process.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any

import numpy as np

from indiclearn.studio.config import Settings

log = logging.getLogger("indiclearn.llm")

NLLB_CODES = {"en": "eng_Latn", "te": "tel_Telu", "hi": "hin_Deva", "sa": "san_Deva", "ta": "tam_Taml", "kn": "kan_Knda",
              "ml": "mal_Mlym", "mr": "mar_Deva", "bn": "ben_Beng", "gu": "guj_Gujr", "pa": "pan_Guru", "or": "ory_Orya", "ur": "urd_Arab"}
LANG_FULL = {"en": "English", "te": "Telugu", "hi": "Hindi", "sa": "Sanskrit"}


def _importable(mod: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(mod) is not None


class Embedder:
    def __init__(self, model: str, local_only: bool = False) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model_name = model
        self.m = SentenceTransformer(model, local_files_only=local_only)

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        v = self.m.encode(["passage: " + t for t in texts], normalize_embeddings=True, batch_size=32)
        return np.asarray(v, dtype="float32")

    def encode_query(self, text: str) -> np.ndarray:
        return np.asarray(self.m.encode(["query: " + text], normalize_embeddings=True)[0], dtype="float32")


class Translator:
    def __init__(self, model: str, local_only: bool = False) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

        self.tok = AutoTokenizer.from_pretrained(model, local_files_only=local_only)
        self.m = AutoModelForSeq2SeqLM.from_pretrained(model, local_files_only=local_only)

    def translate(self, text: str, src: str, tgt: str) -> str:
        if src == tgt or src not in NLLB_CODES or tgt not in NLLB_CODES:
            return text
        self.tok.src_lang = NLLB_CODES[src]
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=400)
        out = self.m.generate(**enc, forced_bos_token_id=self.tok.convert_tokens_to_ids(NLLB_CODES[tgt]), max_new_tokens=400)
        return self.tok.batch_decode(out, skip_special_tokens=True)[0]


class LocalLLM:
    def __init__(self, backend: str, model: str, url: str) -> None:
        self.backend, self.model, self.url = backend, model, url.rstrip("/")

    def available(self) -> bool:
        if self.backend != "ollama":
            return False
        try:
            with urllib.request.urlopen(self.url + "/api/tags", timeout=1.5) as r:
                return r.status == 200
        except Exception:
            return False

    def rephrase(self, question: str, grounded: str, lang: str) -> str | None:
        prompt = (
            f"You are a patient tutor. Answer the student's question in {LANG_FULL.get(lang, 'English')} using ONLY the notes below. "
            "Keep technical terms and formulas as written. If the notes do not answer it, say so.\n\n"
            f"NOTES:\n{grounded}\n\nQUESTION: {question}\n\nANSWER:"
        )
        try:
            req = urllib.request.Request(
                self.url + "/api/generate", data=json.dumps({"model": self.model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                return (json.loads(r.read()).get("response") or "").strip() or None
        except Exception as e:  # pragma: no cover
            log.warning("LLM rephrase failed: %s", e)
            return None


class Models:
    """Lazy holder; constructing it never raises."""

    def __init__(self, s: Settings) -> None:
        self.s = s
        self.embedder: Embedder | None = None
        self.translator: Translator | None = None
        self.llm: LocalLLM | None = None
        self.notes: list[str] = []
        # hybrid generative engine (hosted -> ollama -> none); see brain.py
        from indiclearn.studio.brain import Brain

        try:
            self.brain = Brain.from_env(s.llm_model, s.ollama_url)
        except Exception as e:  # pragma: no cover - a bad env var must not stop the app
            self.brain = Brain([])
            self.notes.append(f"brain unavailable ({type(e).__name__})")
        try:  # a local LLM on a small GPU needs the VRAM: run the helper models on the CPU instead
            from indiclearn.studio import brain as _b
            from indiclearn.studio import neural as _n

            if any(isinstance(p, _b.OllamaProvider) for p in self.brain.providers) and os.environ.get("INDICLEARN_AUX_DEVICE", "auto") == "auto":
                _n.set_aux_device("cpu" if _n.small_gpu() else "auto")
        except Exception:  # pragma: no cover
            pass
        if not self.brain.available:
            self.notes.append("no LLM configured (set INDICLEARN_GEMINI_API_KEY / INDICLEARN_SARVAM_API_KEY, or run Ollama); using extractive answers")
        emb_model, emb_on = s.embedding_model, s.embedding_backend == "sentence_transformers"
        if s.embedding_backend == "auto":
            # use the multilingual-e5 copy already in models/pretrained (cross-lingual search, no download)
            from indiclearn.studio.neural import models_dir

            local = models_dir() / "pretrained" / "e5_small"
            if (local / "config.json").exists() and _importable("sentence_transformers"):
                emb_model, emb_on = str(local), True
        if emb_on:
            try:
                self.embedder = Embedder(emb_model, s.local_files_only)
            except Exception as e:
                self.notes.append(f"embedder unavailable ({type(e).__name__}); using lexical+phonetic retrieval")
        if s.translator_backend == "auto":
            from indiclearn.studio.neural import models_dir

            local = models_dir() / "pretrained" / "nllb_600m"
            if (local / "config.json").exists() and _importable("transformers"):
                s.translator_backend, s.translator_model = "nllb", str(local)
        if s.translator_backend == "nllb":
            try:
                self.translator = Translator(s.translator_model, s.local_files_only)
            except Exception as e:
                self.notes.append(f"translator unavailable ({type(e).__name__}); showing source passages as written")
        if s.llm_backend == "ollama":
            m = LocalLLM("ollama", s.llm_model, s.ollama_url)
            if m.available():
                self.llm = m
            else:
                self.notes.append("ollama not reachable; using extractive answers")

    def translate_query(self, text: str, src: str, tgt: str) -> str | None:
        """Offline question translation for cross-lingual search: IndicTrans2, else NLLB. None if neither."""
        if not text.strip() or src == tgt:
            return None
        try:
            from indiclearn.studio import neural

            it = neural.load_indictrans()
            if it is not None and it.can(src, tgt):
                return it.translate([text], src, tgt)[0]
        except Exception as e:  # pragma: no cover
            log.warning("query translation (IndicTrans2) failed: %s", e)
        if self.translator is not None and src in NLLB_CODES and tgt in NLLB_CODES:
            try:
                return self.translator.translate(text, src, tgt)
            except Exception as e:  # pragma: no cover
                log.warning("query translation (NLLB) failed: %s", e)
        return None

    def status(self) -> dict[str, Any]:
        return {"embedder": self.embedder.model_name if self.embedder else None, "translator": self.s.translator_model if self.translator else None,
                "llm": f"ollama:{self.llm.model}" if self.llm else None, "brain": self.brain.status(), "notes": self.notes}
