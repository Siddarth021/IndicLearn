"""The hybrid "brain": one interface over hosted LLMs, a local Ollama model and the offline path.

Why this exists
---------------
The extractive tutor can only *quote* the notes.  Students asked for something else: ask in
Telugu (or Romanized Telugu) about an English chapter and get a real, detailed explanation in
Telugu, with the reasoning shown.  That needs a generative model that is good at Indic languages.

Order of preference (``INDICLEARN_BRAIN``, default ``auto``)::

    hosted providers that have an API key  ->  local Ollama (if reachable)  ->  extractive (no LLM)

Every call walks that chain.  A provider that fails is skipped for ``COOLDOWN_S`` seconds, so a
dead network costs one timeout, not one per request.  Nothing here raises to the caller: every
public method returns ``None`` when no provider could answer, and the caller falls back to the
offline, extractive behaviour that existed before.

Providers (all plain HTTPS via ``urllib`` - no extra dependencies)
------------------------------------------------------------------
* ``gemini``    Google Gemini API   (``INDICLEARN_GEMINI_API_KEY`` or ``GEMINI_API_KEY``; more accounts:
                ``INDICLEARN_GEMINI_API_KEY_2`` ... ``_9``, or several keys separated by commas)
* ``sarvam``    Sarvam AI, Indic-first, OpenAI-compatible (``INDICLEARN_SARVAM_API_KEY``)
* ``openai``    any OpenAI-compatible server: OpenAI, OpenRouter, Groq, vLLM, LM Studio ...
                (``INDICLEARN_OPENAI_BASE_URL`` + ``INDICLEARN_OPENAI_API_KEY`` + ``INDICLEARN_OPENAI_MODEL``)
* ``anthropic`` Claude API          (``INDICLEARN_ANTHROPIC_API_KEY``)
* ``ollama``    local daemon        (``INDICLEARN_OLLAMA_URL``, model ``INDICLEARN_LLM_MODEL``)

Privacy: hosted providers receive the question and the retrieved passages of the student's
notes.  Set ``INDICLEARN_BRAIN=ollama`` (or ``off``) to keep everything on the machine.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from collections.abc import Callable, Iterator
from typing import Any

log = logging.getLogger("indiclearn.brain")

COOLDOWN_S = 60.0
QUOTA_COOLDOWN_S = 1800.0  # a used-up DAILY free quota will not come back in a minute
DEFAULT_GEMINI_MODELS = "gemini-3.8-flash,gemini-3.5-flash,gemini-flash-latest,gemini-2.5-flash"

# Languages a reply can be *converted* into (per-reply dropdown).  The tutor itself chats in
# SUPPORTED_LANGS; any of these is fine as a translation target for an LLM or IndicTrans2.
TARGET_LANGS: dict[str, tuple[str, str]] = {
    # code: (English name, native name)
    "en": ("English", "English"),
    "te": ("Telugu", "తెలుగు"),
    "hi": ("Hindi", "हिन्दी"),
    "ta": ("Tamil", "தமிழ்"),
    "kn": ("Kannada", "ಕನ್ನಡ"),
    "ml": ("Malayalam", "മലയാളം"),
    "mr": ("Marathi", "मराठी"),
    "bn": ("Bengali", "বাংলা"),
    "gu": ("Gujarati", "ગુજરાતી"),
    "pa": ("Punjabi", "ਪੰਜਾਬੀ"),
    "or": ("Odia", "ଓଡ଼ିଆ"),
    "ur": ("Urdu", "اردو"),
    "sa": ("Sanskrit", "संस्कृतम्"),
    "te-Latn": ("Telugu in English letters (Romanized)", "Telugu (English letters)"),
    "hi-Latn": ("Hindi in English letters (Hinglish)", "Hinglish"),
}


def lang_name(code: str) -> str:
    return TARGET_LANGS.get(code, (code, code))[0]


_NATIVE_SCRIPT = {"te": "Telugu script (తెలుగు లిపి)", "hi": "Devanagari script", "sa": "Devanagari script", "mr": "Devanagari script",
                  "ta": "Tamil script", "kn": "Kannada script", "ml": "Malayalam script", "bn": "Bengali script",
                  "gu": "Gujarati script", "pa": "Gurmukhi script", "or": "Odia script", "ur": "Urdu (Perso-Arabic) script"}


def script_rule(code: str) -> str:
    """' in Telugu script (...), even if the student typed in English letters' - stops models mirroring Romanized input."""
    s = _NATIVE_SCRIPT.get(code)
    return f", written in {s} even if the student typed in English letters" if s else ""


_OPT_PREFIX = re.compile(r"^\s*(?:\(?[A-Da-d1-4]\s*[\).:\-]|[A-Da-d]\s+-)\s+")
_PLACEHOLDER = re.compile(r"^\s*(?:[A-Da-d]|option\s*[A-Da-d1-4]|\.\.\.|…)?\s*$", re.IGNORECASE)
_EMPTY_WORDS = {"", "empty string", "none", "n/a", "na", "null", "-", "nothing", "\"\""}


_OPT_REF = re.compile(r"\(?\b[Oo]ption\s*\(?([0-9]|[A-Ea-e])\b\)?\)?")


def fix_option_refs(why: str, opts: list[str]) -> str:
    """Options are shuffled before a student sees them, so "Option 0 refers to the sill" would point at the wrong
    choice. Replace such references with the option's own words."""
    if not why or not _OPT_REF.search(why):
        return why
    nums = [m.group(1) for m in _OPT_REF.finditer(why) if m.group(1).isdigit()]
    zero_based = "0" in nums

    def sub(m: re.Match) -> str:
        k = m.group(1)
        i = (int(k) if zero_based else int(k) - 1) if k.isdigit() else "ABCDE".index(k.upper())
        if not 0 <= i < len(opts):
            return "another option"
        o = opts[i]
        return f"“{o if len(o) <= 60 else o[:57] + '…'}”"

    return _OPT_REF.sub(sub, why)


def clean_mcq(q: Any, min_opts: int = 3) -> dict[str, Any] | None:
    """Validate one model-written MCQ: real (not 'A'/'B') distinct options, an in-range answer.

    Accepts "answer" as an index, a numeric string, a letter ("B") or the option text itself."""
    if not isinstance(q, dict):
        return None
    stem = str(q.get("q") or q.get("question") or "").strip()
    raw = q.get("options")
    if len(stem) < 8 or not isinstance(raw, list):
        return None
    opts = [_OPT_PREFIX.sub("", str(o)).strip() for o in raw]
    if len(opts) < min_opts or len(opts) > 5 or any(_PLACEHOLDER.match(o) for o in opts):
        return None
    if len({o.lower() for o in opts}) != len(opts):
        return None
    a = q.get("answer")
    idx: int | None = None
    if isinstance(a, bool):
        idx = None
    elif isinstance(a, int):
        idx = a
    elif isinstance(a, str):
        t = a.strip()
        if t.isdigit():
            idx = int(t)
        elif len(t) == 1 and t.upper() in "ABCDE":
            idx = "ABCDE".index(t.upper())
        else:
            low = _OPT_PREFIX.sub("", t).strip().lower()
            idx = next((i for i, o in enumerate(opts) if o.lower() == low), None)
    if idx is None or not 0 <= idx < len(opts):
        return None
    why = fix_option_refs(str(q.get("why") or q.get("explanation") or "").strip(), opts)
    return {**q, "q": stem, "options": opts, "answer": idx, "why": why}


def empty_if_placeholder(x: Any) -> str:
    s = str(x or "").strip()
    return "" if s.lower().strip(" .") in _EMPTY_WORDS else s


# --------------------------------------------------------------------------- JSON helpers
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


_THINK = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.IGNORECASE | re.DOTALL)


def strip_think(text: str) -> str:
    """Reasoning models (Sarvam, Qwen, DeepSeek...) may put their scratch-work in <think> tags."""
    if not text:
        return ""
    t = _THINK.sub("", text)
    low = t.lower()
    if "</think>" in low:  # opening tag was cut off
        t = t[low.rindex("</think>") + len("</think>"):]
    elif "<think>" in low:  # never closed: the answer did not fit in the token budget
        t = t[: low.index("<think>")]
    return t.strip()


def parse_json(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a model reply (tolerates ```json fences and chatter)."""
    if not text:
        return None
    t = _FENCE.sub("", strip_think(text))
    try:
        v = json.loads(t)
        return v if isinstance(v, dict) else None
    except Exception:
        pass
    start = t.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    v = json.loads(t[start : i + 1])
                    return v if isinstance(v, dict) else None
                except Exception:
                    return None
    return None


def _http_json(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:3000]
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code}: {detail}") from e


def _friendly_error(e: Exception) -> str:
    m = str(e)
    low = m.lower()
    if "http 401" in low or "http 403" in low:
        return "the API key was rejected (wrong key, expired, or no access to this model) - " + m[:120]
    if "http 404" in low:
        return "model or URL not found - check INDICLEARN_*_MODEL - " + m[:120]
    if "http 429" in low:
        return "rate limit or free quota used up - wait a minute or use another provider - " + m[:100]
    if "timed out" in low or "timeout" in low:
        return "timed out - raise INDICLEARN_BRAIN_TIMEOUT or check the internet"
    if "getaddrinfo" in low or "urlopen error" in low or "connection" in low:
        return "no internet / cannot reach the provider"
    return m[:200]


def _sse_lines(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> Iterator[str]:
    """POST and yield each streamed line (SSE 'data: ...' or NDJSON), decoded."""
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json", **headers})
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code}: {detail}") from e
    with r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if line:
                yield line


_ANSWER_KEY = re.compile(r'"answer"\s*:\s*"')
_ESC = {"n": "\n", "t": "\t", '"': '"', "\\": "\\", "/": "/", "r": "", "b": "", "f": ""}


def partial_answer(raw: str) -> str:
    """The (possibly unfinished) value of "answer" in a JSON reply that is still being streamed."""
    m = _ANSWER_KEY.search(raw or "")
    if not m:
        return ""
    i, out = m.end(), []
    while i < len(raw):
        c = raw[i]
        if c == '"':
            break
        if c == "\\":
            if i + 1 >= len(raw):
                break
            n = raw[i + 1]
            if n == "u":
                if i + 6 > len(raw):
                    break
                try:
                    out.append(chr(int(raw[i + 2 : i + 6], 16)))
                except ValueError:
                    pass
                i += 6
                continue
            out.append(_ESC.get(n, n))
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def gemini_keys() -> list[str]:
    """All Gemini keys, in order: INDICLEARN_GEMINI_API_KEY (may hold several, comma-separated), then _2 ... _9."""
    env = os.environ.get
    raw = [env("INDICLEARN_GEMINI_API_KEY") or env("GEMINI_API_KEY") or env("GOOGLE_API_KEY") or ""]
    raw += [env(f"INDICLEARN_GEMINI_API_KEY_{i}") or "" for i in range(2, 10)]
    keys: list[str] = []
    for chunk in raw:
        for k in chunk.split(","):
            k = k.strip().strip('"').strip("'")
            if k and k not in keys:
                keys.append(k)
    return keys


# --------------------------------------------------------------------------- providers
@dataclass
class Provider:
    name: str
    model: str
    timeout: float = 60.0
    down_until: float = 0.0
    audio: bool = False  # can transcribe speech
    local: bool = False  # runs on this computer (smaller prompts, longer timeout)
    last_error: str = ""  # why the most recent call failed ("" when it worked) - shown to the user

    def ok(self) -> bool:
        return time.time() >= self.down_until

    def complete(self, system: str, user: str, json_mode: bool = True, max_tokens: int = 2048) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def stream(self, system: str, user: str, json_mode: bool = True, max_tokens: int = 2048) -> Iterator[str]:
        """Yield the reply in pieces. Default: one piece (providers override with real streaming)."""
        yield self.complete(system, user, json_mode, max_tokens)

    def transcribe(self, audio: bytes, mime: str) -> str:  # pragma: no cover - only some providers
        raise NotImplementedError


@dataclass
class GeminiProvider(Provider):
    key: str = ""
    account: int = 1  # which Google account's key (each account has its own free quota)
    base: str = "https://generativelanguage.googleapis.com/v1beta"
    audio: bool = True

    def _call(self, parts: list[dict[str, Any]], system: str, json_mode: bool, max_tokens: int) -> str:
        cfg: dict[str, Any] = {"maxOutputTokens": max_tokens, "thinkingConfig": {"thinkingLevel": os.environ.get("INDICLEARN_GEMINI_THINKING", "low")}}
        if json_mode:
            cfg["responseMimeType"] = "application/json"
        body: dict[str, Any] = {"contents": [{"role": "user", "parts": parts}], "generationConfig": cfg}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        url = f"{self.base}/models/{self.model}:generateContent"
        try:
            data = _http_json(url, body, {"x-goog-api-key": self.key}, self.timeout)
        except RuntimeError as e:
            # Older / other Gemini models reject thinkingConfig - retry once without it.
            if "thinking" in str(e).lower():
                cfg.pop("thinkingConfig", None)
                data = _http_json(url, body, {"x-goog-api-key": self.key}, self.timeout)
            else:
                raise
        cands = data.get("candidates") or []
        if not cands:
            raise RuntimeError(f"no candidates ({str(data.get('promptFeedback', ''))[:120]})")
        return "".join(p.get("text", "") for p in (cands[0].get("content") or {}).get("parts", []) if not p.get("thought"))

    def complete(self, system: str, user: str, json_mode: bool = True, max_tokens: int = 2048) -> str:
        return self._call([{"text": user}], system, json_mode, max_tokens)

    def stream(self, system: str, user: str, json_mode: bool = True, max_tokens: int = 2048) -> Iterator[str]:
        cfg: dict[str, Any] = {"maxOutputTokens": max_tokens, "thinkingConfig": {"thinkingLevel": os.environ.get("INDICLEARN_GEMINI_THINKING", "low")}}
        if json_mode:
            cfg["responseMimeType"] = "application/json"
        body: dict[str, Any] = {"contents": [{"role": "user", "parts": [{"text": user}]}], "generationConfig": cfg}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        url = f"{self.base}/models/{self.model}:streamGenerateContent?alt=sse"
        for line in _sse_lines(url, body, {"x-goog-api-key": self.key}, self.timeout):
            if not line.startswith("data:"):
                continue
            try:
                d = json.loads(line[5:].strip())
            except ValueError:
                continue
            for c in d.get("candidates") or []:
                for p in (c.get("content") or {}).get("parts", []):
                    if p.get("text") and not p.get("thought"):
                        yield p["text"]

    def transcribe(self, audio: bytes, mime: str) -> str:
        parts = [{"inlineData": {"mimeType": mime, "data": base64.b64encode(audio).decode()}},
                 {"text": "Transcribe this student's spoken question exactly, in the script of the language spoken "
                          "(Telugu in Telugu script, Hindi in Devanagari, English in Latin). Keep English technical "
                          "terms in English. Reply with JSON: {\"text\": ..., \"lang\": \"te|hi|en|sa|...\"}."}]
        return self._call(parts, "", True, 512)


@dataclass
class OpenAICompatProvider(Provider):
    key: str = ""
    base: str = "https://api.openai.com/v1"
    extra_headers: dict[str, str] = field(default_factory=dict)
    send_json_format: bool = True
    extra_body: dict[str, Any] = field(default_factory=dict)  # provider-specific switches

    def complete(self, system: str, user: str, json_mode: bool = True, max_tokens: int = 2048) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        body: dict[str, Any] = {"model": self.model, "messages": msgs, "max_tokens": max_tokens, **self.extra_body}
        if json_mode and self.send_json_format:
            body["response_format"] = {"type": "json_object"}
        headers = {**self.extra_headers}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        url = self.base.rstrip("/") + "/chat/completions"
        try:
            data = _http_json(url, body, headers, self.timeout)
        except RuntimeError as e:
            # Not every OpenAI-compatible server (Sarvam, LM Studio...) knows response_format: retry plainly.
            if json_mode and "response_format" in body and re.search(r"HTTP (400|422)", str(e)):
                body.pop("response_format", None)
                self.send_json_format = False
                body["messages"][-1]["content"] += "\n\nReply with ONE valid JSON object only - no text before or after it."
                data = _http_json(url, body, headers, self.timeout)
            else:
                raise
        msg = ((data.get("choices") or [{}])[0].get("message") or {})
        content = msg.get("content") or ""
        if isinstance(content, list):  # some servers return content parts
            content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
        return content


    def stream(self, system: str, user: str, json_mode: bool = True, max_tokens: int = 2048) -> Iterator[str]:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        body: dict[str, Any] = {"model": self.model, "messages": msgs, "max_tokens": max_tokens, "stream": True, **self.extra_body}
        if json_mode and self.send_json_format:
            body["response_format"] = {"type": "json_object"}
        elif json_mode:
            body["messages"][-1]["content"] += "\n\nReply with ONE valid JSON object only - no text before or after it."
        headers = {**self.extra_headers}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        for line in _sse_lines(self.base.rstrip("/") + "/chat/completions", body, headers, self.timeout):
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                d = json.loads(data)
            except ValueError:
                continue
            delta = ((d.get("choices") or [{}])[0].get("delta") or {}).get("content")
            if delta:
                yield delta


@dataclass
class AnthropicProvider(Provider):
    key: str = ""

    def complete(self, system: str, user: str, json_mode: bool = True, max_tokens: int = 2048) -> str:
        body: dict[str, Any] = {"model": self.model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": user}]}
        if system:
            body["system"] = system + ("\nReply with a single JSON object and nothing else." if json_mode else "")
        data = _http_json("https://api.anthropic.com/v1/messages", body, {"x-api-key": self.key, "anthropic-version": "2023-06-01"}, self.timeout)
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


# Free local models, best first, for a 4-8 GB GPU. Apache-2.0 Gemma 4 handles Telugu/Hindi/Tamil/... well
# for its size; Ollama offloads to the CPU automatically when the GPU is too small.
OLLAMA_PREFERRED = ("gemma4:e4b", "gemma4:e2b", "gemma4", "gemma4:12b", "qwen3:4b", "gemma3:4b", "llama3.2:3b")


@dataclass
class OllamaProvider(Provider):
    """Local LLM through the Ollama daemon (https://ollama.com). Works with no internet at all.

    Detected lazily: if Ollama is not running, the provider simply reports not-ok and the chain moves
    on; start Ollama later and it is picked up within ~20 seconds, no app restart needed.
    """

    url: str = "http://127.0.0.1:11434"
    local: bool = True
    _seen_at: float = 0.0
    _alive: bool = False
    _installed: tuple[str, ...] = ()

    def _probe(self) -> None:
        if time.time() - self._seen_at < 20:
            return
        self._seen_at = time.time()
        try:
            with urllib.request.urlopen(self.url.rstrip("/") + "/api/tags", timeout=1.5) as r:
                tags = json.loads(r.read().decode("utf-8")).get("models") or []
            self._installed = tuple(str(m.get("name") or m.get("model") or "") for m in tags)
            self._alive = True
        except Exception:
            self._alive, self._installed = False, ()

    def pick_model(self) -> str | None:
        """The configured model if it is pulled, else the best preferred model that is, else any."""
        self._probe()
        if not self._installed:
            return None
        have = {n: n for n in self._installed} | {n.split(":")[0] + (":latest" if ":" not in n else ""): n for n in self._installed}
        norm = lambda m: m if ":" in m else m + ":latest"
        for want in (self.model, *OLLAMA_PREFERRED):
            if want in have:
                return have[want]
            if norm(want) in self._installed:
                return norm(want)
        return self._installed[0]

    def reachable(self) -> bool:
        return self.pick_model() is not None

    def ok(self) -> bool:
        return super().ok() and self.reachable()

    def complete(self, system: str, user: str, json_mode: bool = True, max_tokens: int = 2048) -> str:
        model = self.pick_model() or self.model
        body: dict[str, Any] = {
            "model": model, "stream": False, "think": False, "keep_alive": os.environ.get("INDICLEARN_OLLAMA_KEEP_ALIVE", "30m"),
            "options": {"temperature": 0.3, "num_predict": max_tokens, "num_ctx": int(os.environ.get("INDICLEARN_OLLAMA_CTX", "8192"))},
            "messages": ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}],
        }
        if json_mode:
            body["format"] = "json"
        try:
            data = _http_json(self.url.rstrip("/") + "/api/chat", body, {}, self.timeout)
        except RuntimeError as e:
            if "think" in str(e).lower():  # older Ollama or a model without a thinking switch
                body.pop("think", None)
                data = _http_json(self.url.rstrip("/") + "/api/chat", body, {}, self.timeout)
            else:
                raise
        return (data.get("message") or {}).get("content") or ""

    def stream(self, system: str, user: str, json_mode: bool = True, max_tokens: int = 2048) -> Iterator[str]:
        body: dict[str, Any] = {
            "model": self.pick_model() or self.model, "stream": True, "think": False, "keep_alive": os.environ.get("INDICLEARN_OLLAMA_KEEP_ALIVE", "30m"),
            "options": {"temperature": 0.3, "num_predict": max_tokens, "num_ctx": int(os.environ.get("INDICLEARN_OLLAMA_CTX", "8192"))},
            "messages": ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}],
        }
        if json_mode:
            body["format"] = "json"
        for line in _sse_lines(self.url.rstrip("/") + "/api/chat", body, {}, self.timeout):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            piece = (d.get("message") or {}).get("content")
            if piece:
                yield piece
            if d.get("done"):
                break


# --------------------------------------------------------------------------- prompts
SYSTEM_TUTOR = """You are "Guru", a warm, patient tutor for Indian engineering students inside the IndicLearn app.
Students often think in Telugu (or another Indian language) while their notes are in English.
Your job: make the concept truly click, in the student's language.

Rules
- Write the whole answer in {lang_name}{script_rule}. If that is a Romanized form, write that language in English letters.
- Keep technical terms, symbols and formulas in English as they appear in the notes (e.g. "convolution", "x(t)", "LTI"),
  and explain them in {lang_name} the first time. Do not translate formulas. Write English technical words in English
  letters ("model", "parameters", "accuracy") - never transliterate them into the {lang_name} script; students search
  and write exams with the English terms.
- Teach, don't just define: start with the core idea in one or two sentences, then explain step by step,
  then give a concrete everyday or engineering example, then a one-line "remember this".
- Use the NOTES first. Put [p.N] after a sentence that comes from page N of the notes.
- If the notes do not cover something, you may use your general knowledge, but set "used_general_knowledge": true
  and say so in "beyond_notes". Never invent page numbers or claim the notes say something they don't.
- Be encouraging and friendly, like a good senior explaining over tea. No filler, no lectures about studying.
- Markdown allowed in "answer": **bold**, bullet lines starting with "- ", `code`. No headings, no tables, no LaTeX.

Return ONLY a JSON object with exactly these keys:
{{
  "answer": "the full explanation in {lang_name} (markdown, 120-350 words; shorter for simple follow-ups)",
  "reasoning": ["3-5 short steps, in {lang_name}, showing how you worked out the answer: what the question asks, which parts of the notes you used, how they connect"],
  "key_terms": [{{"term": "English term", "meaning": "one-line meaning in {lang_name}"}}],
  "follow_ups": ["3 short next questions the student could tap, written in {lang_name}"],
  "check": {{"q": "one quick multiple-choice question in {lang_name} that tests understanding (not recall of a number)",
            "options": ["four short, plausible, DIFFERENT answer texts in {lang_name} - real content, no letters like A/B"],
            "answer": "<0-3: index of the single correct option>", "why": "one line in {lang_name} explaining why it is correct"}},
  "used_general_knowledge": false,
  "beyond_notes": "" or "one line in {lang_name} saying what came from outside the notes"
}}"""

SUMMARY_RULES = """

THIS IS A SUMMARY REQUEST. In "answer": (1) one or two sentences saying what the topic is and why it matters,
then (2) 4-6 bullet lines ("- ") with the key ideas IN YOUR OWN WORDS, each a complete thought a student can revise from,
then (3) a last line starting with "Remember:". Never paste sentences from the notes; ignore stray commas or citation
leftovers such as ", , ,". Merge repeated ideas. Keep technical terms in English."""

SYSTEM_UNDERSTAND = """You read a student's question for a study app. The student may write in English, Telugu, Hindi,
Sanskrit, any other Indian language, Romanized Telugu/Hindi (Indian language typed in English letters, e.g.
"convolution ante enti", "fourier transform gurinchi cheppu", "ye kya hota hai"), or a code-mixed blend.
Judge the language from the STUDENT QUESTION ONLY - never from the earlier conversation. A question written in
plain English is "en" even if the conversation so far was in Telugu.
Return ONLY JSON:
{"lang": "<ISO code of the language of the STUDENT QUESTION: en, te, hi, sa, ta, kn, ml, mr, bn, gu, ...>",
 "romanized": <true if an Indian language typed in Latin letters>,
 "search_query": "<the question rewritten as a clear, standalone {doc_lang_name} search query using the technical terms a textbook would use; resolve 'this/it' using the conversation>",
 "intent": "<one of: explain, example, simple, summary, compare, formula, quiz, recommend, exam, greet, other>"}
Use "exam" when the student asks what is important for the exam, which questions may come, or how to prepare for it;
"quiz" only when they want to be asked questions now; "recommend" only for what to study next in this app."""

SYSTEM_TRANSLATE = """You translate study-app tutor replies for Indian students.
Translate the given JSON values into {lang_name}. If {lang_name} is a Romanized form, write it in English letters.
Keep markdown (**bold**, "- " bullets, `code`), page tags like [p.3], formulas, symbols and English technical
terms (e.g. convolution, LTI, Fourier transform) exactly as they are. Natural, simple, student-friendly wording.
Return ONLY a JSON object with the same keys as the input."""

SYSTEM_GRAPH = """You build a clean concept map from a student's study notes (any subject).
From the SECTIONS given, extract the concepts a student must learn - real ideas, methods, laws, components,
properties or named techniques. NOT: author names, figure/table labels, citations, numbers, generic words
("introduction", "method", "result"), or sentence fragments.
Return ONLY JSON:
{"concepts": [{"name": "canonical name as in the notes", "aliases": ["abbreviation or other spelling used"],
               "section": "<section id the concept belongs to>", "definition": "one sentence, from the notes when possible",
               "importance": 0.0-1.0}],
 "prereqs": [{"before": "concept name", "after": "concept name", "why": "short reason"}],
 "related": [{"a": "concept name", "b": "concept name", "why": "short reason"}]}
Prerequisite = you genuinely must understand "before" to understand "after". Be strict; fewer, correct links beat many.
At most {max_concepts} concepts."""


SYSTEM_QUIZ = """You write exam-quality multiple-choice questions for Indian engineering students, from their own notes.
For EACH concept below write {per} question(s) that test UNDERSTANDING: why / how / what happens if / which statement is
true / apply it to a small case. Do NOT ask for exact numbers, dataset sizes, author names, years or page numbers.
Every question has exactly 4 options and exactly one correct answer, which must be supported by the concept's notes.
The 3 wrong options are plausible misconceptions from the same topic, with similar length and style to the right one.
Never use "all of the above", "none of the above", or options that differ only in wording.
Write in {lang_name}{script_rule}; keep technical terms, symbols and formulas in English.
Return ONLY JSON:
{{"questions": [{{"concept_id": "<id given>", "q": "question", "options": ["...", "...", "...", "..."],
                 "answer": <0-3>, "why": "1-2 lines: why the right answer is right, and the trap in the most tempting wrong option (quote its words - never say 'option 2' or 'B': options are shuffled)",
                 "page": <page number of the notes it comes from>}}]}}"""


# --------------------------------------------------------------------------- the brain
def label(p: Provider) -> str:
    """'gemini:model', or 'gemini#2:model' for the second Google account."""
    acc = getattr(p, "account", 1)
    return f"{p.name}{'#' + str(acc) if acc > 1 else ''}:{p.model}"


@dataclass
class Brain:
    providers: list[Provider] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    # ---- construction
    @classmethod
    def from_env(cls, llm_model: str = "gemma3:4b", ollama_url: str = "http://127.0.0.1:11434", probe_ollama: bool = True) -> Brain:
        env = os.environ.get
        order_raw = (env("INDICLEARN_BRAIN") or "auto").strip().lower()
        if order_raw in ("off", "none", "extractive", "0"):
            return cls([])
        order = ["gemini", "sarvam", "openai", "anthropic", "ollama"] if order_raw == "auto" else [x.strip() for x in order_raw.split(",") if x.strip()]
        timeout = float(env("INDICLEARN_BRAIN_TIMEOUT", "90"))
        out: list[Provider] = []
        for name in order:
            if name == "gemini":
                keys = gemini_keys()
                # The free tier allows only ~20 requests/day PER MODEL PER ACCOUNT. Order: the best model on every
                # account first, then the next model, ... - whichever account still has quota answers; one that hits
                # its limit cools down on its own while the others carry on.
                for m in [x.strip() for x in env("INDICLEARN_GEMINI_MODEL", DEFAULT_GEMINI_MODELS).split(",") if x.strip()]:
                    for i, key in enumerate(keys, 1):
                        out.append(GeminiProvider("gemini", m, timeout, key=key, account=i))
            elif name == "sarvam":
                key = env("INDICLEARN_SARVAM_API_KEY") or env("SARVAM_API_KEY")
                if key:
                    out.append(OpenAICompatProvider("sarvam", env("INDICLEARN_SARVAM_MODEL", "sarvam-105b"), timeout, key=key,
                                                    base="https://api.sarvam.ai/v1", extra_headers={"api-subscription-key": key},
                                                    # sarvam-105b "thinks" by default and can spend the whole token
                                                    # budget on hidden reasoning, returning an EMPTY answer. Off = fast.
                                                    extra_body={"reasoning_effort": None}))
            elif name == "openai":
                base, key = env("INDICLEARN_OPENAI_BASE_URL"), env("INDICLEARN_OPENAI_API_KEY") or ""
                model = env("INDICLEARN_OPENAI_MODEL")
                if base and model:
                    # a local server (LM Studio, llama.cpp, vLLM on this PC) gets the shorter "local" prompts
                    is_local = (env("INDICLEARN_OPENAI_LOCAL") or "").lower() in ("1", "true", "yes", "on") or any(
                        h in base for h in ("127.0.0.1", "localhost", "0.0.0.0"))
                    out.append(OpenAICompatProvider("openai", model, float(env("INDICLEARN_OLLAMA_TIMEOUT", "180")) if is_local else timeout,
                                                    key=key, base=base, local=is_local))
            elif name == "anthropic":
                key = env("INDICLEARN_ANTHROPIC_API_KEY") or env("ANTHROPIC_API_KEY")
                if key:
                    out.append(AnthropicProvider("anthropic", env("INDICLEARN_ANTHROPIC_MODEL", "claude-haiku-5-5"), timeout, key=key))
            elif name == "ollama":
                # Always part of the chain: it is the offline fallback when keys are missing or the
                # internet / provider is down. Costs nothing when Ollama is not installed.
                if (env("INDICLEARN_OLLAMA", "on") or "on").lower() not in ("off", "0", "false", "no"):
                    out.append(OllamaProvider("ollama", env("INDICLEARN_OLLAMA_MODEL") or llm_model,
                                              float(env("INDICLEARN_OLLAMA_TIMEOUT", "180")), url=ollama_url))
        return cls(out)

    # ---- status
    @property
    def available(self) -> bool:
        """True when at least one provider can be tried right now (hosted key set, or Ollama up)."""
        return any(p.ok() or (not p.local and p.down_until > 0) for p in self.providers)

    def first_is_local(self) -> bool:
        """True when the engine that would answer right now runs on this computer (slow: do big jobs in the background)."""
        for p in self.providers:
            if p.ok():
                return p.local
        return False

    def status(self) -> dict[str, Any]:
        now_ = time.time()
        chain = []
        for p in self.providers:
            if isinstance(p, OllamaProvider):
                m = p.pick_model()
                chain.append(f"ollama:{m} (offline)" if m else "ollama (not running)")
            else:
                chain.append(label(p))
        return {"chain": chain + ["extractive"], "cooling_down": [label(p) for p in self.providers if p.down_until > now_],
                "offline_llm": any(isinstance(p, OllamaProvider) and p.reachable() for p in self.providers),
                "errors": {label(p).split(":")[0]: p.last_error for p in self.providers if p.last_error}}

    def why_unavailable(self) -> str:
        """One readable line for the UI when no AI engine answered."""
        bits = []
        for p in self.providers:
            if isinstance(p, OllamaProvider):
                bits.append("local Gemma/Ollama: " + (p.last_error or ("ready" if p.reachable() else "Ollama is not running or has no model pulled")))
            else:
                bits.append(f"{p.name}: " + (p.last_error or ("cooling down after an error" if not p.ok() else "no answer")))
        return "; ".join(bits) if bits else "no AI key is set (add one to the .env file) and Ollama is not installed"

    def diagnose(self) -> list[dict[str, Any]]:
        """Tiny live call to every provider - used by `python scripts/check_ai.py` and /api/ai/status?test=1."""
        out = []
        for p in self.providers:
            t0 = time.time()
            try:
                if isinstance(p, OllamaProvider) and not p.reachable():
                    raise RuntimeError("Ollama is not running or has no model pulled")
                raw = p.complete("Reply with JSON only.", 'Return {"ok": true, "word": "namaste"}', True, 256)
                got = parse_json(raw)
                out.append({"provider": label(p).split(":")[0], "model": p.model, "ok": bool(got), "seconds": round(time.time() - t0, 1),
                            "error": "" if got else f"unexpected reply: {raw[:120]!r}"})
                if got:
                    p.last_error, p.down_until = "", 0.0
            except Exception as e:
                p.last_error = _friendly_error(e)
                out.append({"provider": label(p).split(":")[0], "model": p.model, "ok": False, "seconds": round(time.time() - t0, 1), "error": p.last_error})
        return out

    # ---- core call with failover
    def _run(self, system: Any, user: Any, json_mode: bool = True, max_tokens: int = 2048, validate: Any = None,
             on_delta: Callable[[str | None], None] | None = None) -> tuple[Any, str] | None:
        """system / user may be strings or callables(provider) -> str (local models get shorter prompts).

        on_delta(raw_so_far) is called while a reply streams in (None = "discard what you showed, retrying")."""
        for p in list(self.providers):
            if not p.ok():
                continue
            t0 = time.time()
            shown = False
            try:
                u = user(p) if callable(user) else user
                sysm = system(p) if callable(system) else system
                budget = min(max_tokens, 1800) if p.local else max_tokens
                if on_delta is None:
                    raw = p.complete(sysm, u, json_mode, budget)
                else:
                    raw = ""
                    for piece in p.stream(sysm, u, json_mode, budget):
                        raw += piece
                        on_delta(raw)
                        shown = True
                val = parse_json(raw) if json_mode else strip_think(raw)
                if validate is not None:
                    val = validate(val)
                if val:
                    log.info("brain: %s answered in %.1fs", label(p), time.time() - t0)
                    p.last_error = ""
                    return val, p.name
                p.last_error = "replied, but not in the expected format" if raw else "empty reply"
                if shown and on_delta is not None:
                    on_delta(None)
                # A malformed reply is the model's fault, not the network's: try the next
                # provider now but keep this one in the chain for the next request.
                log.warning("brain: %s gave an empty/invalid reply - trying next", p.name)
            except Exception as e:
                with self.lock:
                    # A local model that is merely slow/busy is still there - don't lock students out of it.
                    busy_local = p.local and ("timed out" in str(e).lower() or "timeout" in str(e).lower())
                    daily = "perday" in str(e).lower().replace(" ", "") or "per day" in str(e).lower()
                    if not busy_local:
                        p.down_until = time.time() + (QUOTA_COOLDOWN_S if daily else COOLDOWN_S)
                    p.last_error = _friendly_error(e)
                if shown and on_delta is not None:
                    on_delta(None)
                log.warning("brain: %s failed (%s) - trying next", label(p), str(e)[:200])
        return None

    # ---- tasks
    def understand(self, question: str, history: list[dict[str, str]], doc_lang: str) -> dict[str, Any] | None:
        hist = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-4:])
        user = f"CONVERSATION SO FAR:\n{hist or '(none)'}\n\nSTUDENT QUESTION:\n{question}"

        def ok(v: Any) -> Any:
            if isinstance(v, dict) and isinstance(v.get("lang"), str) and isinstance(v.get("search_query"), str):
                v["lang"] = v["lang"].strip().lower()[:8]
                return v
            return None

        r = self._run(SYSTEM_UNDERSTAND.replace("{doc_lang_name}", lang_name(doc_lang)), user, True, 400, ok)
        return {**r[0], "engine": r[1]} if r else None

    def tutor(self, question: str, lang: str, passages: list[dict[str, Any]], focus: dict[str, Any] | None,
              history: list[dict[str, str]], doc_title: str, unit: str = "p.", summary: bool = False,
              on_delta: Callable[[str | None], None] | None = None) -> dict[str, Any] | None:
        def notes_for(local: bool) -> str:
            k, n = ((4, 1000) if local else (8, 1300)) if summary else ((3, 900) if local else (5, 1400))
            return "\n\n".join(f"[{unit}{p['page']}] {p['text'][:n]}" for p in passages[:k]) or "(no matching passage found in the notes)"
        concept = ""
        if focus:
            bits = [f"Concept: {focus.get('full_label') or focus.get('label')}"]
            if focus.get("definition"):
                bits.append(f"Definition in notes: {focus['definition']}")
            if focus.get("prereq_labels"):
                bits.append("Learn first: " + ", ".join(focus["prereq_labels"]))
            concept = "\n".join(bits)
        hist = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in history[-6:])
        def user(p: Provider) -> str:
            return (f"DOCUMENT: {doc_title}\n{concept}\n\nNOTES (retrieved passages):\n{notes_for(p.local)}\n\n"
                    f"CONVERSATION SO FAR:\n{hist or '(none)'}\n\nSTUDENT QUESTION:\n{question}\n\n"
                    f"Answer in {lang_name(lang)}. Use page tags like [{unit}N] only for pages shown above."
                    + (SUMMARY_RULES if summary else ""))

        def ok(v: Any) -> Any:
            if not isinstance(v, dict) or not isinstance(v.get("answer"), str) or len(v["answer"].strip()) < 10:
                return None
            v["reasoning"] = [str(x) for x in (v.get("reasoning") or []) if str(x).strip()][:6]
            v["follow_ups"] = [str(x) for x in (v.get("follow_ups") or []) if str(x).strip()][:3]
            v["key_terms"] = [x for x in (v.get("key_terms") or []) if isinstance(x, dict) and x.get("term")][:6]
            v["check"] = clean_mcq(v.get("check"))
            v["used_general_knowledge"] = v.get("used_general_knowledge") in (True, "true", "True", 1)
            v["beyond_notes"] = empty_if_placeholder(v.get("beyond_notes"))
            for kt in v["key_terms"]:
                kt["term"], kt["meaning"] = str(kt.get("term") or "").strip(), str(kt.get("meaning") or "").strip()
            return v

        def system(p: Provider) -> str:
            sp = SYSTEM_TUTOR.format(lang_name=lang_name(lang), script_rule=script_rule(lang))
            return sp.replace("120-350 words", "100-220 words") if p.local else sp

        relay = None
        if on_delta is not None:
            relay = lambda raw: on_delta(None if raw is None else partial_answer(strip_think(raw)))  # noqa: E731
        r = self._run(system, user, True, 3000, ok, relay)
        return {**r[0], "engine": r[1]} if r else None

    def translate(self, fields: dict[str, Any], target: str) -> dict[str, Any] | None:
        def ok(v: Any) -> Any:
            return v if isinstance(v, dict) and all(k in v for k in fields) else None

        r = self._run(SYSTEM_TRANSLATE.format(lang_name=lang_name(target)), json.dumps(fields, ensure_ascii=False), True, 3000, ok)
        return {**r[0], "_engine": r[1]} if r else None

    def transcribe(self, audio: bytes, mime: str) -> dict[str, str] | None:
        for p in self.providers:
            if not (p.audio and p.ok()):
                continue
            try:
                v = parse_json(p.transcribe(audio, mime)) or {}
                if v.get("text"):
                    return {"text": str(v["text"]).strip(), "lang": str(v.get("lang") or "")[:5], "engine": p.name}
            except Exception as e:
                log.warning("brain: %s transcription failed (%s)", p.name, str(e)[:200])
        return None

    def quiz(self, concepts: list[dict[str, Any]], lang: str = "en", per: int = 2) -> tuple[list[dict[str, Any]], str] | None:
        """MCQs for a batch of concepts: [{"id", "label", "definition", "notes": [(page, text)]}] -> validated questions."""
        ids = {c["id"] for c in concepts}

        def body(p: Provider) -> str:
            n = 700 if p.local else 1400
            out = []
            for c in concepts:
                notes = "\n".join(f"[p.{pg}] {t[:n]}" for pg, t in c.get("notes", [])[:2])
                out.append(f"CONCEPT id={c['id']}: {c['label']}\nDefinition: {c.get('definition') or '-'}\nNotes:\n{notes or '-'}")
            return "\n\n".join(out)

        def ok(v: Any) -> Any:
            if not isinstance(v, dict) or not isinstance(v.get("questions"), list):
                return None
            good = []
            for q in v["questions"]:
                cq = clean_mcq(q, min_opts=4)
                if cq and str(cq.get("concept_id")) in ids and len(cq["options"]) == 4:
                    try:
                        cq["page"] = int(cq.get("page") or 0)
                    except (TypeError, ValueError):
                        cq["page"] = 0
                    good.append(cq)
            return good or None

        system = SYSTEM_QUIZ.format(per=per, lang_name=lang_name(lang), script_rule=script_rule(lang))
        r = self._run(system, body, True, 900 * max(1, len(concepts)) * per // 2 + 600, ok)
        return (r[0], r[1]) if r else None

    def verify_mcqs(self, items: list[dict[str, Any]]) -> list[int | None] | None:
        """Answer each MCQ blind (no key given): [{"q", "options", "notes"}] -> chosen index per question.

        A question whose written key disagrees with an independent answer is ambiguous or wrongly keyed,
        and a wrong answer key is the fastest way to lose a student's trust - the caller drops it."""
        if not items:
            return []
        body = "\n\n".join(f"Q{i}. NOTES: {it.get('notes', '')[:700]}\nQUESTION: {it['q']}\n" +
                            "\n".join(f"  {j}) {o}" for j, o in enumerate(it["options"])) for i, it in enumerate(items))
        system = ("You are a careful examiner. For each question pick the ONE best option using the notes and sound engineering "
                  'knowledge. If two options are equally correct or none is, answer -1. Return ONLY JSON {"answers": {"Q0": <option number>, "Q1": ...}}')

        def ok(v: Any) -> Any:
            a = v.get("answers") if isinstance(v, dict) else None
            if isinstance(a, list) and len(a) == len(items):
                return a
            if isinstance(a, dict) and a:
                return [a.get(f"Q{i}", a.get(str(i))) for i in range(len(items))]
            return None

        r = self._run(system, body, True, 400 + 20 * len(items), ok)
        if not r:
            return None
        out: list[int | None] = []
        for x in r[0]:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                out.append(None)
        return out

    def extract_graph(self, sections: list[dict[str, str]], max_concepts: int = 40) -> dict[str, Any] | None:
        def body(p: Provider) -> str:
            # a local model has a small context window (and is slow): give it a trimmed view of every section
            per, cap = (700, 9000) if p.local else (1800, 24000)
            return "\n\n".join(f"SECTION {s['id']}: {s['title']}\n{s['text'][:per]}" for s in sections)[:cap]

        def ok(v: Any) -> Any:
            return v if isinstance(v, dict) and isinstance(v.get("concepts"), list) and v["concepts"] else None

        def system(p: Provider) -> str:
            if p.local:  # keep the reply short enough for a small local model to finish
                return (SYSTEM_GRAPH.replace("{max_concepts}", str(min(max_concepts, 14)))
                        + "\nKeep every definition under 20 words. At most 10 prereqs and 8 related links. No aliases unless used in the notes.")
            return SYSTEM_GRAPH.replace("{max_concepts}", str(max_concepts))

        r = self._run(system, body, True, 6000, ok)
        return {**r[0], "_engine": r[1]} if r else None
