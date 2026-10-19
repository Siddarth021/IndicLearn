"""Hosted speech for Indian languages (Sarvam AI): natural voices out, accurate code-mixed speech in.

Why hosted first: the offline voices (MMS-TTS) switch between a Telugu and an English voice in the
middle of a sentence and sound robotic, and offline Whisper turns "convolution" into "communication".
Sarvam's Bulbul reads Telugu-English code-mix in ONE natural voice, and Saaras transcribes it with
the English terms intact - and accepts the document's own concept names as ``keyterms``.

Both calls are plain HTTPS (urllib, no extra package).  Any failure returns ``None`` and puts the
service on a short cool-down, so the caller falls back to the offline models immediately.

Env: INDICLEARN_SARVAM_API_KEY (shared with the chat engine), INDICLEARN_SARVAM_TTS_MODEL (bulbul:v3),
     INDICLEARN_SARVAM_SPEAKER (optional), INDICLEARN_SARVAM_ASR_MODEL (saaras:v4),
     INDICLEARN_HOSTED_SPEECH=off to keep audio on this computer.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

log = logging.getLogger("indiclearn.speech")

BASE = "https://api.sarvam.ai"
COOLDOWN_S = 60.0
LOCALE = {"en": "en-IN", "te": "te-IN", "hi": "hi-IN", "ta": "ta-IN", "kn": "kn-IN", "ml": "ml-IN", "mr": "mr-IN",
          "bn": "bn-IN", "gu": "gu-IN", "pa": "pa-IN", "or": "od-IN", "sa": "hi-IN"}
_down_until = {"tts": 0.0, "asr": 0.0}
last_error = {"tts": "", "asr": ""}


def _key() -> str:
    if (os.environ.get("INDICLEARN_HOSTED_SPEECH", "on") or "on").lower() in ("off", "0", "false", "no"):
        return ""
    return (os.environ.get("INDICLEARN_SARVAM_API_KEY") or os.environ.get("SARVAM_API_KEY") or "").strip()


def available(kind: str = "tts") -> bool:
    return bool(_key()) and time.time() >= _down_until[kind]


def status() -> dict[str, Any]:
    return {"provider": "sarvam" if _key() else None, "tts": available("tts"), "asr": available("asr"),
            "errors": {k: v for k, v in last_error.items() if v}}


def _fail(kind: str, e: Exception) -> None:
    if "HTTP 400" not in str(e):  # a bad request is about this input, not an outage: don't switch voice off for everyone
        _down_until[kind] = time.time() + COOLDOWN_S
    last_error[kind] = str(e)[:200]
    log.warning("sarvam %s failed: %s", kind, str(e)[:200])


def _post(url: str, data: bytes, headers: dict[str, str], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(url, data=data, headers={"api-subscription-key": _key(), **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}") from e


def _pieces(text: str, limit: int) -> list[str]:
    """Split at sentence ends so no request is over *limit* characters."""
    sents = re.split(r"(?<=[.!?।॥\n])\s+", text)
    out, cur = [], ""
    for s in sents:
        while len(s) > limit:
            out.append(s[:limit])
            s = s[limit:]
        if len(cur) + len(s) + 1 > limit and cur:
            out.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        out.append(cur)
    return [p for p in out if re.search(r"\w", p)]


def tts(text: str, lang: str) -> list[bytes] | None:
    """Text -> list of WAV clips (one per ~1500-character piece), or None to fall back offline."""
    if not available("tts"):
        return None
    code = LOCALE.get((lang or "en").split("-")[0], "en-IN")
    model = os.environ.get("INDICLEARN_SARVAM_TTS_MODEL", "bulbul:v3")
    speaker = os.environ.get("INDICLEARN_SARVAM_SPEAKER", "").strip()
    out: list[bytes] = []
    try:
        for piece in _pieces(text, 1500):
            body: dict[str, Any] = {"text": piece, "target_language_code": code, "language_code": code, "model": model, "pace": 1.0}
            if speaker:
                body["speaker"] = speaker
            data = _post(BASE + "/text-to-speech", json.dumps(body).encode("utf-8"), {"Content-Type": "application/json"}, 40)
            out += [base64.b64decode(a) for a in data.get("audios") or []]
        last_error["tts"] = ""
        return out or None
    except Exception as e:
        _fail("tts", e)
        return None


def stt(audio: bytes, filename: str, mime: str, keyterms: list[str] | None = None) -> dict[str, str] | None:
    """Speech -> {"text", "lang"}. Code-mixed output keeps English technical terms in English letters."""
    if not available("asr") or not audio:
        return None
    model = os.environ.get("INDICLEARN_SARVAM_ASR_MODEL", "saaras:v4")
    fields = {"model": model, "mode": "codemix"}
    # Sarvam rejects keyterms containing , or ; (e.g. a slide title "Encoding, Decoding") -> clean them
    clean = (re.sub(r"\s+", " ", re.sub(r"[,;]", " ", k or "")).strip() for k in (keyterms or []))
    terms = [t for t in dict.fromkeys(clean) if 2 < len(t) <= 64][:50]
    if terms and model.endswith("v4"):
        fields["keyterms"] = json.dumps(terms, ensure_ascii=False)
    got = _stt_call(fields, audio, filename, mime)
    if got is None and "keyterms" in fields and "400" in last_error.get("asr", ""):  # never lose a doubt over the hints
        fields.pop("keyterms")
        got = _stt_call(fields, audio, filename, mime)
    return got


def _stt_call(fields: dict, audio: bytes, filename: str, mime: str) -> dict[str, str] | None:
    boundary = "----il" + uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n'.encode() + audio + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    try:
        data = _post(BASE + "/speech-to-text", b"".join(parts), {"Content-Type": f"multipart/form-data; boundary={boundary}"}, 40)
        last_error["asr"] = ""
        text = str(data.get("transcript") or "").strip()
        lang = str(data.get("language_code") or "").split("-")[0]
        return {"text": text, "lang": lang, "engine": "sarvam"} if text else None
    except Exception as e:
        _fail("asr", e)
        return None
