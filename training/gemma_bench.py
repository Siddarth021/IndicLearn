"""Benchmark the OFFLINE tutor model (Gemma 4 via Ollama) on real student-style doubts - runs on Kaggle.

Uses the app's own tutor prompt (brain.py, local-model variant) on training/data/eval/tutor_eval.jsonl
(56 doubts in English, Telugu, Romanized Telugu and Hindi about engineering passages).
Writes gemma_<model>.json with every raw answer, parse success, reply-language check and speed.
Judging the answers (vs Gemini) happens afterwards with training/judge_tutor.py.

    python gemma_bench.py --src <folder with brain.py and eval/> --out <folder> [--models gemma4:e2b,gemma4:e4b]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def sh(cmd: str) -> None:
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def ollama_chat(model: str, system: str, user: str, ctx: int = 8192) -> tuple[str, dict]:
    body = {"model": model, "stream": False, "think": False, "format": "json",
            "options": {"temperature": 0.3, "num_predict": 1800, "num_ctx": ctx},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    return (d.get("message") or {}).get("content") or "", {k: d.get(k) for k in ("eval_count", "eval_duration", "prompt_eval_count", "total_duration")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="gemma4:e2b,gemma4:e4b")
    ap.add_argument("--no-install", action="store_true", help="Ollama is already running with the models")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    spec = importlib.util.spec_from_file_location("brain", src / "brain.py")
    brain = importlib.util.module_from_spec(spec)
    sys.modules["brain"] = brain  # dataclasses need the module registered
    spec.loader.exec_module(brain)  # type: ignore[union-attr]
    items = [json.loads(x) for x in (src / "eval" / "tutor_eval.jsonl").read_text(encoding="utf-8").split("\n") if x.strip()]

    if not a.no_install:
        sh("apt-get update -qq && apt-get install -y -qq zstd > /dev/null")
        sh("curl -fsSL https://ollama.com/install.sh | sh")
        subprocess.Popen("ollama serve > /kaggle/working/ollama.log 2>&1", shell=True)
        time.sleep(8)
    for model in a.models.split(","):
        if not a.no_install:
            sh(f"ollama pull {model}")
        res = []
        for it in items:
            lang = {"te": "te", "te-Latn": "te", "hi": "hi"}.get(it["style"], "en")
            system = brain.SYSTEM_TUTOR.format(lang_name=brain.lang_name(lang), script_rule=brain.script_rule(lang)).replace("120-350 words", "100-220 words")
            user = (f"DOCUMENT: {it['title']}\n\nNOTES (retrieved passages):\n[p.1] {it['passage'][:900]}\n\nCONVERSATION SO FAR:\n(none)\n\n"
                    f"STUDENT QUESTION:\n{it['question']}\n\nAnswer in {brain.lang_name(lang)}. Use page tags like [p.N] only for pages shown above.")
            t0 = time.time()
            try:
                raw, meta = ollama_chat(model, system, user)
            except Exception as e:
                raw, meta = "", {"error": str(e)}
            v = brain.parse_json(raw)
            ans = (v or {}).get("answer", "") if isinstance(v, dict) else ""
            tel = len(re.findall(r"[ఀ-౿]", ans)) / max(1, len(re.findall(r"\w", ans)))
            res.append({**it, "reply_lang": lang, "raw": raw, "parsed": v is not None and bool(ans), "answer": ans,
                        "check_ok": bool(v and brain.clean_mcq((v or {}).get("check"))), "telugu_ratio": round(tel, 3),
                        "secs": round(time.time() - t0, 1), **meta})
            print(model, it["style"], res[-1]["parsed"], res[-1]["secs"], flush=True)
        (out / f"gemma_{model.replace(':', '_')}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        ok = [r for r in res if r["parsed"]]
        tps = [r["eval_count"] / (r["eval_duration"] / 1e9) for r in res if r.get("eval_count") and r.get("eval_duration")]
        print(model, "parsed", len(ok), "/", len(res), "tok/s", round(sum(tps) / max(1, len(tps)), 1), flush=True)


if __name__ == "__main__":
    main()
