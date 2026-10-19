"""Same 56 doubts as gemma_bench.py, answered by a hosted engine (reference for the offline model).

    INDICLEARN_BRAIN=sarvam python training/online_bench.py training/reports/online_sarvam.json
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from indiclearn.studio.__main__ import load_dotenv  # noqa: E402

load_dotenv(str(ROOT / ".env"))
os.environ["INDICLEARN_OLLAMA"] = "off"
from indiclearn.studio.brain import Brain, clean_mcq, lang_name, parse_json, script_rule, SYSTEM_TUTOR  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1])
    items = [json.loads(x) for x in (ROOT / "training" / "data" / "eval" / "tutor_eval.jsonl").read_text(encoding="utf-8").split("\n") if x.strip()]
    brain = Brain.from_env(probe_ollama=False)
    p = brain.providers[0]

    def run(it: dict) -> dict:
        lang = {"te": "te", "te-Latn": "te", "hi": "hi"}.get(it["style"], "en")
        system = SYSTEM_TUTOR.format(lang_name=lang_name(lang), script_rule=script_rule(lang))
        user = (f"DOCUMENT: {it['title']}\n\nNOTES (retrieved passages):\n[p.1] {it['passage'][:1400]}\n\nCONVERSATION SO FAR:\n(none)\n\n"
                f"STUDENT QUESTION:\n{it['question']}\n\nAnswer in {lang_name(lang)}. Use page tags like [p.N] only for pages shown above.")
        t0 = time.time()
        try:
            raw = p.complete(system, user, True, 3000)
        except Exception as e:
            raw = ""
            print("ERR", e)
        v = parse_json(raw)
        ans = (v or {}).get("answer", "") if isinstance(v, dict) else ""
        tel = len(re.findall(r"[ఀ-౿]", ans)) / max(1, len(re.findall(r"\w", ans)))
        return {**it, "reply_lang": lang, "raw": raw, "parsed": bool(ans), "answer": ans, "check_ok": bool(v and clean_mcq(v.get("check"))),
                "telugu_ratio": round(tel, 3), "secs": round(time.time() - t0, 1), "engine": f"{p.name}:{p.model}"}

    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(run, items))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out, "parsed", sum(r["parsed"] for r in res), "/", len(res), "avg secs", round(sum(r["secs"] for r in res) / len(res), 1))


if __name__ == "__main__":
    main()
