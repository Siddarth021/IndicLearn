"""Score offline-tutor answers (from gemma_bench.py) with a strong LLM judge.

    python training/judge_tutor.py training/reports/gemma_gemma4_e2b.json [more files...]

Judge: Gemma 4 31B through the Gemini API (free tier, generous daily limit) - set JUDGE_MODEL to change.
Per answer it scores correctness vs the passage, clarity for a 2nd-year student, and whether the reply
is in the requested language/script with English technical terms kept. Writes <file>.judged.json and
prints averages.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from indiclearn.studio.__main__ import load_dotenv  # noqa: E402

load_dotenv(str(ROOT / ".env"))
os.environ["INDICLEARN_OLLAMA"] = "off"
os.environ["INDICLEARN_BRAIN"] = "gemini"
os.environ["INDICLEARN_GEMINI_MODEL"] = os.environ.get("JUDGE_MODEL", "gemma-4-31b-it")
os.environ.setdefault("INDICLEARN_BRAIN_TIMEOUT", "240")
from indiclearn.studio.brain import Brain  # noqa: E402

SYSTEM = """You grade answers of an AI tutor for Indian B.Tech students. You get the study PASSAGE, the student's
QUESTION, the REQUESTED reply language, and the tutor's ANSWER. Return ONLY JSON:
{"correct": 1-5 (5 = fully correct and consistent with the passage / standard engineering knowledge; 1 = wrong or invented),
 "clarity": 1-5 (5 = a 2nd-year student would understand it: clear steps, a good example),
 "language_ok": true/false (written in the requested language and script; English technical terms may stay in English),
 "answers_question": true/false,
 "issue": "the single biggest problem in a few words, or empty"}"""


def judge(brain: Brain, r: dict) -> dict:
    if not r.get("answer"):
        return {"correct": 1, "clarity": 1, "language_ok": False, "answers_question": False, "issue": "no answer / invalid JSON"}
    user = (f"PASSAGE:\n{r['passage'][:1200]}\n\nQUESTION: {r['question']}\nREQUESTED LANGUAGE: "
            f"{ {'te': 'Telugu (Telugu script)', 'hi': 'Hindi (Devanagari)', 'en': 'English'}[r['reply_lang']] }\n\nANSWER:\n{r['answer'][:3000]}")
    for _ in range(4):
        got = brain._run(SYSTEM, user, True, 800, lambda v: v if isinstance(v, dict) and "correct" in v else None)
        if got:
            return got[0]
        for p in brain.providers:
            p.down_until = 0
        time.sleep(20)
    return {}


def main() -> None:
    brain = Brain.from_env(probe_ollama=False)
    for f in sys.argv[1:]:
        rows = json.loads(Path(f).read_text(encoding="utf-8"))
        with ThreadPoolExecutor(4) as ex:
            scores = list(ex.map(lambda r: judge(brain, r), rows))
        for r, s in zip(rows, scores):
            r["judge"] = s
        Path(f).with_suffix(".judged.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        ok = [r["judge"] for r in rows if r["judge"]]
        avg = lambda k: round(sum(float(j.get(k, 0)) for j in ok) / max(1, len(ok)), 2)
        rate = lambda k: round(sum(1 for j in ok if j.get(k) is True) / max(1, len(ok)), 2)
        by_lang = {}
        for r in rows:
            if r["judge"]:
                by_lang.setdefault(r["reply_lang"], []).append(float(r["judge"].get("correct", 0)))
        print(Path(f).name, {"n": len(rows), "judged": len(ok), "parsed": sum(r["parsed"] for r in rows), "correct": avg("correct"),
                             "clarity": avg("clarity"), "language_ok": rate("language_ok"), "answers_question": rate("answers_question"),
                             "correct_by_lang": {k: round(sum(v) / len(v), 2) for k, v in by_lang.items()}})


if __name__ == "__main__":
    main()
