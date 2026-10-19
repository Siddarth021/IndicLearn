"""Distillation data for the OFFLINE tutor (Gemma 4 E2B): student doubt + notes -> ideal tutor JSON.

Why: plain Gemma 4 E2B writes good Telugu but turns English technical terms into Telugu letters
("ఓమ్స్ లా" for Ohm's law) and sometimes garbles words. Students read and write exams with the English
terms, so the offline tutor should answer like the online one: Telugu sentences, English terms intact.

1. pick corpus sections (training/data/raw/wikipedia.jsonl), excluding the 56 evaluation passages
2. a hosted engine writes realistic student doubts in English / Telugu / Romanized Telugu / Hindi
3. the hosted tutor (same prompt as the app, local-model length) answers them
4. answers are kept only if they parse, are in the requested script and keep English terms

    INDICLEARN_BRAIN=sarvam python training/gemma_sft_data.py --n 1200
Out: training/datasets/gemma_sft/{train,val}.jsonl  ({"messages": [system, user, assistant]})
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
from indiclearn.studio.__main__ import load_dotenv  # noqa: E402

load_dotenv(str(ROOT / ".env"))
os.environ["INDICLEARN_OLLAMA"] = "off"
from indiclearn.studio.brain import SYSTEM_TUTOR, Brain, clean_mcq, lang_name, parse_json, script_rule  # noqa: E402

OUT = HERE / "datasets" / "gemma_sft"
STYLES = ["te", "te-Latn", "te-Latn", "en", "te", "hi-Latn", "te-Latn", "en"]
LANG_OF = {"te": "te", "te-Latn": "te", "hi-Latn": "hi", "en": "en"}


def local_system(lang: str) -> str:
    """Exactly what the app sends a local model (brain.tutor, p.local=True)."""
    return SYSTEM_TUTOR.format(lang_name=lang_name(lang), script_rule=script_rule(lang)).replace("120-350 words", "100-220 words")


def user_msg(title: str, passage: str, q: str, lang: str) -> str:
    return (f"DOCUMENT: {title}\n\nNOTES (retrieved passages):\n[p.1] {passage[:900]}\n\nCONVERSATION SO FAR:\n(none)\n\n"
            f"STUDENT QUESTION:\n{q}\n\nAnswer in {lang_name(lang)}. Use page tags like [p.N] only for pages shown above.")


def good(ans: dict, lang: str) -> bool:
    a = ans.get("answer", "")
    if len(a.split()) < 40:
        return False
    if lang == "te":
        letters = re.findall(r"\w", a)
        if len(re.findall(r"[ఀ-౿]", a)) / max(1, len(letters)) < 0.3:
            return False
        if not re.search(r"[A-Za-z]{3,}", a):  # English technical terms must survive
            return False
    return bool(clean_mcq(ans.get("check")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    rng = random.Random(21)
    eval_passages = {json.loads(x)["passage"][:200] for x in (HERE / "data" / "eval" / "tutor_eval.jsonl").read_text(encoding="utf-8").split("\n") if x.strip()}
    secs = []
    for line in (HERE / "data" / "raw" / "wikipedia.jsonl").read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        art = json.loads(line)
        for s in art["sections"][:8]:
            if 350 < len(s["text"]) and s["text"][:200] not in eval_passages:
                secs.append({"title": art["title"], "heading": s["heading"], "passage": s["text"][:1400]})
    rng.shuffle(secs)
    secs = secs[: a.n]
    partial = OUT / "_partial.jsonl"
    rows_prev = [json.loads(x) for x in partial.read_text(encoding="utf-8").split("\n") if x.strip()] if partial.exists() else []
    seen = {r["messages"][1]["content"].split("[p.1] ", 1)[-1][:120] for r in rows_prev}
    secs = [s_ for s_ in secs if s_["passage"][:120] not in seen]  # resume: skip passages already answered
    brain = Brain.from_env(probe_ollama=False)
    print("engine:", brain.status()["chain"], "sections:", len(secs), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, lock = list(rows_prev), threading.Lock()

    def run(system, user, budget, check):
        """free-tier engines fail often (500 / 503 / per-minute quota): retry with back-off instead of skipping."""
        for attempt in range(5):
            r = brain._run(system, user, True, budget, check)
            if r:
                return r
            for p in brain.providers:
                p.down_until = 0.0
            time.sleep(15 * (attempt + 1))
        return None

    def batch(i: int) -> None:
        group = secs[i : i + 8]
        styles = [STYLES[(i + j) % len(STYLES)] for j in range(len(group))]
        body = "\n\n".join(f"ITEM {j}: style={st} | {g['title']} / {g['heading']}\n{g['passage'][:700]}" for j, (g, st) in enumerate(zip(group, styles)))
        sysm = ("For each ITEM write ONE doubt a B.Tech student would really type about that passage, in the given style: en = English; "
                "te = Telugu script (English technical terms may stay in English); te-Latn = Telugu in English letters like "
                "'convolution ante enti, example tho cheppu'; hi-Latn = Hinglish. Mix 'why/how', 'explain simply', 'difference between', "
                "'give an example'. Return ONLY JSON {\"q\": [one question per item, in order]}")
        got = run(sysm, body, 1500, lambda v: v if isinstance(v, dict) and isinstance(v.get("q"), list) and len(v["q"]) == len(group) else None)
        if not got:
            return
        for g, st, q in zip(group, styles, got[0]["q"]):
            lang = LANG_OF[st]
            sysp, user = local_system(lang), user_msg(g["title"], g["passage"], str(q), lang)
            r = run(SYSTEM_TUTOR.format(lang_name=lang_name(lang), script_rule=script_rule(lang)), user, 2500,
                    lambda v: v if isinstance(v, dict) and isinstance(v.get("answer"), str) else None)
            if not r or not good(r[0], lang):
                continue
            ans = {k: r[0].get(k) for k in ("answer", "reasoning", "key_terms", "follow_ups", "check", "used_general_knowledge", "beyond_notes")}
            with lock:
                rows.append({"style": st, "messages": [{"role": "system", "content": sysp}, {"role": "user", "content": user},
                                                       {"role": "assistant", "content": json.dumps(ans, ensure_ascii=False)}]})
                with partial.open("a", encoding="utf-8") as fh:  # survives a crash / reboot
                    fh.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
                if len(rows) % 50 == 0:
                    print("  kept", len(rows), flush=True)
            for p in brain.providers:
                p.down_until = 0.0

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(batch, range(0, len(secs), 8)))
    rng.shuffle(rows)
    k = max(30, len(rows) // 20)
    for name, part in (("val", rows[:k]), ("train", rows[k:])):
        (OUT / f"{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in part) + "\n", encoding="utf-8")
    (OUT / "_partial.jsonl").unlink(missing_ok=True)
    print("done", len(rows), "examples ->", OUT)


if __name__ == "__main__":
    main()
