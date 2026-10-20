"""Telugu training data for the multilingual students M2-te (definitions) and M4-te (quizzes).

The teacher (the app's hosted engine chain, Gemini first) rewrites the English targets of the existing
M2 / M4 datasets in Telugu the way IIIT lecturers actually speak: Telugu sentences, technical terms kept
in English letters ("Deadlock అంటే ..."). Inputs stay English (the notes are English), so the students
learn cross-lingual generation: English context in, Telugu out.

Run:  python training/make_telugu_data.py            # resumable
Out:  training/datasets/m2_te/{train,val,test}.jsonl, training/datasets/m4_te/{train,val,test}.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
from indiclearn.studio.__main__ import load_dotenv  # noqa: E402

load_dotenv(str(ROOT / ".env"))
os.environ["INDICLEARN_OLLAMA"] = "off"
os.environ["INDICLEARN_BRAIN_BUDGET_S"] = "0"
os.environ.setdefault("INDICLEARN_BRAIN_TIMEOUT", "240")
os.environ.setdefault("INDICLEARN_GEMINI_THINKING", "minimal")
from indiclearn.studio.brain import Brain  # noqa: E402

DS = HERE / "datasets"
CACHE = HERE / "data" / "interim" / "telugu_cache.jsonl"
SIZES = {"m2": {"train": 1300, "val": 100, "test": 200}, "m4": {"train": 850, "val": 60, "test": 150}}
TE = re.compile(r"[\u0C00-\u0C7F]")

SYSTEM = """You translate study material for Telugu-medium B.Tech students in Andhra Pradesh and Telangana.
Write natural, simple Telugu as a good lecturer would say it in class. Keep every technical term, symbol,
formula, number, unit and proper name in English letters exactly as given (e.g. "Deadlock", "CPU", "O(n log n)").
Do not add or drop information. Return JSON only."""


def te_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    return sum(1 for c in letters if TE.match(c)) / max(1, len(letters))


def parse_mcq(t: str):
    m = re.match(r"Q:\s*(.*?)\s+A\)\s*(.*?)\s+B\)\s*(.*?)\s+C\)\s*(.*?)\s+D\)\s*(.*?)\s+ANSWER:\s*([ABCD])\s*$", t, re.S)
    return None if not m else {"q": m.group(1), "options": [m.group(i) for i in range(2, 6)], "answer": m.group(6)}


def load_cache() -> dict[str, dict]:
    out = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["key"]] = r
    return out


def pick(model: str, split: str, n: int) -> list[dict]:
    rows = [json.loads(x) for x in (DS / model / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    rnd = random.Random(13)
    if model == "m4":
        rows = [r for r in rows if parse_mcq(r["target"])]
        eng = [r for r in rows if r["src"] != "sciq"]  # engineering MCQs written by the teacher come first
        sci = [r for r in rows if r["src"] == "sciq"]
        rnd.shuffle(sci)
        rows = eng + sci
        return rows[:n]
    rnd.shuffle(rows)
    return rows[:n]


def key(model: str, r: dict) -> str:
    return f"{model}|{r['input'][:200]}|{r['target'][:120]}"


def batch_prompt(model: str, batch: list[dict]) -> str:
    if model == "m2":
        items = [{"i": i, "term": r["input"].split("|")[0].replace("define:", "").strip(), "definition": r["target"]} for i, r in enumerate(batch)]
        return ("Translate each one-line definition into Telugu (keep the term itself in English).\n"
                'Reply: {"items":[{"i":0,"te":"..."}, ...]} with every i.\n\n' + json.dumps(items, ensure_ascii=False))
    items = [{"i": i, **parse_mcq(r["target"])} for i, r in enumerate(batch)]
    return ("Translate each multiple-choice question and its four options into Telugu. Keep the option order and the answer letter.\n"
            'Reply: {"items":[{"i":0,"q":"...","options":["...","...","...","..."]}, ...]} with every i.\n\n' + json.dumps(items, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default="m2,m4")
    ap.add_argument("--export", help="write the items still missing to this folder (for an external teacher) and stop")
    ap.add_argument("--build-only", action="store_true", help="only write the datasets from the cache")
    a = ap.parse_args()
    brain = Brain.from_env(probe_ollama=False)
    cache = load_cache()
    lock = threading.Lock()
    jobs = []
    for model in a.only.split(","):
        per = 12 if model == "m2" else 6
        for split, n in SIZES[model].items():
            todo = [r for r in pick(model, split, n) if key(model, r) not in cache]
            jobs += [(model, todo[i:i + per]) for i in range(0, len(todo), per)]
    if a.export:
        ex_dir = Path(a.export); ex_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for model, batch in jobs:
            for r in batch:
                item = {"key": key(model, r), "model": model}
                if model == "m2":
                    item.update(term=r["input"].split("|")[0].replace("define:", "").strip(), definition=r["target"])
                else:
                    item.update(parse_mcq(r["target"]))
                with (ex_dir / f"todo_{n // 400:02d}.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                n += 1
        print("exported", n); return
    if a.build_only:
        jobs = []
    print(f"{len(jobs)} teacher calls to make", flush=True)
    done = [0]

    def run(job):
        model, batch = job

        def ok(v):
            items = (v or {}).get("items") if isinstance(v, dict) else None
            if not isinstance(items, list):
                return None
            good = {}
            for it in items:
                try:
                    i = int(it["i"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not 0 <= i < len(batch):
                    continue
                if model == "m2" and isinstance(it.get("te"), str) and te_ratio(it["te"]) > 0.12:
                    good[i] = {"te": it["te"].strip()}
                if model == "m4" and isinstance(it.get("q"), str) and isinstance(it.get("options"), list) and len(it["options"]) == 4 and te_ratio(it["q"]) > 0.3:
                    good[i] = {"q": it["q"].strip(), "options": [str(o).strip() for o in it["options"]]}
            return good if len(good) >= len(batch) // 2 else None

        got = brain._run(SYSTEM, batch_prompt(model, batch), json_mode=True, max_tokens=8000, validate=ok)
        if not got:
            return
        res, eng = got
        with lock:
            with CACHE.open("a", encoding="utf-8") as f:
                for i, r in enumerate(batch):
                    if i in res:
                        rec = {"key": key(model, r), "model": model, "engine": eng, **res[i]}
                        cache[rec["key"]] = rec
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[0] += 1
            if done[0] % 10 == 0:
                print(f"  {done[0]}/{len(jobs)} calls", flush=True)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run, jobs))
    # ---- write the datasets
    stats = {}
    for model in ("m2", "m4"):
        out = DS / f"{model}_te"
        out.mkdir(exist_ok=True)
        for split, n in SIZES[model].items():
            rows = []
            for r in pick(model, split, n):
                c = cache.get(key(model, r))
                if not c:
                    continue
                if model == "m2":
                    rows.append({"input": "define in Telugu: " + r["input"].replace("define:", "").strip(), "target": c["te"], "target_en": r["target"], "src": r.get("src", "")})
                else:
                    mc = parse_mcq(r["target"])
                    o = c["options"]
                    tgt = f"ప్రశ్న: {c['q']} A) {o[0]} B) {o[1]} C) {o[2]} D) {o[3]} జవాబు: {mc['answer']}"
                    rows.append({"input": "quiz in Telugu: " + r["input"].replace("question:", "").strip(), "target": tgt, "target_en": r["target"], "src": r.get("src", "")})
            (out / f"{split}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
            stats[f"{model}_te/{split}"] = len(rows)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
