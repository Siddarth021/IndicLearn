"""Step 3 - a strong LLM ("teacher") labels the candidates, so the small models can learn from it.

For every article in data/interim/candidates.jsonl ONE call returns:
  * which rule-mined candidates are real concepts a student should learn (M1 positives / negatives),
  * a one-line, student-friendly definition for each real concept (M2 targets),
  * important concepts the rules missed (extra M1 positives + M2 targets),
  * prerequisite and related pairs between concepts (M3),
  * 3 understanding-level MCQs (M4).

The teacher is the app's own engine chain (Gemini free tier first, else Sarvam ...), configured from
the project .env - so labels come from the same models students see online.

Run:  python training/teacher_label.py                 # resumable; --limit N to try a few
      python training/teacher_label.py --workers 3      # parallel calls (Sarvam allows ~40/min)
Out:  training/data/interim/teacher.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
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
os.environ["INDICLEARN_OLLAMA"] = os.environ.get("TEACHER_OLLAMA", "off")  # the teacher must be a strong hosted model

from indiclearn.studio.brain import Brain, clean_mcq  # noqa: E402

IN = HERE / "data" / "interim" / "candidates.jsonl"
OUT = HERE / "data" / "interim" / "teacher.jsonl"
MAX_CANDS = 45

SYSTEM = """You are an expert engineering professor building training data for a study app used by Indian B.Tech students.
You get one textbook-style article and a numbered list of CANDIDATE phrases that a rule-based program extracted from it.

Decide for EVERY candidate: is it a real concept a student should learn and could be a node on a concept map
(a technical term, method, law, theorem, component, property, algorithm, named model or metric)?
Junk = person names, places, years, generic words ("start line", "input frequencies", "physical sciences"),
sentence fragments ("term discrete-time", "length DFT"), units or numbers, overly specific phrases from one sentence.

Return ONLY JSON:
{"keep": [candidate numbers that ARE real concepts],
 "junk": [candidate numbers that are NOT],
 "definitions": {"<number>": "one clear sentence (12-35 words) a 2nd-year student understands; based on the article; do not start with the term itself"},
 "missing": [{"term": "important concept of this article that is NOT in the list", "definition": "one sentence"}],
 "prereqs": [{"before": "concept", "after": "concept", "why": "few words"}],
 "related": [{"a": "concept", "b": "concept", "why": "few words"}],
 "mcqs": [{"concept": "concept", "q": "question testing understanding (why/how/what-if/which is true), not trivia",
           "options": ["four plausible, different answers"], "answer": 0, "why": "one line"}]}
Every candidate number must be in exactly one of keep/junk. Give definitions for ALL kept numbers.
At most 6 missing, 8 prereqs (strict: you truly must know "before" to understand "after"), 8 related, exactly 3 mcqs.
Use concept names exactly as written in the list (or in "missing")."""


def article_text(a: dict, budget: int = 7000) -> str:
    out, used = [], 0
    for s in a["sections"]:
        t = s["text"][: max(400, min(1100, budget - used))]
        out.append(f"## {s['heading']}\n{t}")
        used += len(t)
        if used >= budget:
            break
    return "\n\n".join(out)


def label(brain: Brain, a: dict) -> dict | None:
    cands = a["candidates"][:MAX_CANDS]
    listing = "\n".join(f"{i}. {c['label']}" for i, c in enumerate(cands, 1))
    user = f"ARTICLE: {a['title']}  (subject: {a['subject']})\n\n{article_text(a)}\n\nCANDIDATES:\n{listing or '(none)'}"
    n = len(cands)

    def ok(v):
        if not isinstance(v, dict):
            return None
        keep = {int(x) for x in v.get("keep") or [] if str(x).isdigit() and 1 <= int(x) <= n}
        junk = {int(x) for x in v.get("junk") or [] if str(x).isdigit() and 1 <= int(x) <= n} - keep
        if n and len(keep | junk) < 0.8 * n:  # the teacher skipped too many: ask the next engine
            return None
        defs = {int(k): str(d).strip() for k, d in (v.get("definitions") or {}).items() if str(k).isdigit() and int(k) in keep and len(str(d)) > 20}
        mcqs = [m for m in (clean_mcq(q, 4) for q in v.get("mcqs") or []) if m and len(m["options"]) == 4]
        return {"keep": sorted(keep), "junk": sorted(junk), "definitions": defs,
                "missing": [m for m in v.get("missing") or [] if isinstance(m, dict) and m.get("term") and m.get("definition")][:6],
                "prereqs": [p for p in v.get("prereqs") or [] if isinstance(p, dict) and p.get("before") and p.get("after")][:8],
                "related": [p for p in v.get("related") or [] if isinstance(p, dict) and p.get("a") and p.get("b")][:8],
                "mcqs": mcqs}

    r = brain._run(SYSTEM, user, True, 6000, ok)
    if not r:
        return None
    lab, engine = r
    return {"title": a["title"], "subject": a["subject"], "engine": engine, "candidates": cands, **lab}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--min-interval", type=float, default=float(os.environ.get("TEACHER_MIN_INTERVAL", "6.5")),
                    help="seconds between calls per worker (Gemini free tier ~10 requests/minute)")
    a = ap.parse_args()
    # Each Gemini model has its own free-tier rate limit: one worker per model labels ~3x faster for free.
    models = [m for m in os.environ.get("TEACHER_MODELS", "").split(",") if m.strip()]
    brains = []
    for m in models or [""]:
        b = Brain.from_env(probe_ollama=False)
        if m.strip() == "sarvam":  # "sarvam" in TEACHER_MODELS = a worker on the Sarvam engine (paid credits)
            os.environ["INDICLEARN_BRAIN"], keep = "sarvam", os.environ.get("INDICLEARN_BRAIN", "auto")
            b = Brain.from_env(probe_ollama=False)
            os.environ["INDICLEARN_BRAIN"] = keep
        elif m:
            b.providers = [p for p in b.providers if p.name == "gemini"][:1]
            for p in b.providers:
                p.model = m.strip()
        brains.append(b)
    if not brains[0].providers:
        sys.exit("No hosted engine configured - set INDICLEARN_GEMINI_API_KEY (or SARVAM) in .env")
    print("teachers:", [b.status()["chain"] for b in brains], flush=True)

    done = set()
    if OUT.exists():
        done = {json.loads(ln)["title"] for ln in OUT.read_text(encoding="utf-8").split("\n") if ln.strip()}
    arts = [json.loads(ln) for ln in IN.read_text(encoding="utf-8").split("\n") if ln.strip()]
    todo = [x for x in arts if x["title"] not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(done)} labelled, {len(todo)} to go", flush=True)
    lock = threading.Lock()
    stats = {"ok": 0, "fail": 0}
    import queue as _q
    work_q: _q.Queue = _q.Queue()
    for art in todo:
        work_q.put(art)

    def worker(brain: Brain) -> None:
        """Pull articles until the queue is empty or this model's free daily quota is gone."""
        quota_fails = 0
        while True:
            try:
                art = work_q.get_nowait()
            except _q.Empty:
                return
            t0, res = time.time(), None
            for attempt in range(3):
                try:
                    res = label(brain, art)
                except Exception as e:  # pragma: no cover
                    print("  !", art["title"], e, flush=True)
                if res:
                    break
                err = " ".join(p.last_error for p in brain.providers)
                for p in brain.providers:
                    p.down_until = 0.0
                if "quota" in err or "rate limit" in err:
                    quota_fails += 1
                    if quota_fails >= 2:
                        work_q.put(art)  # give it to another model
                        print(f"  - {brain.providers[0].model}: quota exhausted, worker stops", flush=True)
                        return
                time.sleep(20 * (attempt + 1))
            with lock:
                if res:
                    quota_fails = 0
                    with OUT.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                    stats["ok"] += 1
                else:
                    stats["fail"] += 1
                if (stats["ok"] + stats["fail"]) % 10 == 0:
                    print(f"  {stats} last={res and res['engine']}:{brain.providers[0].model}", flush=True)
            time.sleep(max(0.0, a.min_interval - (time.time() - t0)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    threads = [threading.Thread(target=worker, args=(b,)) for b in brains for _ in range(1 if models else a.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("done", stats, "->", OUT)


if __name__ == "__main__":
    main()
