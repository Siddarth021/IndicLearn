"""Build the Operating Systems theory question bank (the seed dataset for exam papers).

    python training/make_os_bank.py                 # all topics, resumable
    python training/make_os_bank.py --topics paging tlb --per 2

For every topic of src/indiclearn/studio/courses/os_syllabus.json a teacher LLM writes exam questions of five kinds
(fill-in-the-blank, MCQ, 2-3 mark short answer, 4-6 mark answer, 8-10 mark long answer) with a model answer and a
mark scheme; a second, blind pass answers every question independently and a checker keeps only questions whose key
it agrees with. Output: src/indiclearn/studio/courses/os_bank.jsonl (shipped with the app) + training/reports/os_bank.json.
Numericals are NOT here: they are generated with exact answers by courses/oscalc.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from indiclearn.studio import brain as B  # noqa: E402
from indiclearn.studio.courses.qgen import check, gen_topic  # noqa: E402
from indiclearn.studio.__main__ import load_dotenv  # noqa: E402

COURSE = ROOT / "src" / "indiclearn" / "studio" / "courses"
OUT = COURSE / "os_bank.jsonl"

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", nargs="*")
    ap.add_argument("--per", type=int, default=3, help="questions per type per topic (short gets +1, long -1)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    load_dotenv(str(ROOT / ".env"))
    import os
    # the writer and the checker are different models where possible, so the check is really independent
    os.environ["INDICLEARN_BRAIN"] = "sarvam,gemini"
    gen_brain = B.Brain.from_env()
    os.environ["INDICLEARN_BRAIN"] = "gemini,sarvam"  # Sarvam checks only when every Gemini model is out of quota
    os.environ["INDICLEARN_GEMINI_MODEL"] = os.environ.get("CHECK_MODEL", "gemini-3.5-flash,gemini-2.5-flash,gemini-3.5-flash-lite")  # a different model family checks
    check_brain = B.Brain.from_env()
    syl = json.loads((COURSE / "os_syllabus.json").read_text(encoding="utf-8"))
    have = {}
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                q = json.loads(line)
                have[q["id"]] = q
    done_topics = {q["topic"] for q in have.values()}
    jobs = [(u, t) for u in syl["units"] for t in u["topics"] if (not a.topics or t["id"] in a.topics) and (a.topics or t["id"] not in done_topics)]
    print(f"{len(jobs)} topics to generate; {len(have)} questions already in the bank", flush=True)

    def work(job: tuple[dict, dict]) -> list[dict]:
        u, t = job
        qs = gen_topic(gen_brain, u, t, a.per)
        kept = check(check_brain, qs) if qs else []
        print(f"  {t['id']:<24} generated {len(qs):>2}  kept {len(kept):>2}", flush=True)
        return kept

    with ThreadPoolExecutor(a.workers) as ex:
        for kept in ex.map(work, jobs):
            for q in kept:
                have[q["id"]] = q
            OUT.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in have.values()), encoding="utf-8")
    stats: dict = {"questions": len(have), "by_type": {}, "by_unit": {}, "fixed_by_checker": sum(1 for q in have.values() if q.get("fixed"))}
    for q in have.values():
        stats["by_type"][q["qtype"]] = stats["by_type"].get(q["qtype"], 0) + 1
        stats["by_unit"][q["unit"]] = stats["by_unit"].get(q["unit"], 0) + 1
    (ROOT / "training" / "reports").mkdir(exist_ok=True)
    (ROOT / "training" / "reports" / "os_bank.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
