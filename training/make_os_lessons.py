"""Build the Operating Systems lessons (one per syllabus topic), in English and Telugu.

    python training/make_os_lessons.py                  # all topics, resumable
    python training/make_os_lessons.py --topics paging --force

Writer: Sarvam (fast). Reviewer: Gemini, a different model, which corrects any section it finds wrong.
Telugu: translated section by section with English technical terms kept in English letters.
Output: src/indiclearn/studio/courses/os_lessons.jsonl  ({"topic", "unit", "en": lesson, "te": lesson}).
Numerical topics get a live worked example in the app (courses/oscalc.py), so the lesson text explains the method.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from indiclearn.studio import brain as B  # noqa: E402
from indiclearn.studio.__main__ import load_dotenv  # noqa: E402
from indiclearn.studio.courses.lessons import translate  # noqa: E402

COURSE = ROOT / "src" / "indiclearn" / "studio" / "courses"
OUT = COURSE / "os_lessons.jsonl"

SYSTEM_LESSON = """You are the best Operating Systems teacher at an Indian engineering college. Write ONE lesson on the given
topic for a B.Tech CSE student preparing for semester exams (and GATE). Be correct, concrete and exam-oriented; teach the
idea first, then the details; use a real-world analogy where it truly helps. Plain text with **bold** and "- " bullets only
(no LaTeX, no $, no tables). Return ONLY JSON:
{"title": "<topic name>",
 "summary": "2-3 sentences: what it is and why it matters",
 "sections": [{"heading": "...", "body": "80-200 words"}],            // 3-5 sections, in teaching order
 "key_terms": [{"term": "English term", "meaning": "one line"}],      // 4-8 terms
 "example": {"title": "...", "body": "a worked example or scenario, step by step (60-180 words)"},
 "exam_tips": ["what examiners ask on this topic and how to score full marks"],  // 3-5
 "mistakes": ["common student mistakes"],                                          // 2-4
 "diagram": "the one diagram a student should be able to draw for this topic, described in a sentence (or empty)"}"""

SYSTEM_REVIEW = """You are a strict OS professor reviewing a lesson before students read it. Check every statement.
Return ONLY JSON {"verdict": "ok" | "fix", "fixes": {"<field path>": "<corrected text>"}}
Field paths: "summary", "sections.<i>.body", "example.body", "exam_tips.<i>", "mistakes.<i>", "key_terms.<i>.meaning".
Only list fields that contain an actual error or a misleading statement; keep the author's style."""


def write(brain: B.Brain, unit: dict, topic: dict) -> dict | None:
    numerical = topic.get("numerical")
    user = (f"UNIT: {unit['title']}\nTOPIC: {topic['label']}\n"
            + ("This topic has numerical problems in exams: explain the method/algorithm step by step and the exact conventions "
               "(tie-breaking, what is counted); a live solved example is shown separately.\n" if numerical else ""))

    def ok(v: object) -> object:
        return v if isinstance(v, dict) and v.get("summary") and isinstance(v.get("sections"), list) and len(v["sections"]) >= 2 else None

    r = brain._run(SYSTEM_LESSON, user, True, 6000, ok)
    return r[0] if r else None


def review(brain: B.Brain, lesson: dict) -> tuple[dict, str, int]:
    r = brain._run(SYSTEM_REVIEW, json.dumps(lesson, ensure_ascii=False), True, 4000,
                   lambda v: v if isinstance(v, dict) and v.get("verdict") in ("ok", "fix") else None)
    if not r:
        return lesson, "unreviewed", 0
    fixes = r[0].get("fixes") or {}
    n = 0
    for path, text in fixes.items():
        if not isinstance(text, str) or not text.strip():
            continue
        cur: object = lesson
        parts = path.split(".")
        try:
            for p in parts[:-1]:
                cur = cur[int(p)] if isinstance(cur, list) else cur[p]  # type: ignore[index]
            last = parts[-1]
            if isinstance(cur, list):
                cur[int(last)] = text.strip()
            else:
                cur[last] = text.strip()  # type: ignore[index]
            n += 1
        except (KeyError, IndexError, ValueError, TypeError):
            continue
    return lesson, r[1], n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--te-only", action="store_true", help="only (re)translate lessons that have no Telugu yet")
    a = ap.parse_args()
    load_dotenv(str(ROOT / ".env"))
    os.environ["INDICLEARN_BRAIN"] = os.environ.get("WRITER", "sarvam,gemini")
    writer = B.Brain.from_env()
    os.environ["INDICLEARN_BRAIN"] = "gemini"
    os.environ["INDICLEARN_GEMINI_MODEL"] = os.environ.get("TE_MODEL", "gemma-4-26b-a4b-it,gemma-4-31b-it")  # Sarvam out of credits: Gemma translates
    os.environ["INDICLEARN_BRAIN_TIMEOUT"] = "300"  # Gemma thinks before it writes: a whole lesson takes a few minutes
    te_brain = B.Brain.from_env()
    os.environ["INDICLEARN_BRAIN"] = "gemini,sarvam"
    os.environ["INDICLEARN_GEMINI_MODEL"] = os.environ.get("CHECK_MODEL", "gemini-3.5-flash,gemini-2.5-flash,gemini-3.5-flash-lite,gemini-3.8-flash")
    reviewer = B.Brain.from_env()
    syl = json.loads((COURSE / "os_syllabus.json").read_text(encoding="utf-8"))
    have = {}
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                have[d["topic"]] = d
    if a.te_only:
        jobs = [(u, t) for u in syl["units"] for t in u["topics"] if t["id"] in have and not have[t["id"]].get("te")]
    else:
        jobs = [(u, t) for u in syl["units"] for t in u["topics"] if (not a.topics or t["id"] in a.topics) and (a.force or t["id"] not in have)]
    print(f"{len(jobs)} lessons to write; {len(have)} done", flush=True)

    def work(job: tuple[dict, dict]) -> dict | None:
        u, t = job
        if a.te_only:
            d = dict(have[t["id"]])
            d["te"] = translate(te_brain, d["en"], "te")
            print(f"  {t['id']:<24} telugu {'ok' if d['te'] else 'FAILED'}", flush=True)
            return d
        en = write(writer, u, t)
        if not en:
            print(f"  {t['id']:<24} FAILED to write", flush=True)
            return None
        en, by, nfix = review(reviewer, en)
        te = translate(te_brain, en, "te")
        print(f"  {t['id']:<24} reviewed by {by} ({nfix} fixes), telugu {'ok' if te else 'FAILED'}", flush=True)
        return {"topic": t["id"], "unit": u["id"], "reviewed_by": by, "fixes": nfix, "en": en, "te": te}

    with ThreadPoolExecutor(a.workers) as ex:
        for d in ex.map(work, jobs):
            if d:
                have[d["topic"]] = d
                order = [t["id"] for u in syl["units"] for t in u["topics"]]
                OUT.write_text("".join(json.dumps(have[k], ensure_ascii=False) + "\n" for k in order if k in have), encoding="utf-8")
    print(f"done: {len(have)} lessons; reviewed {sum(1 for d in have.values() if d['reviewed_by'] != 'unreviewed')}; "
          f"telugu {sum(1 for d in have.values() if d.get('te'))}", flush=True)


if __name__ == "__main__":
    main()
