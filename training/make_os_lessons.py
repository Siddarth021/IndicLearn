"""Build a course's lessons (one per syllabus topic), in English and Telugu.  Works for every subject course.

    python training/make_os_lessons.py                  # Operating Systems, all topics, resumable
    python training/make_os_lessons.py --subject dsma   # Discrete Structures and Matrix Algebra
    python training/make_os_lessons.py --topics paging --force

Writer: Sarvam (fast). Reviewer: Gemini, a different model, which corrects any section it finds wrong.
Telugu: translated section by section with English technical terms kept in English letters.
Output: src/indiclearn/studio/courses/<subject>_lessons.jsonl  ({"topic", "unit", "en": lesson, "te": lesson}).
Numerical topics get a live worked example in the app (courses/oscalc.py, courses/dsmacalc.py), so the lesson text explains the method.
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

TEACHER = {
    "os": ("the best Operating Systems teacher", 'Plain text with **bold** and "- " bullets only (no LaTeX, no $, no tables).'),
    "dsma": ("the best Discrete Mathematics and Linear Algebra teacher (you teach from Rosen and Gilbert Strang)",
             'Use **bold** and "- " bullets. Write mathematics in LaTeX between single dollar signs, e.g. $p \\rightarrow q$, '
             '$\\binom{n}{k}$, $A = LU$, $\\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\end{bmatrix}$ - each formula on ONE line, never $$ blocks, no '
             'markdown tables. Every number in an example must be exactly right: recompute each step before you write it.'),
}

SYSTEM_LESSON_T = """You are {teacher} at an Indian engineering college. Write ONE lesson on the given
topic for a B.Tech CSE student preparing for semester exams (and GATE). Be correct, concrete and exam-oriented; teach the
idea first, then the details; use a real-world analogy where it truly helps. {fmt} Return ONLY JSON:
{"title": "<topic name>",
 "summary": "2-3 sentences: what it is and why it matters",
 "sections": [{"heading": "...", "body": "80-200 words"}],            // 3-5 sections, in teaching order
 "key_terms": [{"term": "English term", "meaning": "one line"}],      // 4-8 terms
 "example": {"title": "...", "body": "a worked example or scenario, step by step (60-180 words)"},
 "exam_tips": ["what examiners ask on this topic and how to score full marks"],  // 3-5
 "mistakes": ["common student mistakes"],                                          // 2-4
 "diagram": "the one diagram a student should be able to draw for this topic, described in a sentence (or empty)"}"""
SYSTEM_LESSON = SYSTEM_LESSON_T.replace("{teacher}", TEACHER["os"][0]).replace("{fmt}", TEACHER["os"][1])

SYSTEM_REVIEW = """You are a strict professor of this subject reviewing a lesson before students read it. Check every statement.
Return ONLY JSON {"verdict": "ok" | "fix", "fixes": {"<field path>": "<corrected text>"}}
Field paths: "summary", "sections.<i>.body", "example.body", "exam_tips.<i>", "mistakes.<i>", "key_terms.<i>.meaning".
Only list fields that contain an actual error or a misleading statement; keep the author's style. Recompute every number and
every step of the worked example yourself; a wrong number is an error to fix."""


def write(brain: B.Brain, unit: dict, topic: dict, subject: str = "os", course: str = "Operating Systems") -> dict | None:
    numerical = topic.get("numerical")
    user = (f"COURSE: {course}\nUNIT: {unit['title']}\nTOPIC: {topic['label']}\n"
            + ("This topic has numerical problems in exams: explain the method/algorithm step by step and the exact conventions "
               "(tie-breaking, what is counted); a live solved example is shown separately.\n" if numerical else ""))

    def ok(v: object) -> object:
        full = isinstance(v, dict) and v.get("summary") and isinstance(v.get("sections"), list) and len(v["sections"]) >= 2
        lists = all(isinstance(v.get(k), list) and v[k] for k in ("key_terms", "exam_tips", "mistakes"))
        strs = lists and all(isinstance(x, str) for k in ("exam_tips", "mistakes") for x in v[k]) and all(isinstance(x, dict) and x.get("term") for x in v["key_terms"])
        return v if full and strs and isinstance(v.get("example"), dict) and v["example"].get("body") else None

    system = SYSTEM_LESSON_T.replace("{teacher}", TEACHER[subject][0]).replace("{fmt}", TEACHER[subject][1])
    r = brain._run(system, user, True, 6000, ok)
    return r[0] if r else None


def review(brain: B.Brain, lesson: dict, rules: list[str] | None = None) -> tuple[dict, str, int]:
    system = SYSTEM_REVIEW + ("\nGROUND TRUTH - the app solves this topic's numericals with exactly these rules; any statement that "
                              "contradicts them (or teaches a different method, formula or tie-break) is an error to fix:\n- "
                              + "\n- ".join(rules) if rules else "")
    r = brain._run(system, json.dumps(lesson, ensure_ascii=False), True, 4000,
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
    ap.add_argument("--subject", default="os", choices=sorted(TEACHER))
    ap.add_argument("--topics", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--no-te", action="store_true", help="English only now (translate later with --te-only)")
    ap.add_argument("--te-only", action="store_true", help="only (re)translate lessons that have no Telugu yet")
    ap.add_argument("--review-only", action="store_true", help="review lessons that were saved without a review")
    ap.add_argument("--recheck-numerical", action="store_true", help="re-review numerical topics against the solver's exact rules")
    a = ap.parse_args()
    load_dotenv(str(ROOT / ".env"))
    os.environ.setdefault("INDICLEARN_BRAIN_BUDGET_S", "0")  # a batch job may wait for the AI; a student may not
    os.environ["INDICLEARN_BRAIN"] = os.environ.get("WRITER", "sarvam,gemini")
    writer = B.Brain.from_env()
    os.environ["INDICLEARN_BRAIN"] = "gemini"
    os.environ["INDICLEARN_GEMINI_MODEL"] = os.environ.get("TE_MODEL", "gemma-4-26b-a4b-it,gemma-4-31b-it")  # Sarvam out of credits: Gemma translates
    os.environ["INDICLEARN_BRAIN_TIMEOUT"] = "300"  # Gemma thinks before it writes: a whole lesson takes a few minutes
    te_brain = B.Brain.from_env()
    os.environ["INDICLEARN_BRAIN"] = "gemini,sarvam"
    os.environ["INDICLEARN_GEMINI_MODEL"] = os.environ.get("CHECK_MODEL", "gemini-3.5-flash,gemini-2.5-flash,gemini-3.5-flash-lite,gemini-3.8-flash")
    reviewer = B.Brain.from_env()
    global OUT
    OUT = COURSE / f"{a.subject}_lessons.jsonl"
    syl = json.loads((COURSE / f"{a.subject}_syllabus.json").read_text(encoding="utf-8"))
    have = {}
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                have[d["topic"]] = d
    if a.review_only:
        jobs = [(u, t) for u in syl["units"] for t in u["topics"] if t["id"] in have and have[t["id"]].get("reviewed_by") == "unreviewed"]
    elif a.recheck_numerical:
        jobs = [(u, t) for u in syl["units"] for t in u["topics"] if t.get("numerical") and t["id"] in have]
    elif a.te_only:
        jobs = [(u, t) for u in syl["units"] for t in u["topics"] if t["id"] in have and not have[t["id"]].get("te")]
    else:
        jobs = [(u, t) for u in syl["units"] for t in u["topics"] if (not a.topics or t["id"] in a.topics) and (a.force or t["id"] not in have)]
    print(f"{len(jobs)} lessons to write; {len(have)} done", flush=True)

    def work(job: tuple[dict, dict]) -> dict | None:
        u, t = job
        if a.review_only:
            from indiclearn.studio.courses.lessons import rules_for

            d = dict(have[t["id"]])
            en, by, nfix = review(reviewer, d["en"], rules_for(a.subject, t.get("numerical")))
            print(f"  {t['id']:<24} reviewed by {by}: {nfix} fixes", flush=True)
            d["en"], d["reviewed_by"], d["fixes"] = en, by, nfix
            if nfix:
                d["te"] = None
            return d
        if a.recheck_numerical:
            from indiclearn.studio.courses.lessons import rules_for

            d = dict(have[t["id"]])
            en, by, nfix = review(reviewer, d["en"], rules_for(a.subject, t["numerical"]))
            print(f"  {t['id']:<24} rechecked by {by}: {nfix} fixes", flush=True)
            if nfix:
                d["en"], d["te"] = en, None  # the Telugu was made from the old text
            d["rechecked_by"] = by
            return d
        if a.te_only:
            d = dict(have[t["id"]])
            d["te"] = translate(te_brain, d["en"], "te")
            print(f"  {t['id']:<24} telugu {'ok' if d['te'] else 'FAILED'}", flush=True)
            return d
        en = write(writer, u, t, a.subject, syl["title"])
        if not en:
            print(f"  {t['id']:<24} FAILED to write", flush=True)
            return None
        from indiclearn.studio.courses.lessons import rules_for

        en, by, nfix = review(reviewer, en, rules_for(a.subject, t.get("numerical")))
        te = None if a.no_te else translate(te_brain, en, "te")
        print(f"  {t['id']:<24} reviewed by {by} ({nfix} fixes), telugu {'skipped' if a.no_te else 'ok' if te else 'FAILED'}", flush=True)
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
