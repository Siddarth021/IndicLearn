"""Write and check theory exam questions with an LLM (shared by the bank builder and the paper generator).

gen_topic(): a teacher model writes fill / MCQ / short / medium / long questions with model answers and mark schemes,
optionally grounded on passages from the student's own notes. check(): a second model solves every MCQ blind and
reviews every answer; only questions it agrees with are kept.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from indiclearn.studio import brain as B

SYSTEM_GEN = """You are an experienced Operating Systems professor at an Indian engineering college (B.Tech CSE). You set
mid-semester and end-semester papers, and you know the GATE CSE syllabus. Write ORIGINAL exam questions on ONE topic.
Mix difficulty: some direct, some that make students think (why / what happens if / compare / apply to a scenario).
Model answers must be what a top student writes in the exam: correct, concise, exam-style, with the key terms.
Return ONLY JSON:
{"questions": [
 {"qtype": "fill", "marks": 1, "difficulty": "easy|medium|hard", "q": "sentence with ____ for the blank", "answer": "the word(s)"},
 {"qtype": "mcq", "marks": 1, "difficulty": "...", "q": "...", "options": ["4 distinct real options"], "correct": <0-3>, "answer": "one line why"},
 {"qtype": "msq", "marks": 1 or 2, "difficulty": "...", "q": "GATE-style: which of the following is/are correct?", "options": ["4 statements"], "correct": [<indices of ALL correct options, 1 to 3 of them>], "answer": "why each correct one is correct"},
 {"qtype": "short", "marks": 2 or 3, "difficulty": "...", "q": "...", "answer": "40-80 words", "scheme": [{"point": "...", "marks": 1}]},
 {"qtype": "medium", "marks": 4 or 5, "difficulty": "...", "q": "...", "answer": "100-180 words; use '- ' bullets where natural",
  "scheme": [{"point": "...", "marks": 1}], "diagram": "what the expected diagram shows, or empty"},
 {"qtype": "long", "marks": 8 or 10, "difficulty": "...", "q": "...", "answer": "220-380 words, structured with short headings",
  "scheme": [{"point": "...", "marks": 2}], "diagram": "..."}
]}
Rules: scheme marks add up to the question's marks. No numerical problems needing a computed table (those are generated
separately) - but small reasoning with numbers is fine. Plain text only (no LaTeX, no $). Never copy a known paper verbatim."""

SYSTEM_CHECK = """You are a strict OS examiner checking questions before they go into a paper. For EACH question:
- MCQ (no key is given to you): solve it yourself and put the option number in "pick"; if two options are equally right or
  none is, pick -1.
- MSQ (type "msq", several options may be correct, no key given): put the list of ALL correct option numbers in "pick" and
  one short line per option in "why" explaining why it is or is not correct.
- Every question: "v" = "ok" if it is clear, answerable, on its topic, and the given answer is correct and complete enough
  for the marks; "fix" if only the answer needs correcting (give the corrected "answer"); "drop" if it is wrong, ambiguous,
  trivial (the answer is given away in the question) or off-topic.
Return ONLY JSON: {"verdicts": {"Q0": {"v": "ok|fix|drop", "pick": <MCQ only>, "answer": "<only for fix>"}, "Q1": {...}}}"""


def gen_topic(brain: Any, unit: dict, topic: dict, per: int, notes: str = "", want: dict | None = None, tries: int = 3) -> list[dict]:
    want = want or {"fill": per, "mcq": per, "short": per + 1, "medium": per, "long": max(1, per - 1)}
    ask = ", ".join(f"{n} {k}" for k, n in want.items() if n)
    user = (f"SUBJECT: Operating Systems\nUNIT: {unit['title']}\nTOPIC: {topic['label']}\n"
            + (f"STUDENT'S OWN CLASS NOTES on this topic (base the questions on what these notes teach, in the professor's terms):\n{notes[:3500]}\n" if notes else "")
            + f"Write exactly: {ask} questions.")
    r = None
    for _ in range(tries):
        r = brain._run(SYSTEM_GEN, user, True, 7000, lambda v: v if isinstance(v, dict) and isinstance(v.get("questions"), list) and v["questions"] else None)
        if r:
            break
        time.sleep(5)
    if not r:
        return []
    out = []
    only = [k for k, n in want.items() if n]
    for q in r[0]["questions"]:
        q = clean(q)
        if q and (len(only) != 1 or q["qtype"] == only[0]):  # asked for one type only: drop strays
            out.append({**q, "subject": "os", "unit": unit["id"], "topic": topic["id"], "engine": r[1]})
    return out


def clean(q: dict) -> dict | None:
    if isinstance(q, dict) and isinstance(q.get("correct"), list) and isinstance(q.get("options"), list):
        q = {**q, "qtype": "msq"}  # several keyed options = multiple-select, whatever the label says
    if not isinstance(q, dict) or q.get("qtype") not in ("fill", "mcq", "msq", "short", "medium", "long"):
        return None
    text = str(q.get("q") or "").strip()
    if len(text) < 10:
        return None
    try:
        marks = int(q.get("marks") or {"fill": 1, "mcq": 1, "msq": 2, "short": 2, "medium": 4, "long": 8}[q["qtype"]])
    except (TypeError, ValueError):
        return None
    out = {"qtype": q["qtype"], "marks": marks, "difficulty": q.get("difficulty") if q.get("difficulty") in ("easy", "medium", "hard") else "medium",
           "q": text, "answer": str(q.get("answer") or "").strip()}
    if q["qtype"] == "mcq":
        m = B.clean_mcq({"q": text, "options": q.get("options"), "answer": q.get("correct"), "why": out["answer"]})
        if not m or len(m["options"]) != 4:
            return None
        out.update(options=m["options"], correct=m["answer"], answer=m["why"])
    if q["qtype"] == "msq":
        opts = [B._OPT_PREFIX.sub("", str(o)).strip() for o in (q.get("options") or [])]
        corr = q.get("correct")
        if len(opts) != 4 or len({o.lower() for o in opts}) != 4 or not isinstance(corr, list):
            return None
        corr = sorted({int(c) for c in corr if isinstance(c, int) or str(c).isdigit()})
        if not corr or not all(0 <= c < 4 for c in corr):
            return None
        out.update(options=opts, correct=corr)
    if q["qtype"] == "fill":
        ans = str(q.get("answer") or "").split("(")[0].strip().lower()
        if "__" not in text or not ans or ans in text.lower():  # a blank whose answer is printed in the sentence teaches nothing
            return None
    if not out["answer"]:
        return None
    if q["qtype"] in ("short", "medium", "long"):
        scheme = [s for s in (q.get("scheme") or []) if isinstance(s, dict) and s.get("point")]
        total = sum(int(s.get("marks") or 0) for s in scheme if str(s.get("marks", "")).isdigit())
        out["scheme"] = [{"point": str(s["point"]), "marks": int(s.get("marks") or 0)} for s in scheme] if total == marks else []
        out["diagram"] = B.empty_if_placeholder(str(q.get("diagram") or "").strip())
    out["id"] = "os-" + hashlib.sha1(text.lower().encode()).hexdigest()[:10]
    return out


SYSTEM_SOLVE = """Solve these GATE-style multiple-select questions on Operating Systems. Several options may be correct.
Return ONLY JSON {"answers": {"Q0": [<all correct option numbers>], "Q1": [...]}}"""


def _ints(x: Any) -> list[int] | None:
    return sorted({int(v) for v in x if isinstance(v, int) or str(v).isdigit()}) if isinstance(x, list) else None


def check(brain: Any, qs: list[dict], second: Any = None) -> list[dict]:
    items = []
    for i, q in enumerate(qs):
        d = {"id": f"Q{i}", "topic": q["topic"], "type": q["qtype"], "marks": q["marks"], "q": q["q"], "answer": q["answer"]}
        if q["qtype"] in ("mcq", "msq"):
            d.update(options=q["options"])
            d.pop("answer")  # the key and its explanation stay hidden: the checker must solve it
        items.append(d)
    r = brain._run(SYSTEM_CHECK, json.dumps(items, ensure_ascii=False), True, 5000,
                   lambda v: v if isinstance(v, dict) and isinstance(v.get("verdicts"), dict) else None)
    if not r:
        return []  # unchecked questions never ship
    keep: list[dict] = []
    disputed: list[tuple[dict, list[int], Any]] = []
    for i, q in enumerate(qs):
        v = r[0]["verdicts"].get(f"Q{i}") or {}
        verdict = v.get("v")
        if q["qtype"] == "msq":
            pick = _ints(v.get("pick"))
            if not pick or verdict == "drop" or r[1] != "gemini":
                continue
            if pick == q["correct"]:
                keep.append({**q, "checked_by": r[1]})
            else:
                disputed.append((q, pick, v.get("why")))
            continue
        if q["qtype"] == "mcq":
            if v.get("pick") != q["correct"] or verdict == "drop":
                continue  # the blind solve disagrees with the key: ambiguous or wrongly keyed
            keep.append({**q, "checked_by": r[1]})
            continue
        if verdict == "ok":
            keep.append({**q, "checked_by": r[1]})
        elif verdict == "fix" and v.get("answer"):
            q = {**q, "answer": str(v["answer"]).strip(), "checked_by": r[1], "fixed": True}
            keep.append(q)
    if disputed and second is not None:
        # writer and checker disagree on an MSQ key: a second model solves it blind; if it agrees with the checker,
        # the checker's answer becomes the key (two independent solvers beat one writer)
        body = json.dumps([{"id": f"Q{i}", "q": q["q"], "options": q["options"]} for i, (q, _p, _w) in enumerate(disputed)], ensure_ascii=False)
        r2 = second._run(SYSTEM_SOLVE, body, True, 1500, lambda v: v if isinstance(v, dict) and isinstance(v.get("answers"), dict) else None)
        for i, (q, pick, why) in enumerate(disputed):
            if r2 and _ints(r2[0]["answers"].get(f"Q{i}")) == pick:
                expl = why if isinstance(why, str) else "; ".join(f"({'abcd'[k]}) {w}" for k, w in enumerate(why)) if isinstance(why, list) else q["answer"]
                keep.append({**q, "correct": pick, "answer": expl, "checked_by": f"{r[1]}+{r2[1]}", "rekeyed": True})
    return keep

