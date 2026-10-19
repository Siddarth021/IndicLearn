"""Step 4 - turn teacher labels + public datasets into the four training sets.

Run:  python training/build_datasets.py
Out:  training/datasets/m{1,2,3,4}/{train,val,test}.jsonl  and  training/datasets/STATS.json

Splits are by ARTICLE (an article's spans never appear in two splits), stratified by subject, seed 13.

  m1  {"span", "context", "label"}                  1 = a concept worth a node on the map (teacher-judged)
  m2  {"input", "target"}                           "define: <term> | context: <sentence>" -> one-line definition
  m3  {"a", "b", "context", "label"}                prereq (learn a before b) | related | none
  m4  {"input", "target"}                           "question: <concept> | passage: <text>" ->
                                                    "Q: ... A) ... B) ... C) ... D) ... ANSWER: B"   (one line:
                                                    T5 tokenizers cannot emit newlines - the old M4 broke on that)
"""

from __future__ import annotations

import collections
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW, INTERIM, OUT = HERE / "data" / "raw", HERE / "data" / "interim", HERE / "datasets"
SEED = 13
LETTERS = "ABCD"


def jl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()] if path.exists() else []


def write(name: str, split: str, rows: list[dict]) -> None:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\([^)]*\)", "", s or "")).strip().lower()


def sentence_with(term: str, text: str) -> str:
    for s in re.split(r"(?<=[.!?])\s+", text):
        if norm(term) and norm(term) in s.lower():
            return s[:400]
    return ""


def split_articles(arts: list[dict], rng: random.Random) -> dict[str, str]:
    by_sub = collections.defaultdict(list)
    for a in arts:
        by_sub[a["subject"]].append(a["title"])
    where = {}
    for titles in by_sub.values():
        rng.shuffle(titles)
        n = len(titles)
        k = max(1, round(n * 0.1)) if n >= 6 else 0
        for i, t in enumerate(titles):
            where[t] = "test" if i < k else "val" if i < 2 * k else "train"
    return where


def mcq_line(q: str, opts: list[str], ans: int) -> str:
    return f"Q: {q} " + " ".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(opts)) + f" ANSWER: {LETTERS[ans]}"


def main() -> None:
    rng = random.Random(SEED)
    teacher = list({t["title"]: t for t in reversed(jl(INTERIM / "teacher.jsonl"))}.values())  # one label set per article
    cands = {a["title"]: a for a in jl(INTERIM / "candidates.jsonl")}
    where = split_articles([{"title": t["title"], "subject": t["subject"]} for t in teacher], rng)
    sets: dict[str, dict[str, list]] = {m: collections.defaultdict(list) for m in ("m1", "m2", "m3", "m4")}

    for t in teacher:
        sp = where[t["title"]]
        art = cands.get(t["title"], {})
        text = " ".join(s["text"] for s in art.get("sections", []))
        cs = t["candidates"]
        keep = set(t["keep"])
        concepts: dict[str, str] = {}  # name -> definition
        # ---- M1: every judged candidate, with the same context the app gives the model
        for i, c in enumerate(cs, 1):
            if i in keep or i in t["junk"]:
                sets["m1"][sp].append({"span": c["label"], "context": c["context"], "label": int(i in keep), "src": t["title"]})
            if i in keep and str(i) in t["definitions"]:
                concepts[c["label"]] = t["definitions"][str(i)]
        for m in t["missing"]:
            ctx = sentence_with(m["term"], text)
            if ctx:
                sets["m1"][sp].append({"span": m["term"], "context": ctx, "label": 1, "src": t["title"]})
            concepts[m["term"]] = m["definition"]
        # ---- M2: term (+ the sentence it appears in) -> teacher definition
        for term, d in concepts.items():
            ctx = next((c["context"] for c in cs if c["label"] == term), "") or sentence_with(term, text)
            sets["m2"][sp].append({"input": f"define: {term} | context: {ctx[:300]}", "target": d, "src": t["title"]})
        # ---- M3: teacher links + "none" pairs (unlinked concepts of the same article, and reversed prereqs)
        names = list(concepts)
        linked = set()
        num = lambda x: cs[int(x) - 1]["label"] if str(x).strip().isdigit() and 1 <= int(x) <= len(cs) else str(x)  # teacher sometimes answers by number
        for p in t["prereqs"]:
            p["before"], p["after"] = num(p["before"]), num(p["after"])
        for r in t["related"]:
            r["a"], r["b"] = num(r["a"]), num(r["b"])
        defs = {norm(k): v for k, v in concepts.items()}
        ctx_of = lambda a, b: f"{defs.get(norm(a), '')} {defs.get(norm(b), '')}".strip()[:400]
        for p in t["prereqs"]:
            a, b = p["before"], p["after"]
            if norm(a) == norm(b):
                continue
            sets["m3"][sp].append({"a": a, "b": b, "context": ctx_of(a, b), "label": "prereq", "src": t["title"]})
            sets["m3"][sp].append({"a": b, "b": a, "context": ctx_of(b, a), "label": "none", "src": t["title"]})
            linked |= {(norm(a), norm(b)), (norm(b), norm(a))}
        for r in t["related"]:
            a, b = r["a"], r["b"]
            if norm(a) == norm(b):
                continue
            for x, y in ((a, b), (b, a)):
                sets["m3"][sp].append({"a": x, "b": y, "context": ctx_of(x, y), "label": "related", "src": t["title"]})
            linked |= {(norm(a), norm(b)), (norm(b), norm(a))}
        pairs = [(a, b) for a in names for b in names if a != b and (norm(a), norm(b)) not in linked]
        rng.shuffle(pairs)
        n_pos = len(t["prereqs"]) + 2 * len(t["related"])
        for a, b in pairs[: max(2, n_pos)]:
            sets["m3"][sp].append({"a": a, "b": b, "context": ctx_of(a, b), "label": "none", "src": t["title"]})
        # ---- M4: teacher MCQs, passage = the section that mentions the concept
        for q in t["mcqs"]:
            sec = next((s["text"] for s in art.get("sections", []) if norm(q.get("concept", "")) in s["text"].lower()), "")
            passage = (sec or text)[:1200]
            order = list(range(4))
            rng.shuffle(order)
            opts = [q["options"][i] for i in order]
            sets["m4"][sp].append({"input": f"question: {q.get('concept', '')} | passage: {passage}",
                                   "target": mcq_line(q["q"], opts, order.index(q["answer"])), "src": t["title"]})

    # ---- public data: SciQ (human-written distractors) for M4
    for split, fname in (("train", "train"), ("val", "validation"), ("test", "test")):
        for r in jl(RAW / "sciq" / f"{fname}.jsonl"):
            if len(r.get("support") or "") < 80:
                continue
            opts = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
            if len({o.strip().lower() for o in opts}) < 4:
                continue
            order = list(range(4))
            rng.shuffle(order)
            o2 = [opts[i] for i in order]
            topic = r["correct_answer"] if len(r["correct_answer"]) < 40 else ""
            sets["m4"][split].append({"input": f"question: {topic} | passage: {r['support'][:1200]}",
                                      "target": mcq_line(r["question"], o2, order.index(0)), "src": "sciq"})

    # ---- public data: LectureBank 2 prerequisites for M3 (all 471 positives, 4x as many negatives)
    lb = RAW / "lecturebank"
    if (lb / "322topics_final.tsv").exists():
        topics = dict(line.split("|", 1) for line in (lb / "322topics_final.tsv").read_text(encoding="utf-8").splitlines() if "|" in line)
        rows = [l.split(",") for l in (lb / "final_new_annotation.csv").read_text().splitlines() if l.count(",") == 2]
        pos = [(a, b) for a, b, y in rows if y.strip() == "1" and a in topics and b in topics]
        neg = [(a, b) for a, b, y in rows if y.strip() == "0" and a in topics and b in topics]
        rng.shuffle(neg)
        lbrows = [{"a": topics[a].strip(), "b": topics[b].strip(), "context": "", "label": "prereq", "src": "lecturebank"} for a, b in pos]
        lbrows += [{"a": topics[b].strip(), "b": topics[a].strip(), "context": "", "label": "none", "src": "lecturebank"} for a, b in pos]
        lbrows += [{"a": topics[a].strip(), "b": topics[b].strip(), "context": "", "label": "none", "src": "lecturebank"} for a, b in neg[: 3 * len(pos)]]
        sets["m3"]["train"] += lbrows  # LectureBank is NLP/ML topics: train only, never in the engineering test set

    stats = {}
    for m, sp in sets.items():
        stats[m] = {}
        for split in ("train", "val", "test"):
            rows = sp[split]
            rng.shuffle(rows)
            write(m, split, rows)
            lab = collections.Counter(r.get("label", "-") for r in rows)
            stats[m][split] = {"n": len(rows), "labels": dict(lab), "sources": dict(collections.Counter(r["src"] if r["src"] in ("sciq", "lecturebank") else "teacher" for r in rows))}
    (OUT / "STATS.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
