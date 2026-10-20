"""End-to-end Telugu demo on real course terms (runs on CPU).

English students M2 / M4 write the content from the course notes; the fine-tuned mT5 translator
(te_style_translator) turns it into lecturer Telugu. NLLB-200 is shown next to it for comparison when installed.

    python training/demo_telugu.py [--n 6] [--out training/reports/telugu_demo.json]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "models" / "trained"
COURSES = ROOT / "src" / "indiclearn" / "studio" / "courses"


def load(name):
    p = M / name
    return AutoTokenizer.from_pretrained(p), AutoModelForSeq2SeqLM.from_pretrained(p).eval()


def gen(tm, xs, max_new=160):
    tok, model = tm
    with torch.no_grad():
        e = tok(xs, return_tensors="pt", padding=True, truncation=True, max_length=384)
        g = model.generate(**e, max_new_tokens=max_new, num_beams=4, no_repeat_ngram_size=4)
    return [x.strip() for x in tok.batch_decode(g, skip_special_tokens=True)]


def items(n):
    out = []
    for subj in ("os", "dsma"):
        for line in (COURSES / f"{subj}_lessons.jsonl").read_text(encoding="utf-8").splitlines():
            les = json.loads(line).get("en") or {}
            for kt in les.get("key_terms", [])[:1]:
                body = " ".join(s.get("body", "") for s in les.get("sections", []))
                sent = next((s for s in re.split(r"(?<=[.!?])\s+", body) if kt["term"].split(" (")[0].lower() in s.lower()), les.get("summary", ""))
                out.append({"course": subj.upper(), "term": kt["term"], "context": sent[:400], "passage": body[:900]})
    pick = [x for x in out if len(x["context"]) > 40]
    step = max(1, len(pick) // n)
    return pick[::step][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "training" / "reports" / "telugu_demo.json"))
    a = ap.parse_args()
    its = items(a.n)
    m2, m4, tr = load("m2_describe"), load("m4_quiz"), load("te_style_translator")
    en_def = gen(m2, [f"define: {x['term']} | context: {x['context']}" for x in its], 64)
    en_q = gen(m4, [f"question: {x['term']} | passage: {x['passage']}" for x in its], 160)
    te_def = gen(tr, ["to Telugu: " + d for d in en_def], 200)
    te_q = gen(tr, ["to Telugu: " + q for q in en_q], 256)
    res = [{**{k: x[k] for k in ("course", "term")}, "m2_english": d, "telugu_definition": td, "m4_english": q, "telugu_quiz": tq}
           for x, d, td, q, tq in zip(its, en_def, te_def, en_q, te_q)]
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    for r in res:
        print(f"\n[{r['course']}] {r['term']}\n  EN: {r['m2_english']}\n  TE: {r['telugu_definition']}\n  Q-EN: {r['m4_english']}\n  Q-TE: {r['telugu_quiz']}")


if __name__ == "__main__":
    main()
