"""Download human-translated English-Telugu data (AI4Bharat BPCC), keep technical/educational
sentences, and build the 'te_plus' training set = our lecturer-style pairs + BPCC formal pairs.

One-time, on the laptop (the dataset is gated):
  1. Open https://huggingface.co/datasets/ai4bharat/BPCC and click "Agree and access".
  2. .venv\\Scripts\\python -c "from huggingface_hub import login; login()"
  3. .venv\\Scripts\\python training\\get_bpcc.py

Output: training/datasets/te_plus/{train,val,test}.jsonl
  - train = te_style train (prompt 'to Telugu: ')  +  BPCC rows (prompt 'to Telugu (formal): ')
  - val/test = identical to te_style, so the score is comparable with the 72.0 chrF model.
The two prompts let one model learn BOTH registers without mixing them up.
"""
from __future__ import annotations

import csv
import io
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "training" / "datasets"
FILES = ["bpcc-seed-latest/tel_Telu.tsv", "wiki/tel_Telu.tsv", "daily/tel_Telu.tsv"]
MAX_BPCC = 6000
random.seed(7)


def te_ratio(s: str) -> float:
    te = len(re.findall(r"[ఀ-౿]", s))
    al = len(re.findall(r"[A-Za-zఀ-౿]", s))
    return te / al if al else 0.0


def main() -> int:
    from huggingface_hub import hf_hub_download

    gl = json.loads((ROOT / "models" / "trained" / "te_term_keeper" / "glossary.json").read_text(encoding="utf-8"))
    tech = {w for w, p in gl.items() if p >= 0.9 and len(w) > 3}  # words lecturers keep in English = technical vocabulary
    print("technical vocabulary:", len(tech), "words")

    pairs: list[tuple[str, str]] = []
    for f in FILES:
        try:
            path = hf_hub_download("ai4bharat/BPCC", f, repo_type="dataset")
        except Exception as e:
            print("skip", f, "-", str(e)[:120])
            continue
        n0 = len(pairs)
        with open(path, encoding="utf-8", newline="") as fh:
            for row in csv.reader(fh, delimiter="\t"):
                cells = [c for c in row if c and c.strip()]
                en = next((c for c in cells if te_ratio(c) < 0.05 and re.search(r"[A-Za-z]{3}", c)), None)
                te = next((c for c in cells if te_ratio(c) > 0.8), None)
                if en and te and 6 <= len(en.split()) <= 60:
                    pairs.append((en.strip(), te.strip()))
        print(f"{f}: {len(pairs) - n0} usable pairs")
    if not pairs:
        print("No data. Did you click 'Agree and access' on the dataset page and log in?")
        return 1

    def score(en: str) -> int:
        return sum(1 for w in re.findall(r"[a-z]{4,}", en.lower()) if w in tech)

    scored = sorted(((score(en), en, te) for en, te in pairs), key=lambda x: -x[0])
    keep = [(en, te) for sc, en, te in scored if sc >= 2][:MAX_BPCC]
    print(f"technical/educational BPCC pairs kept: {len(keep)} of {len(pairs)}")

    out = D / "te_plus"
    out.mkdir(exist_ok=True)
    base = [json.loads(l) for l in (D / "te_style" / "train.jsonl").open(encoding="utf-8")]
    rows = list(base) + [{"input": "to Telugu (formal): " + en, "target": te, "target_en": en, "src": "bpcc"} for en, te in keep]
    random.shuffle(rows)
    (out / "train.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    for s in ("val", "test"):
        (out / f"{s}.jsonl").write_bytes((D / "te_style" / f"{s}.jsonl").read_bytes())
    print(f"te_plus train = {len(base)} lecturer-style + {len(keep)} human BPCC = {len(rows)} rows")
    print("Next: tell Claude 'bpcc done' - the Kaggle training starts from there.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
