"""Fair baseline: run IndicTrans2 (en->te) on the SAME te_style test set and score chrF
against the SAME code-mixed references, so 'beats the baseline' is a like-for-like claim.

    .venv\\Scripts\\python training\\baseline_fair.py

Prints chrF and Telugu-script ratio for IndicTrans2 next to our model's 72.0 / code-mixed.
Needs: models/pretrained/indictrans2_en_indic (already installed), torch, transformers, sacrebleu.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "training" / "datasets" / "te_style" / "test.jsonl"


def te_ratio(s: str) -> float:
    te = len(re.findall(r"[ఀ-౿]", s))
    al = len(re.findall(r"[A-Za-zఀ-౿]", s))
    return te / al if al else 0.0


def main() -> int:
    try:
        import sacrebleu
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as e:
        print("Missing package:", e, "\n  .venv\\Scripts\\pip install sacrebleu")
        return 1

    rows = [json.loads(l) for l in TEST.open(encoding="utf-8")]
    src = [r["target_en"] if r.get("target_en") else r["input"].replace("to Telugu: ", "") for r in rows]
    gold = [r["target"] for r in rows]

    mdir = ROOT / "models" / "pretrained" / "indictrans2_en_indic"
    if not (mdir / "config.json").exists():
        print("IndicTrans2 not found at", mdir, "- run setup_offline.py first.")
        return 1
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, "| items:", len(rows))
    tok = AutoTokenizer.from_pretrained(str(mdir), trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(mdir), trust_remote_code=True).to(dev).eval()

    out = []
    for i in range(0, len(src), 16):
        batch = [f"eng_Latn tel_Telu {s}" for s in src[i:i + 16]]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(dev)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=256, num_beams=5)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
        print(f"  {min(i+16, len(src))}/{len(src)}", end="\r", flush=True)
    print()

    chrf = sacrebleu.corpus_chrf(out, [gold]).score
    print("\n=== fair baseline on te_style test (same references) ===")
    print(f"IndicTrans2 en->te   chrF = {chrf:5.1f}   te-ratio = {sum(map(te_ratio, out))/len(out):.3f}")
    print(f"Our mT5 (reported)   chrF =  72.0   te-ratio = 0.314")
    print("\nIf IndicTrans2's te-ratio is much higher (~0.9), it over-translates the technical")
    print("terms - that is the register gap, visible even where chrF is close.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
