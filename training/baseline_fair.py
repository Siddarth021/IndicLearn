"""Fair baseline: run IndicTrans2 (en->te) on the SAME te_style test set and score chrF
against the SAME code-mixed references, so 'beats the baseline' is a like-for-like claim.

    .venv\\Scripts\\python training\\baseline_fair.py

Uses the app's own IndicTrans2 loader (the one check_ai reports as working), so it loads
the same way the running app does. Needs sacrebleu:  .venv\\Scripts\\pip install sacrebleu
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TEST = ROOT / "training" / "datasets" / "te_style" / "test.jsonl"


def te_ratio(s: str) -> float:
    te = len(re.findall(r"[ఀ-౿]", s))
    al = len(re.findall(r"[A-Za-zఀ-౿]", s))
    return te / al if al else 0.0


def main() -> int:
    try:
        import sacrebleu
    except ImportError:
        print("Run:  .venv\\Scripts\\pip install sacrebleu")
        return 1
    from indiclearn.studio import neural

    tr = neural.load_indictrans()
    if tr is None or not tr.can("en", "te"):
        print("IndicTrans2 could not load. Reason:", neural.LOAD_ERRORS.get("indictrans2"))
        return 1

    rows = [json.loads(l) for l in TEST.open(encoding="utf-8")]
    src = [(r.get("target_en") or r["input"].replace("to Telugu: ", "")) for r in rows]
    gold = [r["target"] for r in rows]

    print(f"items: {len(rows)} | translating with IndicTrans2 en->te ...")
    out: list[str] = []
    for i in range(0, len(src), 16):
        out.extend(tr.translate(src[i:i + 16], "en", "te"))
        print(f"  {min(i + 16, len(src))}/{len(src)}", end="\r", flush=True)
    print()

    chrf = sacrebleu.corpus_chrf(out, [gold]).score
    it_ratio = sum(map(te_ratio, out)) / len(out)
    print("\n=== fair baseline on te_style test (same 349 references) ===")
    print(f"IndicTrans2 en->te   chrF = {chrf:5.1f}   te-script-ratio = {it_ratio:.3f}")
    print(f"Our mT5 (reported)   chrF =  72.0   te-script-ratio = 0.314")
    print("\nHigher IndicTrans2 te-ratio (~0.9) = it over-translates the technical terms;")
    print("that register gap is the contribution, visible even if chrF is close.")
    (ROOT / "training" / "reports").mkdir(exist_ok=True)
    (ROOT / "training" / "reports" / "te_style__indictrans2_baseline.json").write_text(
        json.dumps({"n_test": len(rows), "chrF": round(chrf, 2), "telugu_script_ratio": round(it_ratio, 3),
                    "model": "ai4bharat/indictrans2-en-indic-dist-200M", "vs_ours_chrF": 72.0}, indent=2),
        encoding="utf-8")
    print("\nsaved training/reports/te_style__indictrans2_baseline.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
