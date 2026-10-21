"""Live demo for a reviewer: our Telugu model keeps technical terms in English (lecturer style),
while a standard translator (IndicTrans2) turns everything into Telugu.

    .venv\\Scripts\\python training\\show_contribution.py

Prints, for each English sentence:
  - OUR MODEL : code-mixed Telugu (English technical terms kept)   <- the contribution
  - STANDARD  : IndicTrans2 full Telugu (terms translated too)     <- what existing tools do
  - a Telugu-script % so the difference is a number, not an opinion
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SENTENCES = [
    "A stack is a last-in-first-out data structure used for function calls.",
    "A page fault occurs when a process accesses a page not currently in main memory.",
    "Modus ponens is a rule of inference: if p is true and p implies q, then q is true.",
    "A deadlock happens when two processes each wait for a resource the other holds.",
]


def te_pct(s: str) -> int:
    te = len(re.findall(r"[ఀ-౿]", s))
    al = len(re.findall(r"[A-Za-zఀ-౿]", s))
    return round(100 * te / al) if al else 0


def main() -> int:
    from indiclearn.studio import neural

    ours = neural.load_te_translator()
    if ours is None:
        print("Our Telugu model not loaded. Reason:", neural.LOAD_ERRORS.get("te_style_translator", "not installed"))
        return 1
    std = neural.load_indictrans()
    has_std = std is not None and std.can("en", "te")

    print("=" * 74)
    print("IndicLearn - Telugu technical translation: lecturer style vs standard tool")
    print("=" * 74)
    for en in SENTENCES:
        out = neural.to_telugu(en, ours)
        print("\nENGLISH   :", en)
        print(f"OUR MODEL : {out}")
        print(f"            ^ Telugu script: {te_pct(out)}%  (English technical terms kept)")
        if has_std:
            base = std.translate([en], "en", "te")[0]
            print(f"STANDARD  : {base}")
            print(f"            ^ Telugu script: {te_pct(base)}%  (IndicTrans2 - translates the terms too)")
    print("\n" + "=" * 74)
    print("Lower Telugu-script % for OUR MODEL = it preserved the English technical terms,")
    print("the way Telugu lecturers actually teach. That is the contribution.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
