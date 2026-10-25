"""Translate English sentences into classroom-style Telugu (Telugu grammar, English technical terms kept)
using the IndicLearn model in models/trained/te_style_translator.

    .venv\\Scripts\\python training\\translate.py                      # interactive: type a sentence, press Enter
    .venv\\Scripts\\python training\\translate.py "A stack is a LIFO data structure."
    .venv\\Scripts\\python training\\translate.py --file notes.txt     # one sentence/paragraph per line
    .venv\\Scripts\\python training\\translate.py --cpu                # if the GPU is busy (Gemma / the app running)

Formulas written between $...$ and code between `...` are protected: they come back exactly as typed.
Type 'quit' (or press Ctrl+C) to leave interactive mode.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def te_pct(s: str) -> int:
    te = len(re.findall(r"[ఀ-౿]", s))
    al = len(re.findall(r"[A-Za-zఀ-౿]", s))
    return round(100 * te / al) if al else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="English -> classroom-style Telugu")
    ap.add_argument("text", nargs="*", help="sentence to translate (omit for interactive mode)")
    ap.add_argument("--file", help="text file: translate it line by line")
    ap.add_argument("--cpu", action="store_true", help="run on the CPU (slower; use if the GPU is out of memory)")
    a = ap.parse_args()
    if a.cpu:
        os.environ["INDICLEARN_FORCE_CPU"] = "1"

    from indiclearn.studio import neural

    print("Loading the translator (first time takes ~10 s)...")
    model = neural.load_te_translator()
    if model is None:
        print("The Telugu model is not installed or could not load.")
        print("Reason:", neural.LOAD_ERRORS.get("te_style_translator", "models/trained/te_style_translator is missing"))
        return 1
    print(f"Ready on {neural._device()}.\n")

    def show(en: str) -> None:
        te = neural.to_telugu(en, model)
        print(f"  English : {en}")
        print(f"  Telugu  : {te}")
        print(f"            (Telugu script: {te_pct(te)}% - the rest is English technical terms kept on purpose)\n")

    if a.file:
        for line in Path(a.file).read_text(encoding="utf-8").splitlines():
            if line.strip():
                show(line.strip())
    elif a.text:
        show(" ".join(a.text))
    else:
        print("Type an English sentence and press Enter ('quit' to exit).\n")
        while True:
            try:
                en = input("English> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if en.lower() in ("quit", "exit", "q"):
                break
            if en:
                show(en)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
