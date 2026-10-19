"""Find out in 10 seconds why the AI answers/translations are not working.

    python scripts/check_ai.py

Prints, for every engine (Gemini, Sarvam, OpenAI-compatible, Anthropic, local Ollama/Gemma): whether a key was
found, and the result of a tiny live call with the exact error if it fails. Also reports OCR and the offline models.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)


def main() -> int:
    from indiclearn.studio.__main__ import load_dotenv

    n = load_dotenv()
    env_file = ROOT / ".env"
    print(f".env file: {'found, ' + str(n) + ' setting(s) loaded' if env_file.exists() else 'NOT FOUND at ' + str(env_file) + '  (copy .env.example to .env and add your key)'}")
    for k in ("INDICLEARN_GEMINI_API_KEY", "INDICLEARN_GEMINI_API_KEY_2", "INDICLEARN_SARVAM_API_KEY", "INDICLEARN_OPENAI_API_KEY", "INDICLEARN_ANTHROPIC_API_KEY"):
        v = os.environ.get(k, "")
        print(f"  {k:32s}", f"set ({len(v)} chars)" if v else "empty")

    from indiclearn.studio import neural, ocr
    from indiclearn.studio.brain import Brain

    brain = Brain.from_env()
    print("\nAI chain:", " -> ".join(brain.status()["chain"]))
    print("\nLive test of each engine:")
    for r in brain.diagnose():
        print(f"  {'OK  ' if r['ok'] else 'FAIL'} {r['provider']:10s} {r['model']:22s} {r['seconds']}s  {r['error']}")
    if not brain.providers:
        print("  (no engine configured)")

    print("\nOCR (text inside pictures):", ocr.status())
    st = neural.status()
    print("Offline models:", {k: st[k] for k in ("torch", "device", "indictrans2", "asr_te", "tts")})
    if st.get("load_errors"):
        print("  problems:", st["load_errors"])
    if not st["torch"]:
        print("  PyTorch is missing in THIS Python. Run setup_offline.bat (it installs into the project's .venv).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
