"""One command to make IndicLearn work with NO internet and NO API key.

    python scripts/setup_offline.py              # recommended set for a 4 GB GPU (RTX 3050)
    python scripts/setup_offline.py --quality    # bigger local LLM (gemma4:e4b) + Telugu Whisper-medium
    python scripts/setup_offline.py --check      # only report what is ready

What it sets up (all free, run on this computer):

  task                              model                                         licence
  --------------------------------  --------------------------------------------  -----------
  explanations, reasoning, map,     Gemma 4 E2B (or E4B) via Ollama               Apache-2.0
  "convert to", question rewrite
  translation without the LLM       IndicTrans2 200M en<->indic (gated, MIT)      MIT
  cross-lingual search              multilingual-e5-small                          MIT
  Telugu speech -> text             Whisper-small fine-tuned on Telugu             Apache-2.0
  other speech -> text              Whisper-small                                  Apache-2.0
  text -> speech                    MMS-TTS te/hi/en (+ Indic Parler-TTS optional) CC-BY-NC / Apache-2.0

The app picks all of these up automatically on its next start. Hosted keys in .env (if any) are
still tried first; these models take over whenever there is no key or the internet/provider is down.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

BASE_MODELS = ["e5_small", "whisper_small", "whisper_te_small", "mms_tts_tel", "mms_tts_hin", "mms_tts_eng",
               "indictrans2_en_indic", "indictrans2_indic_en"]


def ollama_up(url: str = "http://127.0.0.1:11434") -> list[str] | None:
    try:
        with urllib.request.urlopen(url + "/api/tags", timeout=2) as r:
            return [m.get("name", "") for m in json.loads(r.read()).get("models", [])]
    except (OSError, ValueError):  # not running / not Ollama
        return None


def step(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(3, 70 - len(title)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quality", action="store_true", help="gemma4:e4b + whisper_te_medium (slower on a 4 GB GPU)")
    ap.add_argument("--parler", action="store_true", help="also fetch Indic Parler-TTS (gated; commercial-friendly voices)")
    ap.add_argument("--nllb", action="store_true", help="also fetch NLLB-200 600M (non-commercial translation fallback)")
    ap.add_argument("--check", action="store_true", help="report only, download nothing")
    a = ap.parse_args()
    llm = "gemma4:e4b" if a.quality else "gemma4:e2b"

    step("1/3  Local LLM (Ollama)")
    exe = shutil.which("ollama")
    tags = ollama_up()
    if not exe and tags is None:
        print("Ollama is not installed. Install it (free) from https://ollama.com/download, start it, then rerun.")
    elif tags is None:
        print("Ollama is installed but not running. Start the Ollama app (or `ollama serve`) and rerun.")
    elif any(t.startswith(llm) for t in tags):
        print(f"OK - {llm} is already pulled.")
    elif a.check:
        print(f"Missing - run: ollama pull {llm}")
    else:
        print(f"Pulling {llm} (about 3-5 GB, once)...")
        subprocess.call([exe or "ollama", "pull", llm])

    step("2/3  Offline speech, translation and search models (Hugging Face, once)")
    keys = list(BASE_MODELS)
    if a.quality:
        keys.append("whisper_te_medium")
    if a.parler:
        keys.append("indic_parler_tts")
    if a.nllb:
        keys.append("nllb_600m")
    cmd = [PY, str(ROOT / "scripts" / "download_models.py"), "--only", *keys]
    if a.check:
        cmd.append("--check")
    print("running:", " ".join(cmd[1:]))
    subprocess.call(cmd, cwd=str(ROOT))
    print("Gated models (IndicTrans2, Parler) are skipped until you accept their terms on huggingface.co and log in:")
    print('  python -c "from huggingface_hub import login; login()"')

    step("3/3  What the app will use")
    sys.path.insert(0, str(ROOT / "src"))
    from indiclearn.studio import neural
    from indiclearn.studio.brain import Brain

    st = neural.status()
    br = Brain.from_env()
    print("LLM chain      :", " -> ".join(br.status()["chain"]))
    print("Telugu speech  :", st.get("asr_te") or "not installed (generic Whisper or Chrome speech will be used)")
    print("Speech (other) :", "whisper_small" if st.get("asr") else "not installed")
    nllb = (neural.models_dir() / "pretrained" / "nllb_600m" / "config.json").exists()
    print("Translation    :", "IndicTrans2" if st.get("indictrans2") else "NLLB-200" if nllb else "LLM only (no local translator downloaded)")
    print("Voices         :", ", ".join(st.get("tts") or []) or "browser voices only", "+ Parler" if st.get("parler_tts") else "")
    print("Device         :", st.get("device"), "| torch installed:", st.get("torch"))
    if not st.get("torch"):
        print("\nInstall PyTorch for the speech/translation models (CUDA build for an RTX card):")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print("  pip install -r requirements-models.txt")
    print("\nDone. Start the app with run.bat - it uses these automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
