"""Download the PRETRAINED helper models the app runs offline into models/pretrained/<key>/.

    python scripts/download_models.py                    # the default set
    python scripts/download_models.py --only whisper_te_small e5_small
    python scripts/download_models.py --all              # + optional ones
    python scripts/download_models.py --check            # access check only, downloads nothing

(The four TRAINED models M1-M4 come from training/ - see scripts/install_models.py.)
Gated repos (IndicTrans2, Indic Parler-TTS) need one click on the model page while signed in to
Hugging Face, then once on this machine:  python -c "from huggingface_hub import login; login()"
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRETRAINED = ROOT / "models" / "pretrained"


@dataclass(frozen=True)
class Spec:
    key: str
    repo: str
    role: str
    licence: str
    gated: bool = False
    optional: bool = False


MODELS = [
    Spec("e5_small", "intfloat/multilingual-e5-small", "cross-lingual search (Telugu question -> English passage)", "mit"),
    Spec("whisper_te_small", "vasista22/whisper-telugu-small", "offline Telugu speech to text (FLEURS-te WER 11.6)", "apache-2.0"),
    Spec("whisper_small", "openai/whisper-small", "offline speech to text, other languages", "apache-2.0"),
    Spec("indictrans2_en_indic", "ai4bharat/indictrans2-en-indic-dist-200M", "offline English -> Indic translation", "mit", gated=True),
    Spec("indictrans2_indic_en", "ai4bharat/indictrans2-indic-en-dist-200M", "offline Indic -> English translation", "mit", gated=True),
    Spec("mms_tts_tel", "facebook/mms-tts-tel", "offline Telugu voice (non-commercial licence)", "cc-by-nc-4.0"),
    Spec("mms_tts_hin", "facebook/mms-tts-hin", "offline Hindi voice (non-commercial licence)", "cc-by-nc-4.0"),
    Spec("mms_tts_eng", "facebook/mms-tts-eng", "offline English voice (non-commercial licence)", "cc-by-nc-4.0"),
    Spec("whisper_te_medium", "vasista22/whisper-telugu-medium", "more accurate Telugu speech to text (~3 GB)", "apache-2.0", optional=True),
    Spec("nllb_600m", "facebook/nllb-200-distilled-600M", "translation fallback, 13 languages (non-commercial)", "cc-by-nc-4.0", optional=True),
    Spec("indic_parler_tts", "ai4bharat/indic-parler-tts", "natural offline Indic voices", "apache-2.0", gated=True, optional=True),
]
BY_KEY = {m.key: m for m in MODELS}


def check(spec: Spec) -> tuple[bool, str]:
    from huggingface_hub import HfApi, get_token
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    if spec.gated and not get_token():
        return False, f"GATED - accept the terms at https://hf.co/{spec.repo} and log in (see the top of this file)"
    try:
        HfApi().auth_check(spec.repo)
        return True, "ok"
    except GatedRepoError:
        return False, f"GATED - press 'Agree and access' at https://hf.co/{spec.repo}, then retry"
    except RepositoryNotFoundError:
        return False, "not found"
    except Exception as e:  # network / proxy
        return False, f"{type(e).__name__}: {str(e)[:160]}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", metavar="KEY")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("pip install huggingface_hub")
    if a.only:
        bad = [k for k in a.only if k not in BY_KEY]
        if bad:
            sys.exit(f"unknown key(s) {bad}; available: {', '.join(BY_KEY)}")
        picked = [BY_KEY[k] for k in a.only]
    else:
        picked = [m for m in MODELS if a.all or not m.optional]
    failed, skipped = [], []
    for spec in picked:
        ok, msg = check(spec)
        print(f"{spec.key:<22} {spec.licence:<13} {msg}")
        if not ok:
            (skipped if spec.gated else failed).append(f"{spec.key}: {msg}")
            continue
        if not a.check:
            try:
                snapshot_download(repo_id=spec.repo, local_dir=str(PRETRAINED / spec.key),
                                  ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.tflite", "onnx/*", "openvino/*", "pytorch_model.bin*"]
                                  if spec.key in ("e5_small",) else ["*.msgpack", "*.h5", "*.ot", "*.tflite", "onnx/*", "openvino/*"])
            except Exception as e:
                failed.append(f"{spec.key}: {e}")
    for title, items in (("Skipped (gated, access not granted yet)", skipped), ("Failed", failed)):
        if items:
            print(f"\n{title}:\n  - " + "\n  - ".join(items))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
