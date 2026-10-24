"""Install the Telugu models from a Kaggle download (a folder or the .zip from the notebook's Output tab).

    python scripts/install_telugu_models.py D:\\Downloads\\indiclearn-indic-m2-m4-output.zip

Finds te_style / m2_te / m4_te inside it (any depth) and installs them into models/trained.
Nothing is deleted: a model that is already installed is moved to old_useful_trash/.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = {"te_plus": "te_style_translator", "te_style": "te_style_translator", "m2_te": "m2_describe_te", "m4_te": "m4_quiz_te"}


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    src = Path(sys.argv[1])
    work = None
    if src.suffix.lower() == ".zip":
        work = Path(tempfile.mkdtemp(prefix="telugu_", dir=ROOT / ".tmp" if (ROOT / ".tmp").exists() else None))
        print("unpacking", src, "->", work)
        with zipfile.ZipFile(src) as z:
            z.extractall(work)
        src = work
    found = 0
    done = set()
    for name, dest_name in TARGET.items():
        if dest_name in done:  # te_plus (newer) wins over te_style for the same slot
            continue
        hit = next((p for p in src.rglob(name) if (p / "config.json").exists() and (p / "model.safetensors").exists()), None)
        if not hit:
            continue
        dest = ROOT / "models" / "trained" / dest_name
        if dest.exists():
            old = ROOT / "old_useful_trash" / "models" / "trained" / (dest_name + time.strftime("_%Y%m%d_%H%M"))
            old.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dest), str(old))
        shutil.copytree(hit, dest)
        print("installed", dest, "<-", name)
        done.add(dest_name)
        found += 1
    if not found:
        print("No te_style / m2_te / m4_te folder with a model.safetensors found in", src)
        return 1
    if work:
        print("You can delete", work, "and the zip.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
