"""Install the trained M1-M4 models from the zip downloaded from Kaggle.

1. Open https://www.kaggle.com/datasets/gundetisiddarth/indiclearn-models  ->  Download (zip, ~2 GB)
2. python scripts/install_models.py "C:\\Users\\<you>\\Downloads\\indiclearn-models.zip"
   (or: install_models.bat <zip>)

Old model folders are moved to old_useful_trash/models/ (never deleted). Restart the app afterwards.
"""

from __future__ import annotations

import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("m1_concept_filter", "m2_describe", "m3_relations", "m4_quiz")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    z = Path(sys.argv[1])
    dest = ROOT / "models" / "trained"
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
        for m in MODELS:
            if not any(n.startswith(m + "/") for n in names):
                continue
            if (dest / m).exists():
                old = ROOT / "old_useful_trash" / "models" / f"{m}_{time.strftime('%Y%m%d_%H%M')}"
                old.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dest / m), str(old))
            for n in names:
                if n.startswith(m + "/") and not n.endswith("/"):
                    zf.extract(n, dest)
            print("installed", dest / m)
        for n in names:
            if n.startswith("reports/") and n.endswith(".json"):
                (ROOT / "training" / "reports").mkdir(parents=True, exist_ok=True)
                (ROOT / "training" / "reports" / Path(n).name).write_bytes(zf.read(n))
    print("done - restart the app (run.bat)")


if __name__ == "__main__":
    main()
