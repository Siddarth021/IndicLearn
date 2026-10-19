"""Step 5b - train on a free Kaggle GPU and bring only the final models back.

Needs the Kaggle API token. Put kaggle.json in ONE of these places (never commit it):
    <project>/secrets/kaggle.json      (git-ignored)      or      ~/.kaggle/kaggle.json

    python training/kaggle_run.py train            # upload data + code, train M1-M4 on a GPU, download models
    python training/kaggle_run.py train --only m4  # one model
    python training/kaggle_run.py gemma            # benchmark (and optionally QLoRA-tune) Gemma 4 E2B on a GPU
    python training/kaggle_run.py status|fetch     # check a running job / download its results again

What it does: (1) versions a private Kaggle dataset "<user>/indiclearn-train" with training/datasets +
the training code, (2) pushes a private GPU notebook that runs it, (3) polls until it finishes,
(4) downloads the output into models/trained/ (models) and training/reports/ (metrics, samples).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATASET_SLUG = "indiclearn-train"
KERNELS = {"train": "indiclearn-train-m1-m4", "gemma": "indiclearn-gemma-bench",
           "finetune": os.environ.get("KAGGLE_FT_SLUG", "indiclearn-gemma-finetune")}


def setup_token() -> str:
    """Copy secrets/kaggle.json to ~/.kaggle (where the CLI looks), 600 permissions. Returns the username."""
    home = Path.home() / ".kaggle" / "kaggle.json"
    for src in (ROOT / "secrets" / "kaggle.json", ROOT / "kaggle.json"):
        if src.exists() and not home.exists():
            home.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, home)
    if not home.exists():
        sys.exit("kaggle.json not found - put it in secrets/kaggle.json (see the docstring)")
    try:
        os.chmod(home, 0o600)
    except OSError:
        pass
    return json.loads(home.read_text())["username"]


def kg(*args: str, check: bool = True) -> str:
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout + r.stderr


def push_dataset(user: str) -> None:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        shutil.copytree(HERE / "datasets", d / "datasets")
        for f in ("train.py", "gemma_bench.py", "gemma_finetune.py"):
            if (HERE / f).exists():
                shutil.copy(HERE / f, d / f)
        shutil.copy(ROOT / "src" / "indiclearn" / "studio" / "brain.py", d / "brain.py")  # the app's real prompts
        if (HERE / "data" / "eval").exists():
            shutil.copytree(HERE / "data" / "eval", d / "eval")
        (d / "dataset-metadata.json").write_text(json.dumps({"title": "IndicLearn training data", "id": f"{user}/{DATASET_SLUG}",
                                                              "licenses": [{"name": "CC-BY-SA-4.0"}]}))
        exists = DATASET_SLUG in kg("datasets", "list", "--mine", "-s", DATASET_SLUG, check=False)
        if exists:
            print(kg("datasets", "version", "-p", str(d), "-m", time.strftime("update %Y-%m-%d %H:%M"), "--dir-mode", "zip", "-q"))
        else:
            print(kg("datasets", "create", "-p", str(d), "--dir-mode", "zip", "-q"))
    # wait until Kaggle has processed the new version
    for _ in range(60):
        st = kg("datasets", "status", f"{user}/{DATASET_SLUG}", check=False)
        if "ready" in st.lower():
            return
        time.sleep(10)


KERNEL_TRAIN = r'''
import os, subprocess, glob, shutil, sys
def find_src():
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train.py" in files and "brain.py" in files:
            return root
    raise SystemExit("dataset not mounted: " + str(list(os.walk("/kaggle/input"))[:5]))
src = find_src()
print("data at", src, os.listdir(src))
os.environ["TRAIN_OUT"] = "/kaggle/working/trained"
cmd = [sys.executable, f"{src}/train.py", "--data", f"{src}/datasets", "--out", "/kaggle/working/trained", "--model", "__ONLY__"]
print(" ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
'''

# The scripts themselves are embedded in the kernel (always the current version): a freshly versioned dataset
# is sometimes not mounted yet when the kernel starts, which silently ran stale code before.
KERNEL_FINETUNE = r'''
import base64, hashlib, json, os, shutil, subprocess, sys
CODE = json.loads(base64.b64decode("__CODE__").decode())
def find_src():
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train.py" in files and "brain.py" in files:
            return root
    raise SystemExit("dataset not mounted")
data = find_src()
src = "/tmp/src"
shutil.copytree(data, src, dirs_exist_ok=True)
for name, text in CODE.items():
    open(f"{src}/{name}", "w", encoding="utf-8").write(text)
    print("code", name, hashlib.md5(text.encode()).hexdigest()[:8], flush=True)
subprocess.run([sys.executable, f"{src}/gemma_finetune.py", "--src", src, "--out", "/kaggle/working"], check=True)
'''

KERNEL_GEMMA = r'''
import os, subprocess, glob, shutil, sys
def find_src():
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train.py" in files and "brain.py" in files:
            return root
    raise SystemExit("dataset not mounted: " + str(list(os.walk("/kaggle/input"))[:5]))
src = find_src()
subprocess.run([sys.executable, f"{src}/gemma_bench.py", "--src", src, "--out", "/kaggle/working"], check=True)
'''


def push_kernel(user: str, kind: str, only: str) -> str:
    slug = KERNELS[kind]
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        embedded = {n: (HERE / n).read_text(encoding="utf-8") for n in ("gemma_finetune.py", "gemma_bench.py")}
        embedded["brain.py"] = (ROOT / "src" / "indiclearn" / "studio" / "brain.py").read_text(encoding="utf-8")
        for n, t in embedded.items():
            print("embedding", n, hashlib.md5(t.encode()).hexdigest()[:8])
        b64 = base64.b64encode(json.dumps(embedded).encode()).decode()
        code = {"train": KERNEL_TRAIN.replace("__ONLY__", only), "gemma": KERNEL_GEMMA,
                "finetune": KERNEL_FINETUNE.replace("__CODE__", b64)}[kind]
        (d / "run.py").write_text(code)
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{slug}", "title": slug, "code_file": "run.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [f"{user}/{DATASET_SLUG}"], "competition_sources": [], "kernel_sources": []}))
        print(kg("kernels", "push", "-p", str(d)))
    return f"{user}/{slug}"


def wait(ref: str, poll: int = 60) -> str:
    t0 = time.time()
    while True:
        st = kg("kernels", "status", ref, check=False).lower()
        state = next((s for s in ("complete", "error", "cancel", "running", "queued") if s in st), "unknown")
        print(f"  [{int(time.time() - t0) // 60} min] {state}", flush=True)
        if state in ("complete", "error", "cancel"):
            return state
        time.sleep(poll)


def fetch(ref: str, kind: str) -> None:
    with tempfile.TemporaryDirectory() as td:
        print(kg("kernels", "output", ref, "-p", td, check=False)[-2000:])
        out = Path(td)
        reports = HERE / "reports"
        reports.mkdir(exist_ok=True)
        for f in list(out.rglob("*.log")) + list(out.rglob("reports/*.json")) + list(out.rglob("gemma_*.json")):
            shutil.copy(f, reports / f.name)
        trained = next(out.rglob("trained"), None) if kind == "train" else None
        if trained:
            dest = ROOT / "models" / "trained"
            names = {"m1": "m1_concept_filter", "m2": "m2_describe", "m3": "m3_relations", "m4": "m4_quiz"}
            for m, name in names.items():
                if (trained / m / "config.json").exists():
                    if (dest / name).exists():  # keep the previous model next to the new one until it is checked
                        old = ROOT / "old_useful_trash" / "models" / "trained" / (name + time.strftime("_%Y%m%d_%H%M"))
                        old.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(dest / name), str(old))
                    shutil.copytree(trained / m, dest / name)
                    print("installed", dest / name)
        gguf = next(out.rglob("gemma_te_tutor.Q4_K_M.gguf"), None)
        if gguf:  # the fine-tuned offline tutor, ready for: ollama create indiclearn-tutor -f Modelfile
            dest = ROOT / "models" / "trained" / "gemma_te_tutor"
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(gguf, dest / gguf.name)
            shutil.copy(next(out.rglob("Modelfile")), dest / "Modelfile")
            print("installed", dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["train", "gemma", "finetune", "status", "fetch"])
    ap.add_argument("--kind", default="train", choices=list(KERNELS))
    ap.add_argument("--only", default="all")
    ap.add_argument("--no-wait", action="store_true")
    a = ap.parse_args()
    user = setup_token()
    kind = a.action if a.action in KERNELS else a.kind
    ref = f"{user}/{KERNELS[kind]}"
    if a.action in KERNELS:
        push_dataset(user)
        ref = push_kernel(user, kind, a.only)
        if a.no_wait:
            return
        if wait(ref) == "complete":
            fetch(ref, kind)
        else:
            print(kg("kernels", "output", ref, "-p", str(HERE / "reports"), check=False)[-3000:])
    elif a.action == "status":
        print(kg("kernels", "status", ref, check=False))
    else:
        fetch(ref, kind)


if __name__ == "__main__":
    main()
