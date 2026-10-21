"""Set KEY=value pairs in .env (created from .env.example if missing). Usage: set_env.py KEY=value ..."""
import shutil, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env = root / ".env"
if not env.exists():
    shutil.copy(root / ".env.example", env)
lines = env.read_text(encoding="utf-8").splitlines()
for pair in sys.argv[1:]:
    k, v = pair.split("=", 1)
    for i, ln in enumerate(lines):
        if ln.split("=", 1)[0].strip() == k:
            lines[i] = f"{k}={v}"
            break
    else:
        lines.append(f"{k}={v}")
env.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("updated .env:", ", ".join(sys.argv[1:]))
