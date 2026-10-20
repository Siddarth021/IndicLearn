"""Draw the lesson figures into PNGs, so they can be looked at while they are being written.

    python scripts/preview_diagrams.py                 # every registered figure of DSMA
    python scripts/preview_diagrams.py posets lattices # only these topics
    python scripts/preview_diagrams.py --subject os --out /tmp/figs

Each figure becomes <out>/<topic>.png, drawn by the same Chromium the app runs in, with the real theme
colours. A figure that fails to draw is reported instead of silently skipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from indiclearn.studio.courses import diagrams  # noqa: E402

THEME = (ROOT / "frontend" / "css" / "theme.css").read_text(encoding="utf-8")
PAGE = """<!doctype html><meta charset="utf-8"><style>%s
body{margin:0;padding:18px;background:var(--bg);font-family:var(--sans)}
.fig{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:14px;width:760px}
.cap{font-size:12px;color:var(--muted);margin:0 0 8px}</style>
<div class="fig" id="fig"><p class="cap">%s</p>%s</div>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("topics", nargs="*")
    ap.add_argument("--subject", default="dsma")
    ap.add_argument("--out", default=str(ROOT / "build" / "figs"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = a.topics or sorted(diagrams.have(a.subject))
    if not todo:
        print("no figures registered for", a.subject)
        return
    from playwright.sync_api import sync_playwright

    ok, bad = [], []
    with sync_playwright() as p:
        br = p.chromium.launch()
        page = br.new_page(viewport={"width": 820, "height": 900}, device_scale_factor=2)
        for t in todo:
            s = diagrams.render(a.subject, t)
            if not s:
                bad.append((t, "render() returned None"))
                continue
            page.set_content(PAGE % (THEME, t, s))
            page.wait_for_timeout(120)
            el = page.query_selector("#fig")
            el.screenshot(path=str(out / f"{t}.png"))
            ok.append(t)
        br.close()
    print(f"drew {len(ok)} figure(s) into {out}")
    for t, why in bad:
        print("  FAILED", t, "-", why)


if __name__ == "__main__":
    main()
