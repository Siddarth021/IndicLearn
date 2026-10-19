"""Step 1 - collect the B.Tech engineering corpus from Wikipedia (CC BY-SA 4.0).

For every topic in training/topics.json (28 B.Tech subjects, ~460 core topics) this downloads the
English article as plain text, split into sections, plus the Telugu and Hindi versions of the
article when they exist (used for multilingual definitions).

Run (any machine with internet; standard library only, ~15 minutes, polite 1 request/second):

    python training/collect_wikipedia.py            # resumes where it stopped
    python training/collect_wikipedia.py --limit 20 # quick try

Output: training/data/raw/wikipedia.jsonl  - one article per line:
    {"title", "subject", "url", "sections": [{"heading", "text"}], "indic": {"te": {"title", "intro"}, "hi": {...}}}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "raw" / "wikipedia.jsonl"
UA = "IndicLearn-research/1.0 (B.Tech study app dataset; contact: gundetisiddarth@gmail.com)"
SKIP_SECTIONS = re.compile(r"^(see also|references|external links|further reading|notes|bibliography|sources|citations|footnotes|gallery|"
                           r"in popular culture|literature|works cited)$", re.IGNORECASE)


def api(lang: str, **params: str) -> dict:
    params = {"format": "json", "formatversion": "2", "maxlag": "5", **params}
    url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # 429 / maxlag / network: back off
            time.sleep(3 * (attempt + 1))
            err = e
    raise RuntimeError(f"{lang}: {params.get('titles')}: {err}")


def clean(text: str) -> str:
    """Wikipedia plain text keeps LaTeX fallbacks like '{\\displaystyle f(t)}' - drop them and tidy spaces."""
    text = re.sub(r"\{\\displaystyle[^\n]*?\}\s*(?=\n|$)", " ", text)
    lines = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s or "\\displaystyle" in s or re.fullmatch(r"[{}()\[\]^_=+\-*/\\.,;:|\w\s]{0,3}", s):
            continue
        lines.append(s)
    out = " ".join(lines)
    out = re.sub(r"\s+([,.;:)])", r"\1", out)
    return re.sub(r"\s{2,}", " ", out).strip()


def sections(extract: str) -> list[dict[str, str]]:
    parts = re.split(r"\n(={2,4})\s*(.+?)\s*\1\n", "\n" + extract)
    out = [{"heading": "Introduction", "text": clean(parts[0])}]
    for i in range(1, len(parts) - 2, 3):
        heading, body = parts[i + 1].strip(), parts[i + 2]
        if SKIP_SECTIONS.match(heading):
            continue
        out.append({"heading": heading, "text": clean(body)})
    return [s for s in out if len(s["text"]) >= 200]


def fetch(title: str) -> dict | None:
    d = api("en", action="query", prop="extracts|langlinks|info", explaintext="1", exsectionformat="wiki", redirects="1",
            titles=title, lllimit="max", inprop="url")
    pages = d.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return None
    p = pages[0]
    links = {ll["lang"]: ll["title"] for ll in p.get("langlinks", []) if ll.get("lang") in ("te", "hi")}
    indic = {}
    for lang, t in links.items():
        time.sleep(0.5)
        q = api(lang, action="query", prop="extracts", exintro="1", explaintext="1", redirects="1", titles=t)
        pg = (q.get("query", {}).get("pages") or [{}])[0]
        intro = clean(pg.get("extract") or "")
        if len(intro) >= 40:
            indic[lang] = {"title": pg.get("title", t), "intro": intro[:1500]}
    return {"title": p["title"], "url": p.get("fullurl", ""), "sections": sections(p.get("extract") or ""), "indic": indic}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    topics = json.loads((HERE / "topics.json").read_text(encoding="utf-8"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        done = {json.loads(ln)["query"] for ln in OUT.read_text(encoding="utf-8").split("\n") if ln.strip()}
    todo = [(s, t) for s, ts in topics.items() for t in ts if t not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(done)} done, {len(todo)} to fetch", flush=True)
    with OUT.open("a", encoding="utf-8") as fh:
        for i, (subject, title) in enumerate(todo, 1):
            try:
                art = fetch(title)
            except Exception as e:
                print(f"  ! {title}: {e}", file=sys.stderr, flush=True)
                continue
            if art is None:
                print(f"  ? missing: {title}", flush=True)
                art = {"title": title, "sections": [], "indic": {}}
            fh.write(json.dumps({"query": title, "subject": subject, **art}, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
            time.sleep(1.0)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
