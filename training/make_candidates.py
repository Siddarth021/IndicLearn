"""Step 2 - run the app's own rule-based map builder over every corpus article.

The trained models must learn on exactly what they will see inside the app, so each Wikipedia article
is turned into a small Markdown "document" and passed through ``build_knowledge_graph`` (rules only, no
LLM, no trained model). Its concept nodes are the M1 candidates; its sections give M3/M4 context.

Run:  python training/make_candidates.py
Out:  training/data/interim/candidates.jsonl - one article per line:
      {"title", "subject", "sections": [...], "candidates": [{"id", "label", "context", "section"}]}
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
os.environ.setdefault("INDICLEARN_USE_M1", "0")  # never let an old model decide its own training data
os.environ.setdefault("INDICLEARN_USE_M2", "0")

from indiclearn.studio.kgraph import build_knowledge_graph  # noqa: E402
from indiclearn.studio.parse import parse_file  # noqa: E402

RAW = HERE / "data" / "raw" / "wikipedia.jsonl"
OUT = HERE / "data" / "interim" / "candidates.jsonl"
MAX_SECTIONS = 14  # long articles: the first sections carry the core ideas


def to_markdown(art: dict) -> str:
    lines = [f"# {art['title']}", ""]
    for s in art["sections"][:MAX_SECTIONS]:
        lines += [f"## {s['heading']}", "", s["text"], ""]
    return "\n".join(lines)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_art = n_cand = 0
    with OUT.open("w", encoding="utf-8") as fo, tempfile.TemporaryDirectory() as td:
        for line in RAW.read_text(encoding="utf-8").split("\n"):
            art = json.loads(line)
            if not art.get("sections"):
                continue
            md = Path(td) / "doc.md"
            md.write_text(to_markdown(art), encoding="utf-8")
            kg = build_knowledge_graph(parse_file(md, art["title"] + ".md"))
            by_id = {n["id"]: n for n in kg.nodes}
            cands = []
            for n in kg.nodes:
                if n["type"] != "concept":
                    continue
                ctx = n.get("definition") or n.get("summary") or ""
                sec = by_id.get(n.get("section_id") or n.get("parent") or "", {}).get("label", "")
                cands.append({"id": n["id"], "label": n["full_label"], "context": ctx[:400], "section": sec})
            fo.write(json.dumps({"title": art["title"], "subject": art["subject"], "sections": art["sections"][:MAX_SECTIONS],
                                 "indic": art.get("indic", {}), "candidates": cands}, ensure_ascii=False) + "\n")
            n_art += 1
            n_cand += len(cands)
    print(f"{n_art} articles, {n_cand} candidate concepts -> {OUT}")


if __name__ == "__main__":
    main()
