"""Quiz generation from the knowledge graph (no model needed).

Three question types, all built from the student's own notes:

* ``cloze``  - a definition with the term blanked out; pick the term.
* ``which_def`` - given a term, pick its definition among other concepts' definitions.
* ``tf``  - a definition sentence, either intact (true) or with its term swapped for a sibling concept (false).

Distractors come from the same section first (so they are plausible), and never from a concept whose
name contains / is contained in the answer (e.g. "Energy" vs "Energy signal").
"""

from __future__ import annotations

import random
import re
from typing import Any

from indiclearn.studio.i18n import tr

ANAPHORA = re.compile(r"(?i)^\s*(this|these|that|those|it|they|such|here)\b")


def _pattern(label: str, aliases: list[str]) -> re.Pattern[str]:
    names = {label, re.sub(r"\s*\([^)]*\)\s*$", "", label).strip(), *aliases}
    parts = []
    for nm in sorted((n for n in names if n and len(n) > 2), key=len, reverse=True):
        esc = re.escape(nm).replace(r"\-", "\x00").replace(r"\ ", "\x01").replace("\x00", r"[\s\-]").replace("\x01", r"[\s\-]+")
        parts.append(esc)
    return re.compile(r"(?i)\b(?:" + "|".join(parts) + r")(?:s|es)?\b") if parts else re.compile(r"(?!x)x")


def candidates(nodes: list[dict[str, Any]], scope: str | None = None) -> list[dict[str, Any]]:
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for n in nodes:
        by_parent.setdefault(n["parent"] or "", []).append(n)
    allowed: set[str] | None = None
    if scope and scope != "root":
        allowed = {scope}
        stack = [scope]
        while stack:
            for c in by_parent.get(stack.pop(), []):
                allowed.add(c["id"])
                stack.append(c["id"])
    out = []
    for n in nodes:
        if n["type"] == "doubt" or not n.get("definition"):
            continue
        d = n["definition"].strip()
        if not (30 <= len(d) <= 300) or ANAPHORA.match(d):
            continue
        if allowed is not None and n["id"] not in allowed and n["section_id"] not in allowed:
            continue
        out.append(n)
    return out


def _distractors(target: dict[str, Any], pool: list[dict[str, Any]], k: int, rng: random.Random, key: str = "label") -> list[dict[str, Any]]:
    tl = target["label"].lower()
    ok = []
    for n in pool:
        if n["id"] == target["id"]:
            continue
        nl = n["label"].lower()
        if nl in tl or tl in nl:
            continue
        ok.append(n)
    same = [n for n in ok if n["section_id"] == target["section_id"]]
    rest = [n for n in ok if n["section_id"] != target["section_id"]]
    rng.shuffle(same)
    rng.shuffle(rest)
    return (same + rest)[:k]


def generate(nodes: list[dict[str, Any]], scope: str | None, n: int, ui_lang: str, seed: int | None = None, priority: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """Return question dicts (with answers - the service strips them before sending)."""
    rng = random.Random(seed)
    pool_all = candidates(nodes, None)
    pool = candidates(nodes, scope)
    if not pool:
        return []
    priority = priority or {}
    # weak concepts first, then random, so repeated quizzes drill what the student gets wrong
    order = sorted(pool, key=lambda x: (-priority.get(x["id"], 0.0), rng.random()))
    types = ["cloze", "which_def", "tf", "cloze", "tf"]
    qs: list[dict[str, Any]] = []
    used: set[str] = set()
    for node in order:
        if len(qs) >= n:
            break
        if node["id"] in used:
            continue
        for qt in [types[len(qs) % len(types)], "cloze", "which_def", "tf"]:
            q = _make(qt, node, pool_all, ui_lang, rng)
            if q:
                qs.append(q)
                used.add(node["id"])
                break
    return qs


def _make(qt: str, node: dict[str, Any], pool: list[dict[str, Any]], ui_lang: str, rng: random.Random) -> dict[str, Any] | None:
    d = node["definition"].strip()
    pat = _pattern(node["full_label"], node.get("aliases") or [])
    expl = {"definition": d, "label": node["full_label"], "page": node.get("page", 1)}
    if qt == "cloze":
        if not pat.search(d):
            return None
        blanked = pat.sub("_____", d, count=0)
        if blanked.count("_____") > 2 or len(blanked.replace("_____", "").strip()) < 20:
            return None
        ds = _distractors(node, pool, 3, rng)
        if len(ds) < 2:
            return None
        opts = [node["label"]] + [x["label"] for x in ds]
        rng.shuffle(opts)
        return {"qtype": "cloze", "node_id": node["id"], "prompt": tr(ui_lang, "q_cloze"), "stem": blanked, "options": opts, "answer": opts.index(node["label"]), "explain": expl}
    if qt == "which_def":
        ds = _distractors(node, pool, 3, rng)
        ds = [x for x in ds if x["definition"][:40] != d[:40]]
        if len(ds) < 2:
            return None
        hide = lambda s, nd: _pattern(nd["full_label"], nd.get("aliases") or []).sub("…", s)
        opts = [hide(d, node)] + [hide(x["definition"], x) for x in ds]
        correct = opts[0]
        rng.shuffle(opts)
        return {"qtype": "which_def", "node_id": node["id"], "prompt": tr(ui_lang, "q_def", term=node["label"]), "stem": "", "options": opts, "answer": opts.index(correct), "explain": expl}
    if qt == "tf":
        if not pat.search(d):
            return None
        make_false = rng.random() < 0.5
        stem = d
        if make_false:
            ds = _distractors(node, pool, 1, rng)
            if not ds:
                return None
            stem = pat.sub(ds[0]["label"], d, count=1)
        opts = [tr(ui_lang, "true"), tr(ui_lang, "false")]
        return {"qtype": "tf", "node_id": node["id"], "prompt": tr(ui_lang, "q_tf"), "stem": stem, "options": opts, "answer": 1 if make_false else 0, "explain": expl}
    return None


def public(q: dict[str, Any]) -> dict[str, Any]:
    """Question without the answer / explanation (what the browser sees before answering)."""
    return {k: q[k] for k in ("id", "idx", "qtype", "prompt", "stem", "options", "node_id") if k in q}
