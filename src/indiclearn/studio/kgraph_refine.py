"""Make the knowledge map trustworthy: fewer junk nodes, no duplicates, prerequisites that make sense.

Two passes run on the graph that :func:`indiclearn.studio.kgraph.build_knowledge_graph` produced:

1. :func:`clean_graph` - deterministic, always on, no model needed.
   * drops "concepts" that are really modifiers ("Self-supervised", "Fine-tuned", "Multi-label"):
     words that, in the notes, almost always sit in front of another noun;
   * merges near-duplicates ("Lightweight student", "... student model", "... student network");
   * keeps a prerequisite only if it points forward in reading order, joins two concepts (not a
     concept and a whole chapter), and does not close a cycle;
   * keeps at most three "related" links per concept, so the map stays readable.

2. :func:`llm_refine` - when the hybrid brain is configured.  The LLM reads the sections and
   proposes the concept list, one-line definitions and prerequisite / related links.  Its output is
   *validated* against the document (every concept must actually occur in the text, every link must
   join two known nodes, cycles are cut) and merged with the structural skeleton (chapters, sections,
   reading flow) that the parser built.  Node ids are reused when a concept already existed, so a
   student's progress survives a rebuild.  If the LLM output looks thin, the cleaned rule-based graph
   is kept.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter, defaultdict
from typing import Any

log = logging.getLogger("indiclearn.kgraph")

STRUCTURAL = ("root", "chapter", "section", "topic")
GENERIC_TAIL = {"model", "models", "network", "networks", "method", "methods", "approach", "technique", "techniques", "module", "framework", "scheme"}
_WORD = re.compile(r"[A-Za-z][A-Za-z\-]*")
_FUNC = {"is", "are", "was", "were", "of", "in", "on", "to", "and", "or", "for", "with", "by", "as", "at", "from", "the", "a", "an", "that", "which",
         "can", "has", "have", "be", "been", "it", "its", "this", "these", "we", "where", "when", "while", "than", "then", "into", "using", "used", "also"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\([^)]*\)", "", s or "")).strip().lower()


def _key(s: str) -> str:
    toks = [t for t in re.split(r"[\s\-]+", _norm(s)) if t]
    while len(toks) > 1 and toks[-1] in GENERIC_TAIL:
        toks.pop()
    return " ".join(t[:-1] if len(t) > 4 and t.endswith("s") and not t.endswith("ss") else t for t in toks)


def _occurrences(label: str, text: str) -> list[str]:
    """Words that follow each occurrence of *label* in *text* ('' at sentence end)."""
    pat = re.compile(r"(?<![A-Za-z])" + re.escape(label) + r"(?![A-Za-z])\s*([A-Za-z][A-Za-z\-]*)?", re.IGNORECASE)
    return [(m.group(1) or "").lower() for m in pat.finditer(text)]


def _is_modifier(n: dict[str, Any], text: str) -> bool:
    label = re.sub(r"\s*\([^)]*\)\s*$", "", n.get("full_label") or n["label"]).strip()
    words = label.split()
    if len(words) != 1 or "-" not in label and not label.lower().endswith("ed"):
        return False
    nxt = _occurrences(label, text)
    if len(nxt) < 2:
        return False
    mod = sum(1 for w in nxt if w and w not in _FUNC)
    return mod / len(nxt) >= 0.75


def _cut_cycles(prereqs: list[tuple[str, str]], order: dict[str, int]) -> list[tuple[str, str]]:
    """Add edges earliest-first; skip any edge that would close a cycle."""
    adj: dict[str, set[str]] = defaultdict(set)

    def reaches(a: str, b: str) -> bool:
        stack, seen = [a], set()
        while stack:
            x = stack.pop()
            if x == b:
                return True
            if x in seen:
                continue
            seen.add(x)
            stack.extend(adj[x])
        return False

    out = []
    for a, b in sorted(prereqs, key=lambda e: (order.get(e[0], 0), order.get(e[1], 0))):
        if a == b or reaches(b, a):
            continue
        adj[a].add(b)
        out.append((a, b))
    return out


def reading_order(nodes: list[dict[str, Any]], full_text: str) -> dict[str, int]:
    """Position of each node's first appearance in the text (page-based fallback)."""
    low = full_text.lower()
    out: dict[str, int] = {}
    for n in nodes:
        best = -1
        for nm in [n.get("full_label") or n["label"], re.sub(r"\s*\([^)]*\)\s*$", "", n.get("full_label") or n["label"]), *(n.get("aliases") or [])]:
            i = low.find(nm.lower()) if nm else -1
            if i >= 0 and (best < 0 or i < best):
                best = i
        out[n["id"]] = best if best >= 0 else (n.get("page") or 1) * 100000 + n.get("order", 0)
    return out


def compute_path(nodes: list[dict[str, Any]]) -> list[str]:
    order_of = {n["id"]: n.get("order", 0) for n in nodes}
    children: dict[str, list[str]] = defaultdict(list)
    for n in nodes:
        if n.get("parent"):
            children[n["parent"]].append(n["id"])
    path: list[str] = []

    def walk(nid: str) -> None:
        if nid != "root":
            path.append(nid)
        for ch in sorted(children.get(nid, []), key=lambda i: order_of.get(i, 0)):
            walk(ch)

    walk("root")
    return path


def _rebuild_edges(nodes: list[dict[str, Any]], old_edges: list[dict[str, Any]], prereqs: list[tuple[str, str]],
                   related: list[tuple[str, str, float]], whys: dict[tuple[str, str], str] | None = None) -> list[dict[str, Any]]:
    ids = {n["id"] for n in nodes}
    edges: list[dict[str, Any]] = []

    def add(src: str, dst: str, typ: str, w: float = 1.0, why: str = "") -> None:
        e = {"id": f"e{len(edges) + 1}", "source": src, "target": dst, "type": typ, "weight": round(float(w), 3)}
        if why:
            e["why"] = why
        edges.append(e)

    for n in nodes:
        if n.get("parent") in ids:
            add(n["parent"], n["id"], "contains")
    for e in old_edges:
        if e["type"] == "next" and e["source"] in ids and e["target"] in ids:
            add(e["source"], e["target"], "next")
    for a, b in prereqs:
        add(a, b, "prereq", 0.8, (whys or {}).get((a, b), ""))
    for a, b, w in related:
        add(a, b, "related", w)
    return edges


def _stats(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], base: dict[str, Any]) -> dict[str, Any]:
    st = dict(base)
    st["concepts"] = len([n for n in nodes if n["type"] == "concept"]) + len([n for n in nodes if n.get("is_term")])
    st["relations_prereq"] = len([e for e in edges if e["type"] == "prereq"])
    st["relations_related"] = len([e for e in edges if e["type"] == "related"])
    return st


# --------------------------------------------------------------------------- pass 1
def clean_graph(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], full_text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    report = Counter()
    by_id = {n["id"]: n for n in nodes}
    # 1) modifiers masquerading as concepts
    drop = {n["id"] for n in nodes if n["type"] == "concept" and _is_modifier(n, full_text)}
    report["dropped_modifiers"] = len(drop)
    # 2) near-duplicates: same key after removing generic tails ("model", "network"...) and plurals
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for n in nodes:
        if n["type"] == "concept" and n["id"] not in drop:
            groups[(n.get("section_id") or "", _key(n["label"]))].append(n)
    # also across sections when the full key matches exactly
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for n in nodes:
        if n["type"] == "concept" and n["id"] not in drop:
            by_key[_key(n["label"])].append(n)
    alias_of: dict[str, str] = {}
    for grp in list(groups.values()) + list(by_key.values()):
        grp = [g for g in grp if g["id"] not in alias_of]
        if len(grp) < 2:
            continue
        keep = max(grp, key=lambda g: (g.get("importance", 0), len(g.get("pages") or []), -len(g["label"])))
        for g in grp:
            if g["id"] == keep["id"]:
                continue
            alias_of[g["id"]] = keep["id"]
            keep["aliases"] = sorted({*(keep.get("aliases") or []), g["label"], *(g.get("aliases") or [])} - {keep["label"]})
            keep["pages"] = sorted({*(keep.get("pages") or []), *(g.get("pages") or [])})
            if not keep.get("definition") and g.get("definition"):
                keep["definition"], keep["definition_quality"] = g["definition"], g.get("definition_quality")
            keep["importance"] = max(keep.get("importance", 0), g.get("importance", 0))
    report["merged_duplicates"] = len(alias_of)
    gone = drop | set(alias_of)
    kept = [n for n in nodes if n["id"] not in gone]
    ids = {n["id"] for n in kept}
    for n in kept:  # re-home children of removed nodes
        while n.get("parent") in gone:
            n["parent"] = alias_of.get(n["parent"]) or by_id[n["parent"]].get("parent")
    order = reading_order(kept, full_text)

    def res(x: str) -> str:
        return alias_of.get(x, x)

    # a prerequisite joins two ideas - not an idea and a whole chapter
    concept_like = {n["id"] for n in kept if n["type"] == "concept" or (n.get("is_term") and n["type"] == "topic")}
    pre: list[tuple[str, str]] = []
    for e in edges:
        if e["type"] != "prereq":
            continue
        a, b = res(e["source"]), res(e["target"])
        if a not in ids or b not in ids or a == b:
            continue
        if a not in concept_like or b not in concept_like:
            report["dropped_prereq_to_chapter"] += 1
            continue
        if order.get(a, 0) > order.get(b, 0):
            report["dropped_backward_prereq"] += 1
            continue
        pre.append((a, b))
    pre = _cut_cycles(list(dict.fromkeys(pre)), order)
    whys = {(res(e["source"]), res(e["target"])): e["why"] for e in edges if e["type"] == "prereq" and e.get("why")}
    rel_count: Counter[str] = Counter()
    rel: list[tuple[str, str, float]] = []
    seen: set[tuple[str, str]] = set()
    pre_pairs = {frozenset(p) for p in pre}
    for e in sorted((e for e in edges if e["type"] == "related"), key=lambda e: -e.get("weight", 0)):
        a, b = res(e["source"]), res(e["target"])
        k = (min(a, b), max(a, b))
        if a not in ids or b not in ids or a == b or k in seen or frozenset(k) in pre_pairs:
            continue
        if rel_count[a] >= 3 or rel_count[b] >= 3:
            report["trimmed_related"] += 1
            continue
        seen.add(k)
        rel_count[a] += 1
        rel_count[b] += 1
        rel.append((a, b, e.get("weight", 0.5)))
    new_edges = _rebuild_edges(kept, edges, pre, rel, whys)
    for n in kept:
        n["n_children"] = sum(1 for m in kept if m.get("parent") == n["id"])
    return kept, new_edges, dict(report)


# --------------------------------------------------------------------------- pass 2
def _cid(name: str) -> str:
    return "c" + hashlib.sha1(_norm(name).encode()).hexdigest()[:9]


def llm_refine(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], chunks: list[dict[str, Any]], brain: Any,
               max_concepts: int = 45) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]] | None:
    """Ask the LLM for concepts + links, validate them against the text, merge with the skeleton."""
    struct = [n for n in nodes if n["type"] in STRUCTURAL]
    struct_ids = {n["id"] for n in struct}
    text_by_sec: dict[str, list[str]] = defaultdict(list)
    for c in chunks:
        text_by_sec[c.get("section_id") or "root"].append(c["text"])
    sections = [{"id": n["id"], "title": n.get("full_label") or n["label"], "text": " ".join(text_by_sec.get(n["id"], []))}
                for n in struct if n["type"] != "root" and text_by_sec.get(n["id"])]
    if not sections:
        return None
    # batch so a long book still fits the model's context; each batch gets its share of concepts
    batches: list[list[dict[str, str]]] = [[]]
    size = 0
    for s in sections:
        ln = min(len(s["text"]), 1800) + len(s["title"]) + 20
        if size + ln > 20000 and batches[-1]:
            batches.append([])
            size = 0
        batches[-1].append(s)
        size += ln
    batches = batches[:6]
    per = max(12, max_concepts // len(batches))
    raw_c: list[dict[str, Any]] = []
    raw_p: list[dict[str, Any]] = []
    raw_r: list[dict[str, Any]] = []
    engine = ""
    for b in batches:
        got = brain.extract_graph(b, per)
        if not got:
            continue
        engine = got.get("_engine", engine)
        raw_c += [x for x in got.get("concepts", []) if isinstance(x, dict) and isinstance(x.get("name"), str)]
        raw_p += [x for x in got.get("prereqs", []) if isinstance(x, dict)]
        raw_r += [x for x in got.get("related", []) if isinstance(x, dict)]
    if not raw_c:
        return None

    full = "\n".join(c["text"] for c in chunks)
    full_l = full.lower()
    old_concepts = [n for n in nodes if n["type"] == "concept" or n.get("is_term")]
    old_by_name: dict[str, dict[str, Any]] = {}
    for n in old_concepts:
        for nm in [n["label"], n.get("full_label") or "", *(n.get("aliases") or [])]:
            if nm:
                old_by_name.setdefault(_key(nm), n)
    struct_by_name = {_key(n.get("full_label") or n["label"]): n for n in struct}
    sec_of_chunk_pages: dict[str, list[int]] = defaultdict(list)

    new_nodes: dict[str, dict[str, Any]] = {}
    name_to_id: dict[str, str] = {}
    enriched = 0
    pending: list[tuple[dict[str, Any], str]] = []  # section nodes to enrich once the result is accepted
    order_base = max((n.get("order", 0) for n in nodes), default=0) + 1
    for i, c in enumerate(raw_c):
        name = c["name"].strip()[:80]
        aliases = [a.strip()[:60] for a in (c.get("aliases") or []) if isinstance(a, str) and a.strip()][:4]
        forms = [name, re.sub(r"\s*\([^)]*\)\s*$", "", name), *aliases]
        if not any(f and f.lower() in full_l for f in forms):
            continue  # the model invented it - it is not in these notes
        k = _key(name)
        if k in name_to_id:
            continue
        if k in struct_by_name:  # it is a section title: enrich that node instead of duplicating it
            sn = struct_by_name[k]
            pending.append((sn, str(c.get("definition") or "")[:400]))
            name_to_id[k] = sn["id"]
            enriched += 1
            continue
        old = old_by_name.get(k)
        sec = c.get("section") if c.get("section") in struct_ids else None
        hits = [ch for ch in chunks if any(f and f.lower() in ch["text"].lower() for f in forms)]
        if sec is None:
            cnt = Counter(ch.get("section_id") for ch in hits)
            sec = next((s for s, _ in cnt.most_common() if s in struct_ids), (old or {}).get("parent") or "root")
        parent = next(n for n in struct if n["id"] == sec) if sec in struct_ids else None
        pages = sorted({ch.get("page", 1) for ch in hits})
        nid = old["id"] if old else _cid(name)
        try:
            imp = max(0.2, min(1.0, float(c.get("importance", 0.5))))
        except (TypeError, ValueError):
            imp = 0.5
        new_nodes[nid] = {
            "id": nid, "label": name if len(name) <= 44 else name[:42] + "…", "full_label": name, "type": "concept",
            "level": (parent or {"level": 0})["level"] + 1, "parent": sec if sec in struct_ids else "root", "importance": round(imp, 2),
            "definition": (str(c.get("definition") or "")[:400] or (old or {}).get("definition")), "definition_quality": "llm" if c.get("definition") else (old or {}).get("definition_quality"),
            "summary": "", "examples": (old or {}).get("examples") or [], "formulas": (old or {}).get("formulas") or [], "facts": (old or {}).get("facts") or [],
            "aliases": sorted({*aliases, *((old or {}).get("aliases") or [])} - {name}), "pages": pages, "page": pages[0] if pages else (parent or {}).get("page", 1),
            "mentions": [], "section_id": sec, "n_children": 0, "order": order_base + i, "count": len(hits),
        }
        name_to_id[k] = nid
        for a in aliases:
            name_to_id.setdefault(_key(a), nid)
        sec_of_chunk_pages[sec] += pages

    n_old = len([n for n in nodes if n["type"] == "concept"])
    # A clean LLM list is usually *shorter* than the noisy rule-based one; only reject it when it is
    # clearly a failed extraction (a handful of concepts for a document that has many).
    found = len(new_nodes) + enriched
    if found < 5 or found < min(8, 0.2 * n_old):
        log.info("LLM graph too thin (%d concepts vs %d) - keeping the rule-based graph", found, n_old)
        return None
    for sn, d in pending:
        sn["is_term"] = True
        if d and not sn.get("definition"):
            sn["definition"], sn["definition_quality"] = d, "llm"
    # concepts take their reading order from where they first appear
    first_pos = {nid: (full_l.find(n["full_label"].lower()) if n["full_label"].lower() in full_l else 10**9) for nid, n in new_nodes.items()}
    for rank, nid in enumerate(sorted(new_nodes, key=lambda x: first_pos[x])):
        new_nodes[nid]["order"] = order_base + rank

    out_nodes = [n for n in struct] + list(new_nodes.values())
    order = reading_order(out_nodes, full)
    concept_like = set(new_nodes) | {n["id"] for n in struct if n.get("is_term") and n["type"] == "topic"}

    def look(nm: Any) -> str | None:
        return name_to_id.get(_key(nm)) if isinstance(nm, str) else None

    pre: list[tuple[str, str]] = []
    whys: dict[tuple[str, str], str] = {}
    for p in raw_p:
        a, b = look(p.get("before")), look(p.get("after"))
        if a and b and a != b and a in concept_like and b in concept_like:
            pre.append((a, b))
            whys[(a, b)] = str(p.get("why") or "")[:160]
    pre = _cut_cycles(list(dict.fromkeys(pre)), order)
    rel: list[tuple[str, str, float]] = []
    seen: set[frozenset[str]] = {frozenset(x) for x in pre}
    cnt: Counter[str] = Counter()
    for r in raw_r:
        a, b = look(r.get("a")), look(r.get("b"))
        if not a or not b or a == b or frozenset((a, b)) in seen or cnt[a] >= 3 or cnt[b] >= 3:
            continue
        seen.add(frozenset((a, b)))
        cnt[a] += 1
        cnt[b] += 1
        rel.append((a, b, 0.6))
    out_edges = _rebuild_edges(out_nodes, edges, pre, rel, whys)
    for n in out_nodes:
        n["n_children"] = sum(1 for m in out_nodes if m.get("parent") == n["id"])
    return out_nodes, out_edges, {"engine": engine, "llm_concepts": len(new_nodes), "llm_prereqs": len(pre), "llm_related": len(rel)}


def refine(kg: Any, brain: Any = None) -> dict[str, Any]:
    """Apply both passes in place on a KnowledgeGraph. Returns a small report for the stats."""
    chunks = [{"id": c.id, "section_id": c.section_id, "page": c.page, "text": c.text} for c in kg.chunks]
    full = "\n".join(c["text"] for c in chunks)
    report: dict[str, Any] = {"graph_engine": "rules"}
    if brain is not None and getattr(brain, "available", False):
        try:
            got = llm_refine(kg.nodes, kg.edges, chunks, brain)
        except Exception as e:  # pragma: no cover - never fail an upload because of the LLM
            log.warning("LLM graph refinement failed: %s", e)
            got = None
        if got:
            kg.nodes, kg.edges, info = got
            report.update(info)
            report["graph_engine"] = "llm:" + (info.get("engine") or "")
    nodes, edges, rep = clean_graph(kg.nodes, kg.edges, full)
    kg.nodes, kg.edges = nodes, edges
    kg.path = compute_path(nodes)
    kg.stats = _stats(nodes, edges, kg.stats)
    report.update(rep)
    kg.stats.update({k: v for k, v in report.items() if k in ("graph_engine", "dropped_modifiers", "merged_duplicates")})
    return report
