"""Evidence -> per-concept mastery estimate -> personalised overlay of the knowledge graph.

Model (a transparent Beta-style posterior, parameters reused from ``configs/config.yaml``)::

    score      = (a0 + sum(w_i * s_i * d_i)) / (a0 + b0 + sum(w_i * d_i))        # 0..1, prior 0.5
    confidence = W / (W + 2)          where W = sum(w_i * d_i)                    # 0..1
    d_i        = 0.5 ** (age_days / half_life)                                    # recency decay

Evidence kinds:

=================  =====  ======  ==============================================================
kind               score  weight  meaning
=================  =====  ======  ==============================================================
quiz_correct       1.0    1.00    answered a generated question correctly      (strong)
quiz_incorrect     0.0    1.00    answered incorrectly                         (strong)
self_understood    0.9    0.50    pressed "I understand this"                  (medium)
self_confused      0.1    0.50    pressed "Still confused"                     (medium)
asked              0.3    0.15    asked the tutor about it - a *doubt signal*  (weak)
viewed             0.55   0.03    opened the concept / read about it           (very weak)
=================  =====  ======  ==============================================================

A question is deliberately *weak* evidence: asking is a sign of engagement/uncertainty, not proof
that the student does not know the concept.  The weights are a proposed heuristic and are not
claimed to be validated.
"""

from __future__ import annotations

import time
from typing import Any

from indiclearn.studio.db import Database, jdump, now

EVIDENCE = {
    "quiz_correct": (1.0, 1.00),
    "quiz_incorrect": (0.0, 1.00),
    "self_understood": (0.9, 0.50),
    "self_confused": (0.1, 0.50),
    "asked": (0.3, 0.15),
    "viewed": (0.55, 0.03),
}
STRONG_KINDS = {"quiz_correct", "quiz_incorrect", "self_understood", "self_confused"}
PRIOR_A = 1.0
PRIOR_B = 1.0
HALF_LIFE_DAYS = 30.0
STATUSES = ("new", "doubt", "revise", "learning", "mastered")


def record_evidence(db: Database, user_id: str, doc_id: str, node_id: str, kind: str, meta: dict[str, Any] | None = None, ts: float | None = None) -> None:
    if kind not in EVIDENCE:
        raise ValueError(f"unknown evidence kind {kind!r}")
    score, weight = EVIDENCE[kind]
    db.run(
        "INSERT INTO evidence(user_id,doc_id,node_id,kind,score,weight,ts,meta_json) VALUES (?,?,?,?,?,?,?,?)",
        (user_id, doc_id, node_id, kind, score, weight, ts or now(), jdump(meta or {})),
    )


def _status(score: float, conf: float, strong_w: float, asked: int, total_w: float) -> str:
    if total_w <= 0.0:
        return "new"
    if strong_w < 0.2:
        return "doubt" if asked > 0 else "new"
    if score >= 0.8 and conf >= 0.4:
        return "mastered"
    if score < 0.45 and conf >= 0.25:
        return "revise"
    if conf >= 0.2:
        return "learning"
    return "doubt" if asked > 0 else "new"


def node_states(db: Database, user_id: str, doc_id: str, at: float | None = None) -> dict[str, dict[str, Any]]:
    """Per-node mastery for every node that has any evidence."""
    t_now = at or time.time()
    rows = db.all("SELECT node_id, kind, score, weight, ts FROM evidence WHERE user_id=? AND doc_id=? ORDER BY ts", (user_id, doc_id))
    acc: dict[str, dict[str, float]] = {}
    for r in rows:
        age_days = max(0.0, (t_now - (r["ts"] or t_now)) / 86400.0)
        d = 0.5 ** (age_days / HALF_LIFE_DAYS)
        a = acc.setdefault(r["node_id"], {"num": 0.0, "den": 0.0, "strong": 0.0, "asked": 0, "viewed": 0, "n": 0, "correct": 0, "wrong": 0, "last": 0.0})
        w = r["weight"] * d
        a["num"] += w * r["score"]
        a["den"] += w
        a["n"] += 1
        if r["kind"] in STRONG_KINDS:
            a["strong"] += w
        if r["kind"] == "asked":
            a["asked"] += 1
        if r["kind"] == "viewed":
            a["viewed"] += 1
        if r["kind"] == "quiz_correct":
            a["correct"] += 1
        if r["kind"] == "quiz_incorrect":
            a["wrong"] += 1
        a["last"] = max(a["last"], r["ts"] or 0.0)
    out: dict[str, dict[str, Any]] = {}
    for nid, a in acc.items():
        score = (PRIOR_A + a["num"]) / (PRIOR_A + PRIOR_B + a["den"])
        conf = a["den"] / (a["den"] + 2.0)
        out[nid] = {
            "score": round(score, 3), "confidence": round(conf, 3), "status": _status(score, conf, a["strong"], int(a["asked"]), a["den"]),
            "evidence": int(a["n"]), "asked": int(a["asked"]), "viewed": int(a["viewed"]), "correct": int(a["correct"]),
            "wrong": int(a["wrong"]), "last": a["last"],
        }
    return out


def _empty_state() -> dict[str, Any]:
    return {"score": 0.5, "confidence": 0.0, "status": "new", "evidence": 0, "asked": 0, "viewed": 0, "correct": 0, "wrong": 0, "last": 0.0}


def personal_overlay(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]], states: dict[str, dict[str, Any]], doubts: list[dict[str, Any]]
) -> dict[str, Any]:
    """Return personalised nodes/edges: mastery overlay, aggregated section states, doubt nodes."""
    by_id = {n["id"]: n for n in nodes}
    children: dict[str, list[str]] = {}
    for n in nodes:
        if n["parent"]:
            children.setdefault(n["parent"], []).append(n["id"])

    def descendants(nid: str) -> list[str]:
        out: list[str] = []
        stack = list(children.get(nid, []))
        while stack:
            c = stack.pop()
            out.append(c)
            stack.extend(children.get(c, []))
        return out

    out_nodes: list[dict[str, Any]] = []
    for n in nodes:
        m = dict(n)
        own = states.get(n["id"])
        if n["type"] in ("concept", "topic") and not children.get(n["id"]) or (n.get("is_term") and own):
            st = own or _empty_state()
            m["mastery"] = {**st, "aggregate": False}
        else:
            kids = [d for d in descendants(n["id"]) if by_id[d]["type"] in ("concept",) or (by_id[d]["type"] == "topic" and not children.get(d))]
            if own and n.get("is_term"):
                kids.append(n["id"])
            touched = [k for k in kids if k in states and states[k]["evidence"] > 0]
            total = max(len(kids), 1)
            mastered = sum(1 for k in kids if states.get(k, {}).get("status") == "mastered")
            revise = sum(1 for k in kids if states.get(k, {}).get("status") == "revise")
            doubt = sum(1 for k in kids if states.get(k, {}).get("status") == "doubt")
            wsum = sum(by_id[k]["importance"] for k in kids) or 1.0
            score = sum(by_id[k]["importance"] * states.get(k, _empty_state())["score"] for k in kids) / wsum if kids else 0.5
            if not touched:
                status = "new"
            elif mastered / total >= 0.8:
                status = "mastered"
            elif revise / total >= 0.25:
                status = "revise"
            elif all(states[k]["status"] == "doubt" for k in touched):
                status = "doubt"
            else:
                status = "learning"
            m["mastery"] = {
                "score": round(score, 3), "confidence": round(len(touched) / total, 3), "status": status, "evidence": sum(states[k]["evidence"] for k in touched),
                "asked": sum(states[k]["asked"] for k in touched), "aggregate": True, "mastered": mastered, "revise": revise, "doubt": doubt, "total": len(kids),
                "touched": len(touched), "progress": round(mastered / total, 3),
            }
        out_nodes.append(m)

    out_edges = list(edges)
    next_id = 1
    for d in doubts:
        nid = f"d{d['id']}"
        parent = d["parent_node_id"] if d["parent_node_id"] in by_id else "root"
        out_nodes.append({
            "id": nid, "label": d["label"][:46], "full_label": d["label"], "type": "doubt", "level": by_id.get(parent, {"level": 0})["level"] + 1,
            "parent": parent, "importance": 0.45, "definition": None, "definition_quality": None, "summary": d["question"], "examples": [], "formulas": [],
            "facts": [], "aliases": [], "pages": [], "page": by_id.get(parent, {"page": 1}).get("page", 1), "mentions": [], "section_id": by_id.get(parent, {}).get("section_id", "s0"),
            "n_children": 0, "order": 10_000 + d["id"], "covered": bool(d["covered"]), "asked_at": d["created_at"],
            "mastery": {"score": 0.3, "confidence": 0.15, "status": "doubt", "evidence": 1, "asked": 1, "aggregate": False},
        })
        out_edges.append({"id": f"ed{next_id}", "source": parent, "target": nid, "type": "contains", "weight": 1.0})
        next_id += 1
    return {"nodes": out_nodes, "edges": out_edges}


def recommend(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], states: dict[str, dict[str, Any]], k: int = 6) -> list[dict[str, Any]]:
    """Next-best concepts to study: weak/unfinished first, prerequisites respected."""
    prereq: dict[str, list[str]] = {}
    for e in edges:
        if e["type"] == "prereq":
            prereq.setdefault(e["target"], []).append(e["source"])
    recs: list[dict[str, Any]] = []
    for n in nodes:
        if n["type"] not in ("concept", "topic") or n.get("n_children", 0) > 0 and n["type"] == "topic":
            continue
        if n["type"] == "topic" and not n.get("is_term"):
            continue
        st = states.get(n["id"]) or _empty_state()
        if st["status"] == "mastered":
            continue
        gaps = [p for p in prereq.get(n["id"], []) if (states.get(p) or _empty_state())["score"] < 0.6 or (states.get(p) or _empty_state())["status"] in ("new", "doubt")]
        base = (1.0 - st["score"]) * (0.4 + n["importance"])
        reason = "not_started"
        if st["status"] == "revise":
            base *= 1.6
            reason = "weak_quiz"
        elif st["status"] == "doubt":
            base *= 1.4
            reason = "asked_doubt"
        elif st["status"] == "learning":
            base *= 1.0
            reason = "keep_practising"
        if gaps:
            reason_p = "prereq_gap"
        else:
            reason_p = None
        recs.append({"node_id": n["id"], "label": n["label"], "priority": round(base * (0.7 if gaps else 1.0), 3), "reason": reason, "prereq_gap": [
            {"id": g, "label": next((x["label"] for x in nodes if x["id"] == g), g)} for g in gaps[:2]
        ], "reason_p": reason_p, "status": st["status"], "score": st["score"], "page": n.get("page", 1)})
    recs.sort(key=lambda r: -r["priority"])
    # if there are prerequisite gaps, surface the prerequisite itself first
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r in recs:
        for g in r["prereq_gap"]:
            if g["id"] not in seen and not any(o["node_id"] == g["id"] for o in out):
                gn = next((x for x in nodes if x["id"] == g["id"]), None)
                if gn:
                    gs = states.get(g["id"]) or _empty_state()
                    out.append({"node_id": g["id"], "label": gn["label"], "priority": round(r["priority"] + 0.05, 3), "reason": "prereq_for", "prereq_gap": [], "reason_p": None,
                                "status": gs["status"], "score": gs["score"], "page": gn.get("page", 1), "for_label": r["label"]})
                    seen.add(g["id"])
        if r["node_id"] not in seen:
            out.append(r)
            seen.add(r["node_id"])
        if len(out) >= k:
            break
    return out[:k]
