"""Graph evaluation metrics for concept and relationship extraction (Phase 3).

Calculates Precision, Recall, and F1 score against manual gold-standard graphs.
Only valid when comparing to an independent ground-truth evaluation set.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from indiclearn.domain.concepts import Concept, ConceptRelation


class MetricScore(BaseModel):
    """Precision, Recall, and F1 evaluation score."""

    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)


class GraphEvaluationReport(BaseModel):
    """Complete comparative evaluation between extracted graph and gold standard."""

    concept_metrics: MetricScore
    relation_metrics: MetricScore
    missed_concepts: list[str] = Field(default_factory=list)
    spurious_concepts: list[str] = Field(default_factory=list)
    missed_relations: list[str] = Field(default_factory=list)
    spurious_relations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def compute_prf1(tp: int, fp: int, fn: int) -> MetricScore:
    """Compute precision, recall, and harmonic F1 score."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return MetricScore(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )


def evaluate_concepts(
    extracted_concepts: list[Concept],
    gold_concepts: list[dict[str, Any] | Concept],
) -> tuple[MetricScore, list[str], list[str]]:
    """Compare extracted concept set to gold concepts by canonical name (case-insensitive)."""
    extracted_names = {c.canonical_name.strip().lower() for c in extracted_concepts}

    gold_names: set[str] = set()
    for gc in gold_concepts:
        name = gc.canonical_name if isinstance(gc, Concept) else gc["canonical_name"]
        gold_names.add(name.strip().lower())

    tp_set = extracted_names & gold_names
    fp_set = extracted_names - gold_names
    fn_set = gold_names - extracted_names

    score = compute_prf1(len(tp_set), len(fp_set), len(fn_set))
    return score, sorted(fn_set), sorted(fp_set)


def evaluate_relations(
    extracted_relations: list[ConceptRelation],
    gold_relations: list[dict[str, Any] | ConceptRelation],
) -> tuple[MetricScore, list[str], list[str]]:
    """Compare extracted relation set to gold relations by (source, relation_type, target) triplet."""

    def to_sig(item: dict[str, Any] | ConceptRelation) -> str:
        if isinstance(item, ConceptRelation):
            src = item.source_concept_id
            tgt = item.target_concept_id
            rtype = (
                item.relation_type.value
                if hasattr(item.relation_type, "value")
                else str(item.relation_type)
            )
        else:
            src = item["source_concept_id"]
            tgt = item["target_concept_id"]
            rtype = item["relation_type"]

        def strip_prefix(cid: str) -> str:
            cid = cid.strip().lower()
            if cid.startswith("cpt_signals_and_systems_"):
                return cid[len("cpt_signals_and_systems_") :]
            if cid.startswith("cpt_"):
                parts = cid.split("_", 2)
                return parts[-1] if len(parts) >= 3 else cid[4:]
            return cid

        return f"{strip_prefix(src)}:{rtype.lower()}:{strip_prefix(tgt)}"

    extracted_sigs = {to_sig(r) for r in extracted_relations}
    gold_sigs = {to_sig(r) for r in gold_relations}

    tp_set = extracted_sigs & gold_sigs
    fp_set = extracted_sigs - gold_sigs
    fn_set = gold_sigs - extracted_sigs

    score = compute_prf1(len(tp_set), len(fp_set), len(fn_set))
    return score, sorted(fn_set), sorted(fp_set)


def evaluate_graph_extraction(
    extracted_concepts: list[Concept],
    extracted_relations: list[ConceptRelation],
    gold_data: dict[str, Any],
) -> GraphEvaluationReport:
    """Evaluate extracted graph against a gold evaluation standard."""
    c_score, missed_c, spurious_c = evaluate_concepts(
        extracted_concepts, gold_data.get("concepts", [])
    )
    r_score, missed_r, spurious_r = evaluate_relations(
        extracted_relations, gold_data.get("relations", [])
    )

    return GraphEvaluationReport(
        concept_metrics=c_score,
        relation_metrics=r_score,
        missed_concepts=missed_c,
        spurious_concepts=spurious_c,
        missed_relations=missed_r,
        spurious_relations=spurious_r,
        metadata={"subject_id": gold_data.get("subject_id", "signals_and_systems")},
    )
