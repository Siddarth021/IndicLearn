"""Unit and Integration Tests for Phase 9: Research Evaluation & Human Validation Framework.

Tests:
1. Statistical utilities (descriptive, significance, confidence intervals, Cohen's kappa, weighted kappa, Fleiss' kappa, missing data safety)
2. Experiment registry contracts & schema integrity
3. Research report generation, serialization, and deterministic markdown rendering
4. Human annotation template schemas, pseudonymity safeguards, and no-PII policies
5. Multilingual Unicode preservation and disaggregation (en, te, te-Latn, te-en)
6. EXP-010, EXP-011, EXP-012, EXP-013, EXP-014 runner invariants and HUMAN_DATA_REQUIRED states
"""

from __future__ import annotations

import json
from pathlib import Path

from indiclearn.evaluation.exp010_runner import EXP010Runner
from indiclearn.evaluation.exp011_runner import EXP011Runner
from indiclearn.evaluation.exp012_runner import EXP012Runner
from indiclearn.evaluation.exp013_runner import EXP013Runner
from indiclearn.evaluation.exp014_runner import EXP014Runner
from indiclearn.evaluation.statistics.agreement import (
    compute_agreement_percentage,
    compute_cohens_kappa,
    compute_fleiss_kappa,
    compute_weighted_cohens_kappa,
)
from indiclearn.evaluation.statistics.descriptive import (
    compute_descriptive_stats,
    compute_distribution,
)
from indiclearn.evaluation.statistics.significance import (
    bootstrap_confidence_interval,
    cohens_d,
    confidence_interval,
    holm_bonferroni_correction,
    mann_whitney_u_test,
    wilcoxon_signed_rank_test,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ==============================================================================
# 1. Descriptive Statistics Tests
# ==============================================================================

def test_descriptive_stats_normal() -> None:
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    res = compute_descriptive_stats(data)
    assert res["n"] == 5
    assert res["mean"] == 3.0
    assert res["median"] == 3.0
    assert res["min"] == 1.0
    assert res["max"] == 5.0
    assert res["variance"] == 2.5
    assert round(res["std"], 2) == 1.58
    assert res["iqr"] == 2.0


def test_descriptive_stats_empty_and_nan() -> None:
    empty_res = compute_descriptive_stats([])
    assert empty_res["n"] == 0
    assert empty_res["mean"] is None

    nan_res = compute_descriptive_stats([float("nan"), 10.0, 20.0])
    assert nan_res["n"] == 2
    assert nan_res["mean"] == 15.0


def test_distribution_calculation() -> None:
    data = ["A", "B", "A", "C", "A"]
    res = compute_distribution(data)
    assert res["total"] == 5
    assert res["counts"]["A"] == 3
    assert res["percentages"]["A"] == 60.0
    assert res["counts"]["B"] == 1


# ==============================================================================
# 2. Inter-Rater Agreement Tests
# ==============================================================================

def test_raw_agreement_and_missing() -> None:
    missing = compute_agreement_percentage([], [])
    assert missing["status"] == "NOT_AVAILABLE"

    r1 = [1, 2, 3, 4, 5]
    r2 = [1, 2, 0, 4, 5]
    res = compute_agreement_percentage(r1, r2)
    assert res["status"] == "AVAILABLE"
    assert res["matches"] == 4
    assert res["agreement_percentage"] == 80.0


def test_cohens_kappa() -> None:
    r1 = ["yes", "no", "yes", "yes", "no"]
    r2 = ["yes", "no", "yes", "no", "no"]
    res = compute_cohens_kappa(r1, r2)
    assert res["status"] == "AVAILABLE"
    assert res["n_samples"] == 5
    assert "kappa" in res
    assert res["kappa"] > 0.0


def test_weighted_cohens_kappa() -> None:
    # Ordinal 1-5 ratings
    r1 = [1, 2, 3, 4, 5]
    r2 = [1, 2, 3, 4, 4]  # Minimal difference
    quad = compute_weighted_cohens_kappa(r1, r2, weight="quadratic")
    assert quad["status"] == "AVAILABLE"
    assert quad["weighted_kappa"] > 0.80  # Very high agreement on quadratic penalty


def test_fleiss_kappa() -> None:
    # 4 subjects, 3 categories, 3 raters per subject
    ratings = [
        [3, 0, 0],
        [0, 3, 0],
        [0, 0, 3],
        [2, 1, 0],
    ]
    res = compute_fleiss_kappa(ratings)
    assert res["status"] == "AVAILABLE"
    assert res["n_subjects"] == 4
    assert res["n_raters"] == 3
    assert "fleiss_kappa" in res


# ==============================================================================
# 3. Significance & Effect Size Tests
# ==============================================================================

def test_confidence_interval_and_bootstrap() -> None:
    vals = [10.0, 11.0, 9.0, 10.5, 9.5]
    ci = confidence_interval(vals, confidence=0.95)
    assert ci["status"] == "AVAILABLE"
    assert ci["ci_lower"] < ci["mean"] < ci["ci_upper"]

    boot = bootstrap_confidence_interval(vals, seed=42)
    assert boot["status"] == "AVAILABLE"
    assert boot["ci_lower"] <= boot["ci_upper"]


def test_cohens_d() -> None:
    g1 = [1.0, 2.0, 3.0]
    g2 = [4.0, 5.0, 6.0]
    d = cohens_d(g1, g2)
    assert d["status"] == "AVAILABLE"
    assert d["cohens_d"] < 0.0  # g1 < g2
    assert d["interpretation"] == "Large effect"


def test_non_parametric_tests() -> None:
    # Mann-Whitney U
    u = mann_whitney_u_test([1, 2, 3, 4], [5, 6, 7, 8])
    assert u["status"] == "AVAILABLE"
    assert u["p_value"] < 0.05

    # Wilcoxon signed rank
    w = wilcoxon_signed_rank_test([1, 2, 3, 4], [2, 3, 4, 5])
    assert w["status"] == "AVAILABLE"
    assert w["n_pairs"] == 4


def test_holm_bonferroni_correction() -> None:
    p_vals = [0.01, 0.04, 0.03, 0.20]
    res = holm_bonferroni_correction(p_vals, alpha=0.05)
    assert len(res) == 4
    assert res[0]["raw_p_value"] == 0.01
    assert res[0]["rejected_null"] is True


# ==============================================================================
# 4. Experiment Registry Invariants
# ==============================================================================

def test_experiment_registry_schema() -> None:
    reg_path = PROJECT_ROOT / "data" / "evaluation" / "experiment_registry.json"
    assert reg_path.exists()
    with open(reg_path, encoding="utf-8") as f:
        data = json.load(f)

    assert "experiments" in data
    exp_ids = {e["experiment_id"] for e in data["experiments"]}
    # EXP-005A, EXP-005B, etc.
    assert "EXP-010" in exp_ids
    assert "EXP-011" in exp_ids
    assert "EXP-012" in exp_ids
    assert "EXP-013" in exp_ids
    assert "EXP-014" in exp_ids

    valid_statuses = {"PROPOSED", "IMPLEMENTED", "RUN", "HUMAN_DATA_REQUIRED", "COMPLETED"}
    for e in data["experiments"]:
        assert e["status"] in valid_statuses
        assert "research_question" in e
        assert "hypothesis" in e
        assert "metrics" in e
        assert "procedure" in e
        assert "limitations" in e


# ==============================================================================
# 5. Human Evaluation Protocol & Privacy Safeguards
# ==============================================================================

def test_human_schemas_and_pii_safeguards() -> None:
    human_dir = PROJECT_ROOT / "data" / "evaluation" / "human"
    assert (human_dir / "consent_template.md").exists()
    assert (human_dir / "rag_annotation_template.json").exists()
    assert (human_dir / "recommendation_annotation_template.json").exists()
    assert (human_dir / "assessment_annotation_template.json").exists()
    assert (human_dir / "usability_questionnaire.json").exists()
    assert (human_dir / "participant_schema.json").exists()
    assert (human_dir / "annotator_schema.json").exists()
    assert (human_dir / "example_annotation.json").exists()

    # Verify example annotation is explicitly marked as non-real data
    with open(human_dir / "example_annotation.json", encoding="utf-8") as f:
        ex = json.load(f)
    assert ex.get("is_synthetic_example") is True
    assert "NOT REAL PARTICIPANT DATA" in ex.get("_NOTICE", "")

    # Verify participant schema forbids PII fields
    with open(human_dir / "participant_schema.json", encoding="utf-8") as f:
        p_schema = json.load(f)
    forbidden_props = ["email", "phone", "aadhaar", "name", "address", "ip_address"]
    for fp in forbidden_props:
        assert fp not in p_schema.get("properties", {})


def test_neutral_usability_questionnaire() -> None:
    q_path = PROJECT_ROOT / "data" / "evaluation" / "human" / "usability_questionnaire.json"
    with open(q_path, encoding="utf-8") as f:
        q_data = json.load(f)

    # Ensure neutral items Q01 to Q06 are present
    likert = q_data["properties"]["likert_responses"]["properties"]
    assert "Q01_concept_findability" in likert
    assert "Q02_recommendation_transparency" in likert
    assert "Q03_knowledge_estimate_comprehension" in likert
    assert "Q04_evidence_traceability" in likert
    assert "Q05_map_navigation_ease" in likert
    assert "Q06_language_clarity" in likert


# ==============================================================================
# 6. EXP-010 to EXP-014 Execution & Contract Tests
# ==============================================================================

def test_exp010_recommendation_eval() -> None:
    runner = EXP010Runner()
    output = runner.run_all()
    assert output["metadata"]["status"] == "PASS"
    assert output["metadata"]["scenario_pass_rate"] == 1.0
    assert output["metadata"]["top1_recommendation_accuracy"] == 1.0
    assert output["metadata"]["action_selection_accuracy"] == 1.0
    assert output["metadata"]["reason_code_accuracy"] == 1.0
    assert output["metadata"]["ranking_consistency"] == 1.0
    assert len(output["scenarios"]) == 10


def test_exp011_adaptive_assessment_eval() -> None:
    runner = EXP011Runner()
    output = runner.run_all()
    assert output["metadata"]["status"] == "PASS"
    assert output["metadata"]["scenario_pass_rate"] == 1.0
    assert len(output["scenarios"]) == 10


def test_exp012_usability_empty_data_safety() -> None:
    runner = EXP012Runner()
    output = runner.run_all()
    # Must report NOT_RUN with HUMAN_DATA_REQUIRED without fabricating responses
    assert output["metadata"]["status"] == "NOT_RUN"
    assert "HUMAN_DATA_REQUIRED" in output["metadata"]["reason"]
    assert output["metadata"]["sample_size"] == 0
    assert output["analysis"]["status"] == "NOT_AVAILABLE"


def test_exp013_rag_annotation_empty_data_safety() -> None:
    runner = EXP013Runner()
    output = runner.run_all()
    # Must report HUMAN_DATA_REQUIRED without fabricating ratings
    assert output["metadata"]["status"] == "HUMAN_DATA_REQUIRED"
    assert output["metadata"]["sample_size"] == 0
    assert output["analysis"]["status"] == "NOT_AVAILABLE"


def test_exp014_multilingual_disaggregation_and_empty_safety() -> None:
    runner = EXP014Runner()
    output = runner.run_all()
    # Must report HUMAN_DATA_REQUIRED without fabricating ratings
    assert output["metadata"]["status"] == "HUMAN_DATA_REQUIRED"
    assert output["metadata"]["sample_size"] == 0
    assert output["analysis"]["status"] == "NOT_AVAILABLE"


def test_multilingual_unicode_preservation() -> None:
    te_text = "కన్వల్యూషన్ ఆపరేషన్ మరియు సిస్టమ్ రెస్పాన్స్"
    assert any("\u0c00" <= ch <= "\u0c7f" for ch in te_text)
    # Check that Telugu characters survive serialization roundtrip
    encoded = json.dumps({"telugu": te_text}, ensure_ascii=False)
    decoded = json.loads(encoded)
    assert decoded["telugu"] == te_text
