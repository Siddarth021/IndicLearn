"""Unit tests for EXP-005A evaluation runner and benchmark validation (Phase 5.1)."""

from __future__ import annotations

from indiclearn.evaluation.exp005a_runner import Exp005aRunner


def test_exp005a_runner_initialization():
    runner = Exp005aRunner()
    assert runner.eval_path.exists()
    assert runner.corpus_path.exists()
    assert len(runner.eval_data.get("queries", [])) == 60
    assert len(runner.chunks) > 0
    assert len(runner.concepts) > 0


def test_exp005a_leakage_audit_passes():
    runner = Exp005aRunner()
    leakage = runner.check_benchmark_leakage()
    assert leakage["leakage_detected"] is False
    assert leakage["total_queries_checked"] == 60
    assert "PASS" in leakage["status"]


def test_exp005a_retrieval_evaluation():
    runner = Exp005aRunner()
    ret = runner.evaluate_retrieval(top_k=5)
    agg = ret["aggregate"]

    assert agg["queries_count"] == 50
    assert agg["recall_at_5"] >= 0.85
    assert agg["gold_evidence_recall_at_5"] >= 0.85
    assert agg["mrr"] > 0.40

    # Ensure all 4 languages are present
    assert set(ret["by_language"].keys()) == {"en", "te", "te-Latn", "te-en"}
    # Ensure all 3 difficulties are present
    assert set(ret["by_difficulty"].keys()) == {"easy", "medium", "hard"}


def test_exp005a_kg_and_cm_ablations():
    runner = Exp005aRunner()
    kg = runner.evaluate_knowledge_graph_ablation(top_k=5)
    assert "system_a_hybrid_only" in kg
    assert "system_b_hybrid_plus_concept_mapping" in kg
    assert "system_c_hybrid_plus_concept_mapping_plus_graph" in kg

    cm = runner.evaluate_concept_mapping_ablation(top_k=5)
    assert "setting_1_raw_query" in cm
    assert "setting_2_normalized_query" in cm
    assert "setting_3_normalized_plus_canonical_expansion" in cm


def test_exp005a_real_llm_opt_in_offline_guard():
    # In mock mode, real LLM evaluation should report NOT RUN
    runner = Exp005aRunner(provider_mode="mock")
    res = runner.evaluate_real_llm_opt_in()
    assert res["status"] == "REAL LLM EVALUATION: NOT RUN"

    # In real mode without credentials, should safely report NOT RUN
    runner_real = Exp005aRunner(provider_mode="real")
    res_real = runner_real.evaluate_real_llm_opt_in()
    assert res_real["status"] == "REAL LLM EVALUATION: NOT RUN"
