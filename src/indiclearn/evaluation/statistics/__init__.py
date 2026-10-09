"""Evaluation statistics package for IndicLearn (Phase 9)."""

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

__all__ = [
    "bootstrap_confidence_interval",
    "cohens_d",
    "compute_agreement_percentage",
    "compute_cohens_kappa",
    "compute_descriptive_stats",
    "compute_distribution",
    "compute_fleiss_kappa",
    "compute_weighted_cohens_kappa",
    "confidence_interval",
    "holm_bonferroni_correction",
    "mann_whitney_u_test",
    "wilcoxon_signed_rank_test",
]
