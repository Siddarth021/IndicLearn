"""Inter-rater reliability and agreement statistics module (Phase 9).

Implements Cohen's Kappa, Weighted Cohen's Kappa (linear & quadratic),
Fleiss' Kappa, and exact agreement metrics for human evaluation panels.

Core Principle:
If human annotation data is absent, returns explicit NOT_AVAILABLE state
rather than fabricating reliability statistics.
"""

from __future__ import annotations

import math
from typing import Any


def _interpret_kappa(kappa: float) -> str:
    """Landis and Koch (1977) benchmark interpretations."""
    if kappa < 0.0:
        return "Poor (Less than chance)"
    if kappa <= 0.20:
        return "Slight agreement"
    if kappa <= 0.40:
        return "Fair agreement"
    if kappa <= 0.60:
        return "Moderate agreement"
    if kappa <= 0.80:
        return "Substantial agreement"
    return "Almost perfect agreement"


def compute_agreement_percentage(rater1: list[Any], rater2: list[Any]) -> dict[str, Any]:
    """Compute raw percentage agreement between two evaluators."""
    if not rater1 or not rater2:
        return {
            "status": "NOT_AVAILABLE",
            "reason": "No human annotation data supplied",
            "n_samples": 0,
            "agreement_percentage": None,
        }

    if len(rater1) != len(rater2):
        raise ValueError(f"Rater sequence length mismatch: {len(rater1)} vs {len(rater2)}")

    n = len(rater1)
    matches = sum(1 for a, b in zip(rater1, rater2, strict=True) if a == b)
    pct = (matches / n) * 100.0

    return {
        "status": "AVAILABLE",
        "n_samples": n,
        "matches": matches,
        "agreement_percentage": round(pct, 2),
    }


def compute_cohens_kappa(rater1: list[Any], rater2: list[Any]) -> dict[str, Any]:
    """Compute unweighted Cohen's Kappa for categorical evaluations between two raters."""
    if not rater1 or not rater2:
        return {
            "status": "NOT_AVAILABLE",
            "reason": "No human annotation data supplied",
            "n_samples": 0,
            "observed_agreement": None,
            "expected_agreement": None,
            "kappa": None,
            "interpretation": "N/A",
        }

    if len(rater1) != len(rater2):
        raise ValueError(f"Rater sequence length mismatch: {len(rater1)} vs {len(rater2)}")

    n = len(rater1)
    if n == 0:
        return {
            "status": "NOT_AVAILABLE",
            "reason": "Empty annotation set",
            "n_samples": 0,
            "kappa": None,
        }

    categories = sorted({str(x) for x in rater1} | {str(x) for x in rater2})
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    num_cats = len(categories)

    # Contingency matrix
    matrix = [[0] * num_cats for _ in range(num_cats)]
    for a, b in zip(rater1, rater2, strict=True):
        matrix[cat_to_idx[str(a)]][cat_to_idx[str(b)]] += 1

    # Observed agreement Po
    po = sum(matrix[i][i] for i in range(num_cats)) / n

    # Expected agreement Pe
    row_sums = [sum(matrix[i][j] for j in range(num_cats)) for i in range(num_cats)]
    col_sums = [sum(matrix[i][j] for i in range(num_cats)) for j in range(num_cats)]
    pe = sum((row_sums[i] * col_sums[i]) for i in range(num_cats)) / (n * n)

    if pe >= 1.0:
        kappa = 1.0
    else:
        kappa = (po - pe) / (1.0 - pe)

    # Standard error of kappa
    try:
        se = math.sqrt((po * (1.0 - po)) / (n * ((1.0 - pe) ** 2)))
    except (ZeroDivisionError, ValueError):
        se = 0.0

    return {
        "status": "AVAILABLE",
        "n_samples": n,
        "num_categories": num_cats,
        "categories": categories,
        "observed_agreement": round(po, 4),
        "expected_agreement": round(pe, 4),
        "kappa": round(kappa, 4),
        "std_error": round(se, 4),
        "interpretation": _interpret_kappa(kappa),
    }


def compute_weighted_cohens_kappa(
    rater1: list[int | float],
    rater2: list[int | float],
    weight: str = "quadratic",
) -> dict[str, Any]:
    """Compute Weighted Cohen's Kappa for ordinal ratings (e.g. Likert 1 to 5).

    Args:
        rater1: List of ordinal numeric ratings.
        rater2: List of ordinal numeric ratings.
        weight: 'linear' or 'quadratic'.
    """
    if not rater1 or not rater2:
        return {
            "status": "NOT_AVAILABLE",
            "reason": "No human annotation data supplied",
            "n_samples": 0,
            "kappa": None,
            "weight_type": weight,
        }

    if len(rater1) != len(rater2):
        raise ValueError(f"Rater sequence length mismatch: {len(rater1)} vs {len(rater2)}")

    n = len(rater1)
    if n == 0:
        return {
            "status": "NOT_AVAILABLE",
            "reason": "Empty annotation set",
            "n_samples": 0,
            "kappa": None,
        }

    categories = sorted({int(x) for x in rater1} | {int(x) for x in rater2})
    num_cats = len(categories)
    if num_cats <= 1:
        return {
            "status": "AVAILABLE",
            "n_samples": n,
            "kappa": 1.0,
            "interpretation": "Almost perfect agreement (constant rating)",
            "weight_type": weight,
        }

    cat_to_idx = {c: i for i, c in enumerate(categories)}
    max_dist = max(categories) - min(categories)

    # Weights matrix: 0 for identical, >0 for distance
    w = [[0.0] * num_cats for _ in range(num_cats)]
    for i, c1 in enumerate(categories):
        for j, c2 in enumerate(categories):
            diff = abs(c1 - c2)
            if weight == "linear":
                w[i][j] = diff / max_dist if max_dist > 0 else 0.0
            else:  # quadratic
                w[i][j] = (diff**2) / (max_dist**2) if max_dist > 0 else 0.0

    # Observed contingency
    obs = [[0] * num_cats for _ in range(num_cats)]
    for a, b in zip(rater1, rater2, strict=True):
        obs[cat_to_idx[int(a)]][cat_to_idx[int(b)]] += 1

    row_sums = [sum(obs[i][j] for j in range(num_cats)) for i in range(num_cats)]
    col_sums = [sum(obs[i][j] for i in range(num_cats)) for j in range(num_cats)]

    # Expected contingency
    exp = [[(row_sums[i] * col_sums[j]) / n for j in range(num_cats)] for i in range(num_cats)]

    num = sum(w[i][j] * obs[i][j] for i in range(num_cats) for j in range(num_cats))
    den = sum(w[i][j] * exp[i][j] for i in range(num_cats) for j in range(num_cats))

    if den == 0.0:
        kappa = 1.0
    else:
        kappa = 1.0 - (num / den)

    return {
        "status": "AVAILABLE",
        "n_samples": n,
        "num_categories": num_cats,
        "weight_type": weight,
        "weighted_kappa": round(kappa, 4),
        "interpretation": _interpret_kappa(kappa),
    }


def compute_fleiss_kappa(ratings_matrix: list[list[int]]) -> dict[str, Any]:
    """Compute Fleiss' Kappa for inter-rater agreement across multiple raters (>2).

    Args:
        ratings_matrix: An N x K matrix where N is the number of subjects/items,
                        K is the number of categories, and table[i][j] is the count
                        of raters who assigned subject i to category j.
    """
    if not ratings_matrix or len(ratings_matrix) == 0:
        return {
            "status": "NOT_AVAILABLE",
            "reason": "No human annotation data supplied",
            "n_subjects": 0,
            "kappa": None,
        }

    n_subjects = len(ratings_matrix)
    n_categories = len(ratings_matrix[0])

    # Total raters per subject
    n_raters = sum(ratings_matrix[0])
    if n_raters <= 1:
        return {
            "status": "NOT_AVAILABLE",
            "reason": "Fleiss' Kappa requires at least 2 raters per subject",
            "n_subjects": n_subjects,
            "kappa": None,
        }

    # Verify all subjects have the same number of ratings
    for row in ratings_matrix:
        if sum(row) != n_raters:
            raise ValueError("All subjects in Fleiss' matrix must have equal total rater counts")

    # Proportion of all assignments to each category
    total_ratings = n_subjects * n_raters
    pj = [sum(ratings_matrix[i][j] for i in range(n_subjects)) / total_ratings for j in range(n_categories)]

    # Extent of agreement for the i-th subject Pi
    pi_list = []
    for i in range(n_subjects):
        p_i = (sum(ratings_matrix[i][j] ** 2 for j in range(n_categories)) - n_raters) / (
            n_raters * (n_raters - 1)
        )
        pi_list.append(p_i)

    p_bar = sum(pi_list) / n_subjects
    p_bar_e = sum(p**2 for p in pj)

    if p_bar_e >= 1.0:
        kappa = 1.0
    else:
        kappa = (p_bar - p_bar_e) / (1.0 - p_bar_e)

    return {
        "status": "AVAILABLE",
        "n_subjects": n_subjects,
        "n_categories": n_categories,
        "n_raters": n_raters,
        "observed_mean_agreement": round(p_bar, 4),
        "expected_mean_agreement": round(p_bar_e, 4),
        "fleiss_kappa": round(kappa, 4),
        "interpretation": _interpret_kappa(kappa),
    }
