"""Statistical significance, confidence intervals, and effect size utilities (Phase 9).

Provides hypothesis testing and estimation tools for educational research evaluation.

Core Research Principles:
1. Document statistical assumptions explicitly (e.g. normality vs non-parametric ranks).
2. Avoid p-hacking: support multiple comparison corrections (Holm-Bonferroni).
3. Distinguish descriptive differences from statistically verified effects.
4. Report 'NOT_AVAILABLE' if data is empty or inadequate rather than inventing results.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import stats


def confidence_interval(values: list[float], confidence: float = 0.95) -> dict[str, Any]:
    """Compute Student's t confidence interval for the mean.

    Assumptions: Continuous distribution, approximately normal or large sample size (CLT).
    """
    cleaned = [float(v) for v in values if v is not None and not math.isnan(v)]
    n = len(cleaned)
    if n < 2:
        return {
            "status": "NOT_AVAILABLE",
            "reason": "Requires at least 2 observations to compute standard error",
            "mean": cleaned[0] if n == 1 else None,
            "ci_lower": None,
            "ci_upper": None,
            "confidence_level": confidence,
        }

    mean_val = float(np.mean(cleaned))
    sem = float(stats.sem(cleaned))
    margin = sem * float(stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1))

    return {
        "status": "AVAILABLE",
        "n": n,
        "mean": round(mean_val, 4),
        "std_error": round(sem, 4),
        "ci_lower": round(mean_val - margin, 4),
        "ci_upper": round(mean_val + margin, 4),
        "confidence_level": confidence,
    }


def bootstrap_confidence_interval(
    values: list[float],
    statistic_fn: Callable[[list[float]], float] | None = None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute non-parametric bootstrap confidence interval (percentile method).

    Does not assume normal distribution. Fixed seed ensures exact reproducibility.
    """
    cleaned = [float(v) for v in values if v is not None and not math.isnan(v)]
    n = len(cleaned)
    if n < 3:
        return {
            "status": "NOT_AVAILABLE",
            "reason": "Requires at least 3 observations for bootstrap resampling",
            "ci_lower": None,
            "ci_upper": None,
        }

    if statistic_fn is None:
        statistic_fn = lambda x: float(np.mean(x))

    rng = np.random.default_rng(seed)
    arr = np.array(cleaned)
    boot_stats = []

    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_stats.append(statistic_fn(sample.tolist()))

    alpha = 1.0 - confidence
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0

    ci_low = float(np.percentile(boot_stats, lower_pct))
    ci_high = float(np.percentile(boot_stats, upper_pct))
    point_est = statistic_fn(cleaned)

    return {
        "status": "AVAILABLE",
        "n": n,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "point_estimate": round(point_est, 4),
        "ci_lower": round(ci_low, 4),
        "ci_upper": round(ci_high, 4),
        "confidence_level": confidence,
    }


def cohens_d(group1: list[float], group2: list[float]) -> dict[str, Any]:
    """Compute Cohen's d standardized effect size between two independent groups.

    $$d = \frac{\bar{x}_1 - \bar{x}_2}{s_{\text{pooled}}}$$
    """
    g1 = [float(x) for x in group1 if x is not None and not math.isnan(x)]
    g2 = [float(x) for x in group2 if x is not None and not math.isnan(x)]

    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return {
            "status": "NOT_AVAILABLE",
            "reason": "Each group requires at least 2 observations to compute pooled variance",
            "effect_size": None,
        }

    m1, m2 = float(np.mean(g1)), float(np.mean(g2))
    v1, v2 = float(np.var(g1, ddof=1)), float(np.var(g2, ddof=1))

    # Pooled standard deviation
    s_pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if s_pooled == 0.0:
        d = 0.0
    else:
        d = (m1 - m2) / s_pooled

    # Effect size convention: small (|d| >= 0.2), medium (|d| >= 0.5), large (|d| >= 0.8)
    abs_d = abs(d)
    if abs_d < 0.20:
        interpretation = "Negligible effect"
    elif abs_d < 0.50:
        interpretation = "Small effect"
    elif abs_d < 0.80:
        interpretation = "Medium effect"
    else:
        interpretation = "Large effect"

    return {
        "status": "AVAILABLE",
        "n1": n1,
        "n2": n2,
        "mean1": round(m1, 4),
        "mean2": round(m2, 4),
        "cohens_d": round(d, 4),
        "interpretation": interpretation,
    }


def mann_whitney_u_test(group1: list[float], group2: list[float]) -> dict[str, Any]:
    """Perform two-sided Mann-Whitney U rank-sum test for two independent samples.

    Non-parametric alternative to independent samples t-test.
    Assumptions: Ordinal or continuous observations, independent observations.
    """
    g1 = [float(x) for x in group1 if x is not None and not math.isnan(x)]
    g2 = [float(x) for x in group2 if x is not None and not math.isnan(x)]

    if len(g1) < 2 or len(g2) < 2:
        return {
            "status": "NOT_AVAILABLE",
            "reason": "Each group requires at least 2 observations",
            "u_statistic": None,
            "p_value": None,
        }

    res = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    return {
        "status": "AVAILABLE",
        "n1": len(g1),
        "n2": len(g2),
        "u_statistic": round(float(res.statistic), 4),
        "p_value": round(float(res.pvalue), 6),
        "is_significant_05": res.pvalue < 0.05,
        "test_name": "Mann-Whitney U (two-sided)",
    }


def wilcoxon_signed_rank_test(group1: list[float], group2: list[float]) -> dict[str, Any]:
    """Perform two-sided Wilcoxon signed-rank test for paired observations.

    Non-parametric alternative to paired samples t-test.
    """
    if len(group1) != len(group2):
        raise ValueError(f"Paired samples must have identical length: {len(group1)} vs {len(group2)}")

    diffs = [float(a - b) for a, b in zip(group1, group2, strict=True) if a is not None and b is not None]
    non_zero_diffs = [d for d in diffs if d != 0.0]

    if len(non_zero_diffs) < 3:
        return {
            "status": "NOT_AVAILABLE",
            "reason": "Requires at least 3 non-zero paired differences",
            "statistic": None,
            "p_value": None,
        }

    res = stats.wilcoxon(diffs, alternative="two-sided")
    return {
        "status": "AVAILABLE",
        "n_pairs": len(diffs),
        "statistic": round(float(res.statistic), 4),
        "p_value": round(float(res.pvalue), 6),
        "is_significant_05": res.pvalue < 0.05,
        "test_name": "Wilcoxon Signed-Rank (two-sided)",
    }


def holm_bonferroni_correction(
    p_values: list[float], alpha: float = 0.05
) -> list[dict[str, Any]]:
    """Apply Holm-Bonferroni step-down procedure to adjust for multiple hypothesis comparisons.

    Controls Family-Wise Error Rate (FWER) without the conservative over-penalty of standard Bonferroni.
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort p-values ascending while keeping track of original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    results = [None] * m

    still_rejecting = True
    for rank, (orig_idx, p) in enumerate(indexed):
        k = rank + 1
        threshold = alpha / (m - k + 1)
        if still_rejecting and p <= threshold:
            rejected = True
        else:
            still_rejecting = False
            rejected = False

        results[orig_idx] = {
            "original_index": orig_idx,
            "raw_p_value": round(p, 6),
            "rank": k,
            "adjusted_threshold": round(threshold, 6),
            "rejected_null": rejected,
        }

    return results
