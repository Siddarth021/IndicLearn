"""Descriptive statistics module for educational research evaluation (Phase 9).

Provides robust, reproducible summary statistics for benchmark metrics,
human ratings, and student interaction logs without making parametric assumptions.
"""

from __future__ import annotations

import math
from typing import Any


def compute_descriptive_stats(values: list[float]) -> dict[str, Any]:
    """Compute comprehensive descriptive statistics on a list of numerical values.

    Args:
        values: List of float or integer values.

    Returns:
        Dictionary containing sample size (n), mean, median, sample variance,
        sample standard deviation, interquartile range (IQR), min, max, and quartiles.
    """
    cleaned = [float(v) for v in values if v is not None and not math.isnan(v)]
    n = len(cleaned)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "variance": None,
            "std": None,
            "min": None,
            "max": None,
            "q25": None,
            "q75": None,
            "iqr": None,
        }

    mean_val = sum(cleaned) / n
    sorted_vals = sorted(cleaned)

    # Median
    if n % 2 == 1:
        median_val = sorted_vals[n // 2]
    else:
        median_val = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0

    # Sample variance and standard deviation (Bessel's correction n-1)
    if n > 1:
        variance_val = sum((x - mean_val) ** 2 for x in cleaned) / (n - 1)
        std_val = math.sqrt(variance_val)
    else:
        variance_val = 0.0
        std_val = 0.0

    # Quartiles (linear interpolation)
    def _percentile(p: float) -> float:
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_vals[int(k)]
        return sorted_vals[int(f)] * (c - k) + sorted_vals[int(c)] * (k - f)

    q25 = _percentile(0.25)
    q75 = _percentile(0.75)
    iqr_val = q75 - q25

    return {
        "n": n,
        "mean": round(mean_val, 4),
        "median": round(median_val, 4),
        "variance": round(variance_val, 4),
        "std": round(std_val, 4),
        "min": round(sorted_vals[0], 4),
        "max": round(sorted_vals[-1], 4),
        "q25": round(q25, 4),
        "q75": round(q75, 4),
        "iqr": round(iqr_val, 4),
    }


def compute_distribution(values: list[Any]) -> dict[str, Any]:
    """Compute frequency counts and percentage distribution of discrete or ordinal values."""
    if not values:
        return {"total": 0, "counts": {}, "percentages": {}}

    counts: dict[str, int] = {}
    for v in values:
        key = str(v)
        counts[key] = counts.get(key, 0) + 1

    total = len(values)
    percentages = {k: round((c / total) * 100.0, 2) for k, c in counts.items()}

    return {
        "total": total,
        "counts": counts,
        "percentages": percentages,
    }
