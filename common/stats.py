"""Uncertainty and paired-comparison helpers.

The statistical unit for Monte Carlo uncertainty is the evaluation replicate
(common random seed).  This module never treats Monte Carlo replicates as
independent optimizer runs.  Paired comparisons use the same evaluation seeds
(common random numbers) and are therefore paired at the replicate level.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

try:  # optional; scipy is available in this environment but keep a fallback
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover
    _scipy_stats = None


def _as_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("empty input")
    return arr


def mean_ci(values: Iterable[float], confidence: float = 0.95,
            n_boot: int = 10000, seed: int = 12345) -> dict:
    """Mean, standard error and percentile bootstrap CI for a mean."""
    arr = _as_array(values)
    n = arr.size
    mean = float(arr.mean())
    if n == 1:
        return {
            "mean": mean, "se": float("nan"), "ci_low": float("nan"),
            "ci_high": float("nan"), "n": 1,
        }
    se = float(arr.std(ddof=1) / math.sqrt(n))
    rng = np.random.default_rng(seed)
    boot = rng.choice(arr, size=(n_boot, n), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": mean,
        "se": se,
        "ci_low": float(np.quantile(boot, alpha)),
        "ci_high": float(np.quantile(boot, 1.0 - alpha)),
        "n": int(n),
    }


def paired_diff(a: Iterable[float], b: Iterable[float], confidence: float = 0.95,
                n_boot: int = 10000, seed: int = 12345) -> dict:
    """Paired difference ``a - b`` with bootstrap CI and a paired t-test.

    ``a`` and ``b`` must be aligned by evaluation replicate (common random
    numbers).  The p-value is a paired t-test on the replicate-level differences;
    it is conditional on the two fixed seed sets and does not estimate
    between-optimizer-run variability.
    """
    a_arr = _as_array(a)
    b_arr = _as_array(b)
    if a_arr.size != b_arr.size:
        raise ValueError("paired inputs must have equal length")
    diff = a_arr - b_arr
    n = diff.size
    mean = float(diff.mean())
    result = {
        "mean_diff": mean,
        "median_diff": float(np.median(diff)),
        "n": int(n),
        "fraction_a_better": float(np.mean(diff > 0)),
        "fraction_tied": float(np.mean(diff == 0)),
    }
    if n == 1:
        result.update({"se": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                       "p_value": float("nan"), "cohens_dz": float("nan")})
        return result
    se = float(diff.std(ddof=1) / math.sqrt(n))
    rng = np.random.default_rng(seed)
    boot = rng.choice(diff, size=(n_boot, n), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    result["se"] = se
    result["ci_low"] = float(np.quantile(boot, alpha))
    result["ci_high"] = float(np.quantile(boot, 1.0 - alpha))
    sd = float(diff.std(ddof=1))
    result["cohens_dz"] = float(mean / sd) if sd > 0 else float("nan")
    if _scipy_stats is not None:
        try:
            result["p_value"] = float(_scipy_stats.ttest_rel(a_arr, b_arr).pvalue)
        except Exception:  # pragma: no cover
            result["p_value"] = float("nan")
    else:  # pragma: no cover
        result["p_value"] = float("nan")
    return result


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni adjustment. Returns per-key adjusted p and reject flag."""
    items = [(k, float(v)) for k, v in p_values.items() if not math.isnan(float(v))]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict] = {}
    previous = 0.0
    for rank, (key, p) in enumerate(items):
        adjusted = min(1.0, max(previous, (m - rank) * p))
        previous = adjusted
        out[key] = {"p_value": p, "p_adjusted": adjusted, "reject": adjusted <= alpha}
    for key, p in p_values.items():
        if key not in out:
            out[key] = {"p_value": float(p), "p_adjusted": float("nan"), "reject": False}
    return out


def summarize(values: Iterable[float]) -> dict:
    arr = _as_array(values)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "n": int(arr.size),
    }
