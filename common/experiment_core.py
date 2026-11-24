"""Compatibility facade for the original starter API.

New code should import the focused modules directly:

* ``common.network``, ``common.opponents``, ``common.candidates``,
  ``common.objectives``, ``common.ssa``, ``common.swap``,
  ``common.evaluation``, ``common.experiment``.

This module keeps the old names working for any external scripts, while fixing
the old objective-cache / Blocked_B defects.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd

try:  # package import
    from .candidates import make_candidate_pool as _make_candidate_pool
    from .network import build_adjacency, load_network, validate_network
    from .objectives import VARIANTS as _VARIANTS
    from .objectives import budget_coefficients, objective_score, objective_validity
    from .opponents import opponent_order
    from .ssa import (
        SSA_ITERATIONS,
        SSA_MUTATION_RATE,
        SSA_POPULATION,
        repair_solution,
        run_discrete_ssa as _run_discrete_ssa,
    )
    from .swap import SWAP_MAX_TESTS, sampled_swap_refinement
except ImportError:  # top-level import
    from candidates import make_candidate_pool as _make_candidate_pool  # type: ignore
    from network import build_adjacency, load_network, validate_network  # type: ignore
    from objectives import VARIANTS as _VARIANTS  # type: ignore
    from objectives import budget_coefficients, objective_score, objective_validity  # type: ignore
    from opponents import opponent_order  # type: ignore
    from ssa import (  # type: ignore
        SSA_ITERATIONS,
        SSA_MUTATION_RATE,
        SSA_POPULATION,
        repair_solution,
        run_discrete_ssa as _run_discrete_ssa,
    )
    from swap import SWAP_MAX_TESTS, sampled_swap_refinement  # type: ignore

K_VALUES = [10, 20, 30, 40, 50]
MASTER_SEED = 42
OPT_MC_RUNS = 15
FINAL_MC_RUNS = 100
POOL_RATIO = 10
RANDOM_POOL_FRACTION = 0.30

# Old code indexes VARIANTS by the human-readable name.
VARIANTS = OrderedDict((spec["name"], dict(spec)) for spec in _VARIANTS.values())


def opponent_orders(G, nodes, scenario_id, seed=42, max_k=50):
    """Old API: return the single ranked opponent order."""
    return opponent_order(G, nodes, scenario_id, seed=seed, max_k=max_k)


def make_candidate_pool(G, nodes, B, k, seed):
    """Old API: return ``(pool, scores)``."""
    pool, scores, _meta = _make_candidate_pool(G, nodes, B, k, seed)
    return pool, scores


def make_candidate_pool_full(G, nodes, B, k, seed, **kwargs):
    """New API: return ``(pool, scores, metadata)``."""
    return _make_candidate_pool(G, nodes, B, k, seed, **kwargs)


def run_discrete_ssa(*args, **kwargs):
    """Old API: return ``(food, food_score, history)``.

    Use ``common.ssa.run_discrete_ssa`` for the extended info dictionary.
    """
    food, score, history, _info = _run_discrete_ssa(*args, **kwargs)
    return food, score, history


def run_discrete_ssa_full(*args, **kwargs):
    return _run_discrete_ssa(*args, **kwargs)


def metric_cache_eval(sim_func, A, B, seeds):
    """Evaluate one seed set with the explicit paired B-only baseline.

    ``sim_func(A, B, seed)`` must return a mapping with ``spread_A``,
    ``spread_B``, ``ignorant`` and ``rounds``.  This replaces the starter's
    broken helper, which referenced a ``blocked_B`` key it never computed.
    """
    seeds = list(seeds)
    outcomes = [sim_func(A, B, s) for s in seeds]
    baseline = [sim_func([], B, s) for s in seeds]
    spread_A = np.array([x["spread_A"] for x in outcomes], dtype=float)
    spread_B = np.array([x["spread_B"] for x in outcomes], dtype=float)
    baseline_B = np.array([x["spread_B"] for x in baseline], dtype=float)
    net = spread_A - spread_B
    return {
        "spread_A": float(spread_A.mean()),
        "spread_B": float(spread_B.mean()),
        "net_advantage": float(net.mean()),
        "gain_A": float(net.mean()),
        "blocked_B": max(0.0, float(baseline_B.mean() - spread_B.mean())),
        "baseline_B": float(baseline_B.mean()),
        "ignorant": float(np.mean([x["ignorant"] for x in outcomes])),
        "rounds": float(np.mean([x["rounds"] for x in outcomes])),
    }
