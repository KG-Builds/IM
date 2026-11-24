"""Monte Carlo metric evaluation with safe caching.

Caching rules (fixes the starter's objective-cache bug):

* The metric cache is keyed by graph fingerprint, A seed set, B seed set, the
  evaluation seed list, and the evaluation protocol.  Metrics are objective-
  independent, so sharing them across objectives is safe.
* Objective values are always recomputed from metrics with the requested
  objective id and K; they are never cached under an objective-agnostic key.
* The B-only baseline is cached separately under (B, seeds, protocol) because
  it does not depend on A.

Blocked_B protocol (explicit, reproducible, provisional):

    baseline_B(B, s) = Spread_B(A=empty, B, s)
    Blocked_B        = max(0, mean_s baseline_B - mean_s Spread_B(A, B, s))

The same Monte Carlo seeds are used for the baseline and the A/B run (paired /
common random numbers).  Per-seed unclipped differences are retained for
analysis.  This is the starter's only implementation.  No reference notebook
exists in this workspace to confirm it is the intended research definition, so
Blocked_B-dependent objectives are labelled ``provisional_blocked_b``.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

try:  # package import
    from .objectives import objective_score
except ImportError:  # top-level import
    from objectives import objective_score  # type: ignore

PROTOCOL_ID = "paired_b_only_baseline_v1"


class MetricEvaluator:
    def __init__(self, simulator, fingerprint: str, protocol: str = PROTOCOL_ID,
                 n_threads: int = 0):
        self.simulator = simulator
        self.fingerprint = str(fingerprint)
        self.protocol = str(protocol)
        self.n_threads = int(n_threads)
        self.metric_cache: dict[tuple, dict] = {}
        self.baseline_cache: dict[tuple, dict] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.simulations = 0
        self.simulation_seconds = 0.0

    # ------------------------------------------------------------------
    def _metric_key(self, A: tuple, B: tuple, seeds: tuple) -> tuple:
        return (self.fingerprint, A, B, seeds, self.protocol)

    def _baseline_key(self, B: tuple, seeds: tuple) -> tuple:
        return (self.fingerprint, B, seeds, self.protocol)

    def reset_counters(self) -> None:
        self.cache_hits = 0
        self.cache_misses = 0
        self.simulations = 0
        self.simulation_seconds = 0.0

    # ------------------------------------------------------------------
    def _baseline(self, B: tuple, seeds: tuple) -> dict:
        key = self._baseline_key(B, seeds)
        if key in self.baseline_cache:
            return self.baseline_cache[key]
        import time

        t0 = time.perf_counter()
        results = self.simulator.simulate_many([], list(B), list(seeds), n_threads=self.n_threads)
        self.simulation_seconds += time.perf_counter() - t0
        self.simulations += len(results)
        payload = {
            "spread_B": np.array([r["spread_B"] for r in results], dtype=np.int32),
            "ignorant": np.array([r["ignorant"] for r in results], dtype=np.int32),
            "rounds": np.array([r["rounds"] for r in results], dtype=np.int32),
        }
        self.baseline_cache[key] = payload
        return payload

    # ------------------------------------------------------------------
    def evaluate(self, A: Iterable[int], B: Iterable[int], seeds: Iterable[int]) -> dict:
        """Return mean metrics and per-seed arrays for one seed set."""
        A_t = tuple(sorted({int(x) for x in A}))
        B_t = tuple(sorted({int(x) for x in B}))
        seeds_t = tuple(int(s) for s in seeds)
        if not seeds_t:
            raise ValueError("At least one evaluation seed is required")
        if set(A_t) & set(B_t):
            raise ValueError("A and B seed sets overlap")

        key = self._metric_key(A_t, B_t, seeds_t)
        if key in self.metric_cache:
            self.cache_hits += 1
            cached = dict(self.metric_cache[key])
            cached["from_cache"] = True
            return cached

        self.cache_misses += 1
        import time

        t0 = time.perf_counter()
        results = self.simulator.simulate_many(list(A_t), list(B_t), list(seeds_t),
                                               n_threads=self.n_threads)
        self.simulation_seconds += time.perf_counter() - t0
        self.simulations += len(results)

        spread_A = np.array([r["spread_A"] for r in results], dtype=np.int32)
        spread_B = np.array([r["spread_B"] for r in results], dtype=np.int32)
        ignorant = np.array([r["ignorant"] for r in results], dtype=np.int32)
        rounds = np.array([r["rounds"] for r in results], dtype=np.int32)

        baseline = self._baseline(B_t, seeds_t)
        baseline_B = baseline["spread_B"]
        blocked_runs = baseline_B.astype(np.int64) - spread_B.astype(np.int64)
        net_runs = spread_A.astype(np.int64) - spread_B.astype(np.int64)

        baseline_mean = float(baseline_B.mean())
        spread_B_mean = float(spread_B.mean())
        metrics = {
            "spread_A": float(spread_A.mean()),
            "spread_B": spread_B_mean,
            "net_advantage": float(net_runs.mean()),
            "gain_A": float(net_runs.mean()),
            "blocked_B": max(0.0, baseline_mean - spread_B_mean),
            "baseline_B": baseline_mean,
            "ignorant": float(ignorant.mean()),
            "rounds": float(rounds.mean()),
            "spread_A_runs": spread_A,
            "spread_B_runs": spread_B,
            "net_advantage_runs": net_runs,
            "blocked_B_runs": blocked_runs,
            "baseline_B_runs": baseline_B,
            "ignorant_runs": ignorant,
            "rounds_runs": rounds,
            "n_runs": len(seeds_t),
            "seeds": seeds_t,
            "A": A_t,
            "B": B_t,
            "from_cache": False,
        }
        self.metric_cache[key] = metrics
        return dict(metrics)

    # ------------------------------------------------------------------
    def evaluate_objective(self, A: Iterable[int], B: Iterable[int],
                           seeds: Iterable[int], variant: str, k: int) -> float:
        metrics = self.evaluate(A, B, seeds)
        return objective_score(metrics, variant, k)

    def cache_stats(self) -> dict:
        return {
            "metric_cache_entries": len(self.metric_cache),
            "baseline_cache_entries": len(self.baseline_cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "simulations": self.simulations,
            "simulation_seconds": self.simulation_seconds,
        }
