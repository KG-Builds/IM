"""Experiment coordinator: scenarios, checkpoints, resume and CSV output.

Layout (all inside the workspace)::

    results/<run_id>/<scenario>/experiment_results.csv
    results/<run_id>/<scenario>/checkpoints/<scenario>_K<K>_<objective>.json
    results/<run_id>/<scenario>/mc_runs/<scenario>_K<K>_<objective>_<mode>.csv
    results/<run_id>/<scenario>/ssa_history/<scenario>_K<K>_<objective>.csv
    results/<run_id>/run_metadata.json
    results/<run_id>/manifest.json
    logs/<run_id>.log

Every configuration uses the same network and the same opponent order.  SSA is
run once per (scenario, K, objective) and its saved seed set is shared by the
SSA-only and SSA+Swap modes.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:  # package imports
    from .candidates import make_candidate_pool, pool_signature
    from .cpp_adapter import CppSimulator, HAS_CPP, IMPORT_ERROR, cpp_available, openmp_available
    from .evaluation import MetricEvaluator, PROTOCOL_ID
    from .network import load_network, semantic_fingerprint, validate_network, verify_fingerprint
    from .objectives import VARIANTS, objective_score, objective_validity, resolve_variant
    from .opponents import SCENARIO_NAMES, opponent_order
    from .ssa import SSA_ITERATIONS, SSA_MUTATION_RATE, SSA_POPULATION, run_discrete_ssa
    from .stats import mean_ci
    from .swap import SWAP_MAX_TESTS, sampled_swap_refinement
    from .utils import (
        ANALYSIS_DIR,
        LOGS_DIR,
        NETWORK_CSV,
        NETWORK_METADATA_V2,
        RESULTS_DIR,
        atomic_write_csv,
        atomic_write_json,
        ensure_dir,
        environment_info,
        local_now_iso,
        sha256_bytes,
        sha256_file,
        stable_seed,
        utc_now_iso,
    )
except ImportError:  # top-level imports with common/ on sys.path
    from candidates import make_candidate_pool, pool_signature  # type: ignore
    from cpp_adapter import CppSimulator, HAS_CPP, IMPORT_ERROR, cpp_available, openmp_available  # type: ignore
    from evaluation import MetricEvaluator, PROTOCOL_ID  # type: ignore
    from network import load_network, semantic_fingerprint, validate_network, verify_fingerprint  # type: ignore
    from objectives import VARIANTS, objective_score, objective_validity, resolve_variant  # type: ignore
    from opponents import SCENARIO_NAMES, opponent_order  # type: ignore
    from ssa import SSA_ITERATIONS, SSA_MUTATION_RATE, SSA_POPULATION, run_discrete_ssa  # type: ignore
    from stats import mean_ci  # type: ignore
    from swap import SWAP_MAX_TESTS, sampled_swap_refinement  # type: ignore
    from utils import (  # type: ignore
        ANALYSIS_DIR,
        LOGS_DIR,
        NETWORK_CSV,
        NETWORK_METADATA_V2,
        RESULTS_DIR,
        atomic_write_csv,
        atomic_write_json,
        ensure_dir,
        environment_info,
        local_now_iso,
        sha256_bytes,
        sha256_file,
        stable_seed,
        utc_now_iso,
    )

MODE_SSA = "SSA only"
MODE_SWAP = "SSA + Swap refinement"
MODES = (MODE_SSA, MODE_SWAP)
SCENARIOS = ("B1", "B2", "B3", "B4")
K_VALUES = (10, 20, 30, 40, 50)

RESULT_COLUMNS = [
    "run_id",
    "run_kind",
    "scenario",
    "scenario_name",
    "K",
    "objective_id",
    "objective_name",
    "objective_formula",
    "objective_validity",
    "blocked_b_protocol",
    "mode",
    "network_fingerprint",
    "network_semantic_fingerprint",
    "master_seed",
    "optimization_mc_runs",
    "final_mc_runs",
    "optimization_seed_protocol",
    "final_seed_protocol",
    "ssa_population",
    "ssa_iterations",
    "ssa_mutation_rate",
    "ssa_random_seed",
    "candidate_pool_size",
    "candidate_pool_ranked",
    "candidate_pool_random",
    "candidate_pool_signature",
    "opponent_order_signature",
    "initial_optimizer_score",
    "final_optimizer_score",
    "final_evaluation_objective_score",
    "swap_objective_before",
    "swap_objective_after",
    "swap_objective_gain",
    "spread_A",
    "spread_B",
    "net_advantage",
    "gain_A",
    "blocked_B",
    "baseline_B",
    "ignorant_nodes",
    "average_rounds",
    "spread_A_se",
    "spread_B_se",
    "net_advantage_se",
    "net_advantage_ci_low",
    "net_advantage_ci_high",
    "blocked_B_se",
    "blocked_B_ci_low",
    "blocked_B_ci_high",
    "ssa_runtime_sec",
    "swap_runtime_sec",
    "final_evaluation_runtime_sec",
    "total_runtime_sec",
    "simulations_for_config",
    "simulation_seconds_for_config",
    "cache_hits_for_config",
    "cache_misses_for_config",
    "ssa_from_checkpoint",
    "swap_tests",
    "swap_accepted",
    "swap_accepted_old",
    "swap_accepted_new",
    "ssa_seed_set",
    "swap_initial_seed_set",
    "final_seed_set",
    "opponent_seed_set",
    "config_signature_hash",
    "checkpoint_file",
    "ssa_history_file",
    "mc_runs_file",
    "status",
    "timestamp",
]


class CheckpointMismatch(RuntimeError):
    pass


def _slug(mode: str) -> str:
    return "ssa_only" if mode == MODE_SSA else "ssa_swap"


def _json_list(values: Iterable[int]) -> str:
    return json.dumps([int(x) for x in values], separators=(",", ":"))


def _hash_obj(obj) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, default=str).encode("utf-8"))


def default_run_id(run_kind: str = "full") -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = "SMOKE_TEST" if run_kind.upper() == "SMOKE_TEST" else "run"
    return f"{prefix}_{stamp}"


@dataclass
class ExperimentConfig:
    run_id: str
    run_kind: str = "full"
    scenarios: tuple[str, ...] = SCENARIOS
    k_values: tuple[int, ...] = K_VALUES
    objective_ids: tuple[str, ...] = tuple(VARIANTS.keys())
    modes: tuple[str, ...] = MODES
    master_seed: int = 42
    ssa_population: int = SSA_POPULATION
    ssa_iterations: int = SSA_ITERATIONS
    ssa_mutation_rate: float = SSA_MUTATION_RATE
    optimization_mc_runs: int = 15
    final_mc_runs: int = 100
    pool_ratio: int = 10
    random_pool_fraction: float = 0.30
    swap_max_tests: int = SWAP_MAX_TESTS
    n_threads: int = 0
    network_path: Path = NETWORK_CSV
    results_root: Path = RESULTS_DIR
    resume: bool = True
    log_dir: Path = LOGS_DIR

    def __post_init__(self):
        self.run_id = str(self.run_id).strip()
        if not self.run_id or any(ch in self.run_id for ch in "/\\") or ".." in self.run_id:
            raise ValueError("run_id must be a simple name without path separators or '..'")
        self.scenarios = tuple(self.scenarios)
        self.k_values = tuple(int(k) for k in self.k_values)
        self.objective_ids = tuple(self.objective_ids)
        self.modes = tuple(self.modes)
        for sid in self.scenarios:
            if sid not in SCENARIOS:
                raise ValueError(f"Unknown scenario {sid!r}")
        # Normalise objective names to ids so seed derivation is unambiguous.
        self.objective_ids = tuple(resolve_variant(oid)["id"] for oid in self.objective_ids)
        for mode in self.modes:
            if mode not in MODES:
                raise ValueError(f"Unknown mode {mode!r}")
        if not self.k_values or any(k <= 0 for k in self.k_values):
            raise ValueError("k_values must be positive")
        if self.optimization_mc_runs < 1 or self.final_mc_runs < 1:
            raise ValueError("Monte Carlo run counts must be positive")
        if self.ssa_population < 2 or self.ssa_iterations < 1:
            raise ValueError("SSA population must be >=2 and iterations >=1")
        if self.pool_ratio < 1:
            raise ValueError("pool_ratio must be >=1")
        if not (0.0 <= self.random_pool_fraction <= 1.0):
            raise ValueError("random_pool_fraction must be in [0,1]")
        if self.swap_max_tests < 0:
            raise ValueError("swap_max_tests must be non-negative")
        self.network_path = Path(self.network_path)
        self.results_root = Path(self.results_root)
        self.log_dir = Path(self.log_dir)

    @property
    def run_dir(self) -> Path:
        return self.results_root / self.run_id

    def scenario_dir(self, scenario: str) -> Path:
        return self.run_dir / scenario

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "run_kind": self.run_kind,
            "scenarios": list(self.scenarios),
            "k_values": list(self.k_values),
            "objective_ids": list(self.objective_ids),
            "modes": list(self.modes),
            "master_seed": int(self.master_seed),
            "ssa_population": int(self.ssa_population),
            "ssa_iterations": int(self.ssa_iterations),
            "ssa_mutation_rate": float(self.ssa_mutation_rate),
            "optimization_mc_runs": int(self.optimization_mc_runs),
            "final_mc_runs": int(self.final_mc_runs),
            "pool_ratio": int(self.pool_ratio),
            "random_pool_fraction": float(self.random_pool_fraction),
            "swap_max_tests": int(self.swap_max_tests),
            "n_threads": int(self.n_threads),
            "network_path": str(self.network_path),
            "results_root": str(self.results_root),
            "resume": bool(self.resume),
        }

    def expected_keys(self) -> list[tuple]:
        return [
            (sid, int(k), oid, mode)
            for sid in self.scenarios
            for k in self.k_values
            for oid in self.objective_ids
            for mode in self.modes
        ]


def _scenario_number(scenario: str) -> int:
    return int(scenario[1:])


def derive_seeds(config: ExperimentConfig, scenario: str, k: int, objective_id: str) -> dict:
    """Deterministic seed derivation, preserving the starter's streams."""
    n = _scenario_number(scenario)
    oid_index = list(VARIANTS.keys()).index(objective_id)
    opt_seeds = [
        config.master_seed * 100000 + 3000000 + n * 10000 + i
        for i in range(config.optimization_mc_runs)
    ]
    final_seeds = [900000000 + i for i in range(config.final_mc_runs)]
    return {
        "opponent_seed": int(config.master_seed),
        "candidate_pool_seed": int(config.master_seed + 600000 + n),
        "optimization_seeds": opt_seeds,
        "final_seeds": final_seeds,
        "ssa_random_seed": int(config.master_seed + 500000 + n * 1000 + k * 10 + oid_index),
        "swap_random_seed": int(770000 + n * 1000 + k + oid_index),
        "optimization_seed_protocol": (
            "fixed per scenario: master_seed*100000+3000000+scenario_number*10000+i, "
            "i=0..optimization_mc_runs-1; common random numbers across objectives/modes"
        ),
        "final_seed_protocol": (
            "fixed common list 900000000+i, i=0..final_mc_runs-1, shared by every "
            "scenario/objective/mode; paired comparisons use the same replicate index"
        ),
    }


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        if not cpp_available():
            raise RuntimeError(
                "C++ engine is required for the experiment. "
                f"Import error: {IMPORT_ERROR}"
            )
        self.logger = setup_logger(config.run_id, config.log_dir)
        self.df, self.G, self.nodes = load_network(config.network_path)
        self.network_fingerprint = sha256_file(config.network_path)
        self.network_semantic_fingerprint = semantic_fingerprint(self.df)
        self.validation_report = validate_network(self.df)
        self.simulator = CppSimulator(self.nodes, self.G, n_threads=config.n_threads)
        self.evaluator = MetricEvaluator(
            self.simulator, self.network_fingerprint, protocol=PROTOCOL_ID,
            n_threads=config.n_threads,
        )
        self._opponent_orders: dict[str, list[int]] = {}

    # ------------------------------------------------------------------
    def opponent_order(self, scenario: str) -> list[int]:
        if scenario not in self._opponent_orders:
            self._opponent_orders[scenario] = opponent_order(
                self.G, self.nodes, scenario, seed=self.config.master_seed, max_k=50
            )
        return self._opponent_orders[scenario]

    def _config_signature(self, scenario: str, k: int, objective_id: str,
                          pool_sig: str, order_sig: str, seeds: dict) -> dict:
        spec = resolve_variant(objective_id)
        return {
            "schema_version": 2,
            "run_kind": self.config.run_kind,
            "scenario": scenario,
            "K": int(k),
            "objective_id": objective_id,
            "objective_name": spec["name"],
            "network_fingerprint": self.network_fingerprint,
            "network_semantic_fingerprint": self.network_semantic_fingerprint,
            "master_seed": int(self.config.master_seed),
            "ssa_population": int(self.config.ssa_population),
            "ssa_iterations": int(self.config.ssa_iterations),
            "ssa_mutation_rate": float(self.config.ssa_mutation_rate),
            "optimization_mc_runs": int(self.config.optimization_mc_runs),
            "final_mc_runs": int(self.config.final_mc_runs),
            "pool_ratio": int(self.config.pool_ratio),
            "random_pool_fraction": float(self.config.random_pool_fraction),
            "candidate_pool_signature": pool_sig,
            "opponent_order_signature": order_sig,
            "swap_max_tests": int(self.config.swap_max_tests),
            "evaluation_protocol": PROTOCOL_ID,
            "optimization_seed_protocol": seeds["optimization_seed_protocol"],
            "final_seed_protocol": seeds["final_seed_protocol"],
            "optimization_seeds_hash": _hash_obj(seeds["optimization_seeds"]),
            "final_seeds_hash": _hash_obj(seeds["final_seeds"]),
        }

    def _checkpoint_path(self, scenario: str, k: int, objective_id: str) -> Path:
        return self.config.scenario_dir(scenario) / "checkpoints" / f"{scenario}_K{k}_{objective_id}.json"

    def _mc_path(self, scenario: str, k: int, objective_id: str, mode: str) -> Path:
        return self.config.scenario_dir(scenario) / "mc_runs" / f"{scenario}_K{k}_{objective_id}_{_slug(mode)}.csv"

    def _history_path(self, scenario: str, k: int, objective_id: str) -> Path:
        return self.config.scenario_dir(scenario) / "ssa_history" / f"{scenario}_K{k}_{objective_id}.csv"

    # ------------------------------------------------------------------
    def _read_existing_rows(self, scenario: str) -> tuple[list[dict], list[str]]:
        path = self.config.scenario_dir(scenario) / "experiment_results.csv"
        if not path.exists():
            return [], list(RESULT_COLUMNS)
        df = pd.read_csv(path)
        missing = [c for c in RESULT_COLUMNS if c not in df.columns]
        if missing:
            raise CheckpointMismatch(
                f"Existing CSV {path} is missing required columns {missing}. "
                "Use a new run_id instead of mixing schemas."
            )
        columns = list(df.columns)
        for extra in RESULT_COLUMNS:
            if extra not in columns:
                columns.append(extra)
        records = df.to_dict("records")
        keys = [(r["scenario"], int(r["K"]), r["objective_id"], r["mode"]) for r in records]
        if len(keys) != len(set(keys)):
            raise CheckpointMismatch(f"Existing CSV {path} contains duplicate configurations")
        return records, columns

    def _write_results(self, scenario: str, records: list[dict], columns: list[str]) -> Path:
        path = self.config.scenario_dir(scenario) / "experiment_results.csv"
        df = pd.DataFrame(records)
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df.reindex(columns=columns)
        atomic_write_csv(df, path)
        return path

    def _load_checkpoint(self, path: Path, signature_hash: str) -> dict | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise CheckpointMismatch(f"Unreadable checkpoint {path}: {exc}") from exc
        if payload.get("config_signature_hash") != signature_hash:
            raise CheckpointMismatch(
                f"Checkpoint {path} does not match the current configuration. "
                "Use a new run_id or restore the matching configuration."
            )
        return payload

    # ------------------------------------------------------------------
    def _run_ssa(self, scenario: str, k: int, objective_id: str, pool: list[int],
                 B: list[int], seeds: dict, signature: dict, signature_hash: str,
                 checkpoint_path: Path) -> dict:
        scorer = lambda A, Bv, sv, variant, kk: self.evaluator.evaluate_objective(A, Bv, sv, variant, kk)
        t0 = time.perf_counter()
        ssa_set, ssa_score, history, info = run_discrete_ssa(
            pool, B, k, seeds["optimization_seeds"], objective_id, scorer,
            population_size=self.config.ssa_population,
            iterations=self.config.ssa_iterations,
            mutation_rate=self.config.ssa_mutation_rate,
            random_seed=seeds["ssa_random_seed"],
        )
        ssa_runtime = time.perf_counter() - t0
        history_path = self._history_path(scenario, k, objective_id)
        atomic_write_csv(history, history_path)
        payload = {
            "schema_version": 2,
            "config_signature": signature,
            "config_signature_hash": signature_hash,
            "scenario": scenario,
            "K": int(k),
            "objective_id": objective_id,
            "objective_name": resolve_variant(objective_id)["name"],
            "network_fingerprint": self.network_fingerprint,
            "candidate_pool": [int(u) for u in pool],
            "candidate_pool_signature": signature["candidate_pool_signature"],
            "opponent_seed_set": [int(u) for u in B],
            "optimization_seeds": list(seeds["optimization_seeds"]),
            "final_seeds": list(seeds["final_seeds"]),
            "ssa_seed_set": [int(u) for u in ssa_set],
            "ssa_objective_score": float(ssa_score),
            "ssa_runtime_sec": float(ssa_runtime),
            "ssa_info": info,
            "ssa_history": history.to_dict("records"),
            "created_utc": utc_now_iso(),
        }
        atomic_write_json(checkpoint_path, payload)
        return payload

    def _validate_saved_set(self, values: Sequence[int], pool: list[int], B: list[int], k: int,
                            label: str) -> list[int]:
        values = [int(x) for x in values]
        if len(values) != k or len(set(values)) != k:
            raise CheckpointMismatch(f"{label} does not contain exactly K unique nodes")
        if not set(values).issubset(set(pool)):
            raise CheckpointMismatch(f"{label} contains nodes outside the candidate pool")
        if set(values) & set(B):
            raise CheckpointMismatch(f"{label} contains opponent B seeds")
        return values

    # ------------------------------------------------------------------
    def run_scenario(self, scenario: str) -> pd.DataFrame:
        cfg = self.config
        scenario_dir = ensure_dir(cfg.scenario_dir(scenario))
        for sub in ("checkpoints", "mc_runs", "ssa_history"):
            ensure_dir(scenario_dir / sub)
        records, columns = self._read_existing_rows(scenario)
        existing = {
            (r["scenario"], int(r["K"]), r["objective_id"], r["mode"]): r for r in records
        }
        order = self.opponent_order(scenario)
        order_sig = f"{len(order)}:{stable_seed(*order):016x}"
        self.logger.info("Scenario %s (%s): %d existing rows", scenario, SCENARIO_NAMES[scenario], len(records))

        scenario_meta = {
            "scenario": scenario,
            "scenario_name": SCENARIO_NAMES[scenario],
            "opponent_order": [int(u) for u in order],
            "opponent_order_signature": order_sig,
            "opponent_seed_sets": {},
            "candidate_pools": {},
            "started_utc": utc_now_iso(),
        }

        for k in cfg.k_values:
            B = [int(u) for u in order[:k]]
            scenario_meta["opponent_seed_sets"][str(k)] = B
            pool, _scores, pool_meta = make_candidate_pool(
                self.G, self.nodes, B, k,
                seed=cfg.master_seed + 600000 + _scenario_number(scenario),
                pool_ratio=cfg.pool_ratio,
                random_fraction=cfg.random_pool_fraction,
            )
            pool_sig = pool_signature(pool)
            scenario_meta["candidate_pools"][str(k)] = {
                **pool_meta, "signature": pool_sig, "pool": [int(u) for u in pool],
            }
            self.logger.info(
                "  K=%d: B seeds=%d, pool=%d (ranked=%d, random=%d)",
                k, len(B), len(pool), pool_meta["n_ranked"], pool_meta["n_random"],
            )

            for objective_id in cfg.objective_ids:
                spec = resolve_variant(objective_id)
                seeds = derive_seeds(cfg, scenario, k, objective_id)
                signature = self._config_signature(
                    scenario, k, objective_id, pool_sig, order_sig, seeds
                )
                signature_hash = _hash_obj(signature)
                checkpoint_path = self._checkpoint_path(scenario, k, objective_id)

                checkpoint = self._load_checkpoint(checkpoint_path, signature_hash)
                ssa_from_checkpoint = checkpoint is not None
                if checkpoint is not None:
                    ssa_set = self._validate_saved_set(
                        checkpoint["ssa_seed_set"], pool, B, k, "checkpoint ssa_seed_set"
                    )
                    ssa_score = float(checkpoint["ssa_objective_score"])
                    ssa_runtime = float(checkpoint["ssa_runtime_sec"])
                    ssa_info = checkpoint.get("ssa_info", {})
                    self.logger.info(
                        "  K=%d %s: SSA loaded from checkpoint (score=%.6f)",
                        k, objective_id, ssa_score,
                    )
                else:
                    checkpoint = self._run_ssa(
                        scenario, k, objective_id, pool, B, seeds, signature,
                        signature_hash, checkpoint_path,
                    )
                    ssa_set = self._validate_saved_set(
                        checkpoint["ssa_seed_set"], pool, B, k, "SSA result"
                    )
                    ssa_score = float(checkpoint["ssa_objective_score"])
                    ssa_runtime = float(checkpoint["ssa_runtime_sec"])
                    ssa_info = checkpoint.get("ssa_info", {})
                    self.logger.info(
                        "  K=%d %s: SSA done in %.2fs (score=%.6f, evals=%s)",
                        k, objective_id, ssa_runtime, ssa_score,
                        ssa_info.get("objective_evaluations"),
                    )

                for mode in cfg.modes:
                    key = (scenario, k, objective_id, mode)
                    if key in existing:
                        row = existing[key]
                        if row.get("config_signature_hash") != signature_hash:
                            raise CheckpointMismatch(
                                f"Existing result row {key} has a different configuration "
                                "signature. Use a new run_id."
                            )
                        self.logger.info("  K=%d %s %s: already complete, skipping", k, objective_id, mode)
                        continue

                    stats_before = self.evaluator.cache_stats()
                    swap_info = None
                    swap_runtime = 0.0
                    final_set = list(ssa_set)
                    swap_initial: list[int] = []
                    if mode == MODE_SWAP:
                        swap_initial = list(ssa_set)
                        t0 = time.perf_counter()
                        final_set, swap_info = sampled_swap_refinement(
                            ssa_set, pool, B, k, seeds["optimization_seeds"],
                            objective_id,
                            lambda A, Bv, sv, variant, kk: self.evaluator.evaluate_objective(A, Bv, sv, variant, kk),
                            max_tests=cfg.swap_max_tests,
                            random_seed=seeds["swap_random_seed"],
                        )
                        swap_runtime = time.perf_counter() - t0
                        final_set = self._validate_saved_set(final_set, pool, B, k, "swap result")

                    t0 = time.perf_counter()
                    metrics = self.evaluator.evaluate(final_set, B, seeds["final_seeds"])
                    final_runtime = time.perf_counter() - t0
                    stats_after = self.evaluator.cache_stats()

                    ci_net = mean_ci(metrics["net_advantage_runs"])
                    ci_blocked = mean_ci(metrics["blocked_B_runs"])
                    spread_A_se = _se(metrics["spread_A_runs"])
                    spread_B_se = _se(metrics["spread_B_runs"])
                    final_objective = objective_score(metrics, objective_id, k)
                    if swap_info is not None:
                        final_optimizer = float(swap_info["objective_after"])
                        swap_before = float(swap_info["objective_before"])
                        swap_after = float(swap_info["objective_after"])
                        swap_gain = float(swap_info["objective_gain"])
                        swap_tests = int(swap_info["swap_tests"])
                        swap_accepted = bool(swap_info["accepted"])
                        accepted_old = swap_info["accepted_swap"][0] if swap_accepted else np.nan
                        accepted_new = swap_info["accepted_swap"][1] if swap_accepted else np.nan
                    else:
                        final_optimizer = float(ssa_score)
                        swap_before = np.nan
                        swap_after = np.nan
                        swap_gain = np.nan
                        swap_tests = 0
                        swap_accepted = False
                        accepted_old = np.nan
                        accepted_new = np.nan

                    mc_path = self._mc_path(scenario, k, objective_id, mode)
                    mc_df = pd.DataFrame({
                        "seed": list(metrics["seeds"]),
                        "spread_A": metrics["spread_A_runs"],
                        "spread_B": metrics["spread_B_runs"],
                        "ignorant": metrics["ignorant_runs"],
                        "rounds": metrics["rounds_runs"],
                        "baseline_B": metrics["baseline_B_runs"],
                        "blocked_B_run": metrics["blocked_B_runs"],
                        "net_advantage_run": metrics["net_advantage_runs"],
                    })
                    atomic_write_csv(mc_df, mc_path)

                    row = {
                        "run_id": cfg.run_id,
                        "run_kind": cfg.run_kind,
                        "scenario": scenario,
                        "scenario_name": SCENARIO_NAMES[scenario],
                        "K": int(k),
                        "objective_id": objective_id,
                        "objective_name": spec["name"],
                        "objective_formula": spec["formula"],
                        "objective_validity": spec["validity"],
                        "blocked_b_protocol": PROTOCOL_ID,
                        "mode": mode,
                        "network_fingerprint": self.network_fingerprint,
                        "network_semantic_fingerprint": self.network_semantic_fingerprint,
                        "master_seed": int(cfg.master_seed),
                        "optimization_mc_runs": int(cfg.optimization_mc_runs),
                        "final_mc_runs": int(cfg.final_mc_runs),
                        "optimization_seed_protocol": seeds["optimization_seed_protocol"],
                        "final_seed_protocol": seeds["final_seed_protocol"],
                        "ssa_population": int(cfg.ssa_population),
                        "ssa_iterations": int(cfg.ssa_iterations),
                        "ssa_mutation_rate": float(cfg.ssa_mutation_rate),
                        "ssa_random_seed": int(seeds["ssa_random_seed"]),
                        "candidate_pool_size": int(len(pool)),
                        "candidate_pool_ranked": int(pool_meta["n_ranked"]),
                        "candidate_pool_random": int(pool_meta["n_random"]),
                        "candidate_pool_signature": pool_sig,
                        "opponent_order_signature": order_sig,
                        "initial_optimizer_score": float(ssa_score),
                        "final_optimizer_score": final_optimizer,
                        "final_evaluation_objective_score": float(final_objective),
                        "swap_objective_before": swap_before,
                        "swap_objective_after": swap_after,
                        "swap_objective_gain": swap_gain,
                        "spread_A": float(metrics["spread_A"]),
                        "spread_B": float(metrics["spread_B"]),
                        "net_advantage": float(metrics["net_advantage"]),
                        "gain_A": float(metrics["gain_A"]),
                        "blocked_B": float(metrics["blocked_B"]),
                        "baseline_B": float(metrics["baseline_B"]),
                        "ignorant_nodes": float(metrics["ignorant"]),
                        "average_rounds": float(metrics["rounds"]),
                        "spread_A_se": spread_A_se,
                        "spread_B_se": spread_B_se,
                        "net_advantage_se": float(ci_net["se"]),
                        "net_advantage_ci_low": float(ci_net["ci_low"]),
                        "net_advantage_ci_high": float(ci_net["ci_high"]),
                        "blocked_B_se": float(ci_blocked["se"]),
                        "blocked_B_ci_low": float(ci_blocked["ci_low"]),
                        "blocked_B_ci_high": float(ci_blocked["ci_high"]),
                        "ssa_runtime_sec": float(ssa_runtime),
                        "swap_runtime_sec": float(swap_runtime),
                        "final_evaluation_runtime_sec": float(final_runtime),
                        "total_runtime_sec": float(ssa_runtime + swap_runtime + final_runtime),
                        "simulations_for_config": int(stats_after["simulations"] - stats_before["simulations"]),
                        "simulation_seconds_for_config": float(
                            stats_after["simulation_seconds"] - stats_before["simulation_seconds"]
                        ),
                        "cache_hits_for_config": int(stats_after["cache_hits"] - stats_before["cache_hits"]),
                        "cache_misses_for_config": int(stats_after["cache_misses"] - stats_before["cache_misses"]),
                        "ssa_from_checkpoint": bool(ssa_from_checkpoint),
                        "swap_tests": swap_tests,
                        "swap_accepted": swap_accepted,
                        "swap_accepted_old": accepted_old,
                        "swap_accepted_new": accepted_new,
                        "ssa_seed_set": _json_list(ssa_set),
                        "swap_initial_seed_set": _json_list(swap_initial) if swap_initial else "",
                        "final_seed_set": _json_list(final_set),
                        "opponent_seed_set": _json_list(B),
                        "config_signature_hash": signature_hash,
                        "checkpoint_file": str(checkpoint_path.relative_to(cfg.run_dir)),
                        "ssa_history_file": str(self._history_path(scenario, k, objective_id).relative_to(cfg.run_dir)),
                        "mc_runs_file": str(mc_path.relative_to(cfg.run_dir)),
                        "status": "completed",
                        "timestamp": utc_now_iso(),
                    }
                    records.append(row)
                    existing[key] = row
                    self._write_results(scenario, records, columns)
                    self.logger.info(
                        "  K=%d %s %s: NetAdv=%.3f (A=%.1f B=%.1f) BlockedB=%.3f [%s]",
                        k, objective_id, mode, row["net_advantage"], row["spread_A"],
                        row["spread_B"], row["blocked_B"], row["objective_validity"],
                    )

        scenario_meta["finished_utc"] = utc_now_iso()
        atomic_write_json(scenario_dir / "scenario_metadata.json", scenario_meta)
        return pd.read_csv(scenario_dir / "experiment_results.csv")

    # ------------------------------------------------------------------
    def run_all(self) -> dict:
        cfg = self.config
        ensure_dir(cfg.run_dir)
        metadata_path = cfg.run_dir / "run_metadata.json"
        if metadata_path.exists():
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            if previous.get("config") != cfg.to_dict():
                raise CheckpointMismatch(
                    f"Run metadata at {metadata_path} differs from the current config. "
                    "Use a new run_id instead of changing an existing run."
                )
        metadata = {
            "run_id": cfg.run_id,
            "run_kind": cfg.run_kind,
            "config": cfg.to_dict(),
            "network": {
                "path": str(cfg.network_path),
                "file_sha256": self.network_fingerprint,
                "semantic_sha256": self.network_semantic_fingerprint,
                "validation": self.validation_report,
                "metadata_v2": str(NETWORK_METADATA_V2),
            },
            "environment": environment_info(),
            "cpp": {
                "available": bool(cpp_available()),
                "openmp": bool(openmp_available()),
                "import_error": IMPORT_ERROR,
            },
            "evaluation_protocol": PROTOCOL_ID,
            "started_utc": utc_now_iso(),
            "status": "running",
        }
        atomic_write_json(metadata_path, metadata)
        self.logger.info("Run %s (%s) starting: %d expected configurations",
                         cfg.run_id, cfg.run_kind, len(cfg.expected_keys()))

        for scenario in cfg.scenarios:
            self.run_scenario(scenario)
            manifest = self.build_manifest()
            atomic_write_json(cfg.run_dir / "manifest.json", manifest)
            self.logger.info("Scenario %s complete: %d/%d configurations",
                             scenario, manifest["completed_configurations"],
                             manifest["expected_configurations"])

        manifest = self.build_manifest()
        manifest["status"] = "completed" if not manifest["missing"] else "incomplete"
        manifest["finished_utc"] = utc_now_iso()
        atomic_write_json(cfg.run_dir / "manifest.json", manifest)
        metadata["finished_utc"] = utc_now_iso()
        metadata["status"] = manifest["status"]
        atomic_write_json(metadata_path, metadata)
        self.logger.info("Run %s finished: %d/%d configurations, status=%s",
                         cfg.run_id, manifest["completed_configurations"],
                         manifest["expected_configurations"], manifest["status"])
        return manifest

    # ------------------------------------------------------------------
    def build_manifest(self) -> dict:
        cfg = self.config
        completed = []
        for scenario in cfg.scenarios:
            path = cfg.scenario_dir(scenario) / "experiment_results.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            for row in df.itertuples(index=False):
                completed.append({
                    "scenario": row.scenario,
                    "K": int(row.K),
                    "objective_id": row.objective_id,
                    "mode": row.mode,
                })
        completed_keys = {(c["scenario"], c["K"], c["objective_id"], c["mode"]) for c in completed}
        expected = cfg.expected_keys()
        missing = [
            {"scenario": s, "K": k, "objective_id": o, "mode": m}
            for (s, k, o, m) in expected
            if (s, k, o, m) not in completed_keys
        ]
        per_scenario = {}
        for sid in cfg.scenarios:
            per_scenario[sid] = {
                "completed": sum(1 for c in completed if c["scenario"] == sid),
                "expected": sum(1 for e in expected if e[0] == sid),
            }
        return {
            "run_id": cfg.run_id,
            "run_kind": cfg.run_kind,
            "expected_configurations": len(expected),
            "completed_configurations": len(completed_keys),
            "missing": missing,
            "per_scenario": per_scenario,
            "network_fingerprint": self.network_fingerprint,
            "network_semantic_fingerprint": self.network_semantic_fingerprint,
            "updated_utc": utc_now_iso(),
        }


def _se(values) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return float("nan")
    return float(arr.std(ddof=1) / math.sqrt(arr.size))


def setup_logger(run_id: str, log_dir: Path = LOGS_DIR) -> logging.Logger:
    ensure_dir(log_dir)
    logger = logging.getLogger(f"experiment.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_dir / f"{run_id}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def read_scenario_results(scenario_dir: Path) -> pd.DataFrame:
    path = Path(scenario_dir) / "experiment_results.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_run_results(run_dir: Path) -> pd.DataFrame:
    """Load and concatenate all scenario CSVs of one run directory."""
    run_dir = Path(run_dir)
    frames = []
    for path in sorted(run_dir.glob("*/experiment_results.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No scenario experiment_results.csv found under {run_dir}")
    df = pd.concat(frames, ignore_index=True)
    return df


def validate_results(df: pd.DataFrame, config: ExperimentConfig | None = None) -> dict:
    """Structural validation of a result table (no simulations)."""
    issues: list[str] = []
    missing_cols = [c for c in RESULT_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"missing columns: {missing_cols}")
    core = [
        "run_id", "scenario", "K", "objective_id", "mode", "network_fingerprint",
        "spread_A", "spread_B", "net_advantage", "blocked_B", "final_seed_set",
        "opponent_seed_set",
    ]
    empty = [c for c in core if c in df.columns and df[c].isna().all()]
    if empty:
        issues.append(f"entirely empty core columns: {empty}")
    key_cols = ["scenario", "K", "objective_id", "mode"]
    if all(c in df.columns for c in key_cols):
        dup = df.duplicated(key_cols).sum()
        if dup:
            issues.append(f"{int(dup)} duplicate configuration rows")
        if config is not None:
            got = {(r.scenario, int(r.K), r.objective_id, r.mode) for r in df.itertuples(index=False)}
            expected = set(config.expected_keys())
            missing = sorted(expected - got)
            extra = sorted(got - expected)
            if missing:
                issues.append(f"{len(missing)} expected configurations missing")
            if extra:
                issues.append(f"{len(extra)} unexpected configurations present")
    for col in ("network_fingerprint", "run_id"):
        if col in df.columns and df[col].nunique(dropna=False) > 1:
            issues.append(f"multiple {col} values in one result table")
    if all(c in df.columns for c in ("spread_A", "spread_B", "net_advantage")):
        err = (df["spread_A"] - df["spread_B"] - df["net_advantage"]).abs()
        if (err > 1e-6).any():
            issues.append("net_advantage != spread_A - spread_B in some rows")
    if "blocked_B" in df.columns and (df["blocked_B"] < -1e-9).any():
        issues.append("negative blocked_B values")
    return {"ok": not issues, "issues": issues, "n_rows": len(df)}
