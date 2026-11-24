#!/usr/bin/env python3
"""Run the SSA O1/O4/O6 vs. notebook-baseline + FIBMM benchmark.

Modes
-----
plan     : print the experiment matrix, network identity and reuse plan.
verify   : validate the network, opponent orders, prior SSA reuse, and a
           deterministic SSA reproduction check.  No full experiment is run.
smoke    : run a reduced-budget smoke test for every algorithm on B1/K=10.
full     : run the full 4 x 5 x 12 matrix, reusing valid prior SSA results.
analyze  : build aggregated tables and figures from completed raw results.

The full mode requires ``--confirm-full``.  All writes stay inside
``experiments/benchmark_ssa_baselines_fairness/``.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from algorithms.core import (
    AGG_DIR, CHECKPOINT_DIR, CONFIG_PATH, EXP_DIR, FIBMM_DIAG_DIR, FIG_DIR,
    LEDGER_PATH, MANIFEST_PATH, MC_DIR, OPPONENT_SEEDS_PATH, PREPROCESSING_DIR,
    PREPROCESSING_PATH, PROJECT_ROOT, RAW_DIR, REPORT_DIR, RESULTS_PATH,
    SEED_SETS_PATH, SMOKE_DIR, append_ledger, candidate_pool_for,
    cell_signature, ensure_dirs, environment_block, evaluate_new_method,
    final_seeds, load_config, load_ledger, load_results, load_verified_network,
    make_evaluator, mc_path, method_spec, metrics_from_mc, opponent_orders,
    optimization_seeds, read_mc, save_ledger, save_results, scenario_number,
    sha256_obj, utc_now_iso, write_mc, write_network_reference,
    write_preprocessing,
)
from common.objectives import VARIANTS  # noqa: E402
from common.ssa import run_discrete_ssa  # noqa: E402
from common.utils import atomic_write_csv, atomic_write_json  # noqa: E402
from algorithms.fibmm import FIBMMConfig, detect_communities, select_fibmm
from algorithms.notebook_baselines import (
    celf_lazy_greedy, compute_ncge_centrality_scores, compute_signed_pagerank,
    crieg_seeds, ncge_candidate_pool, random_seeds, run_ssa_notebook,
    signed_ga_seeds, signed_pagerank_seeds, top_degree_seeds,
)
from algorithms.reuse import (
    build_reused_row, load_prior_ssa_rows, prior_run_dir, validate_prior_row,
)

RESULT_COLUMNS = [
    "experiment_id", "experiment_version", "method_id", "method_name",
    "method_group", "method_version", "scenario", "scenario_name", "K",
    "opponent_seed_set", "seed_set", "seed_set_ordered",
    "candidate_pool_size", "candidate_pool_signature",
    "selection_runtime_sec", "final_evaluation_runtime_sec", "total_runtime_sec",
    "preprocessing_runtime_sec",
    "optimization_objective_evaluations", "optimization_mc_runs",
    "optimization_simulations", "optimization_cache_hits",
    "optimization_cache_misses", "final_mc_runs", "final_evaluation_simulations",
    "spread_A", "spread_B", "net_advantage", "blocked_B", "baseline_B",
    "spread_A_se", "spread_B_se", "net_advantage_se",
    "net_advantage_ci_low", "net_advantage_ci_high",
    "blocked_B_se", "blocked_B_ci_low", "blocked_B_ci_high",
    "network_file_sha256", "influence_semantic_sha256", "influence_direction",
    "config_signature_hash", "source", "source_detail",
    "prior_config_signature_hash", "method_specific_budget",
    "optimization_budget_note", "fibmm_pi_total", "fibmm_tau",
    "fibmm_sigma", "fibmm_crp_sets", "fibmm_n_communities",
    "status", "timestamp", "mc_file", "checkpoint_file",
]

SCENARIO_NAMES = {
    "B1": "Random opponent",
    "B2": "Degree-based opponent",
    "B3": "Positive PageRank opponent",
    "B4": "Community-concentrated opponent",
}


def _json_list(values) -> str:
    return json.dumps([int(x) for x in values], separators=(",", ":"))


def _json_obj(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class BenchmarkRunner:
    def __init__(self, config: dict, output_root: Path = RAW_DIR,
                 smoke: bool = False, only_methods: list[str] | None = None,
                 only_scenarios: list[str] | None = None,
                 only_k: list[int] | None = None):
        self.config = config
        self.output_root = Path(output_root)
        self.smoke = bool(smoke)
        self.only_methods = set(only_methods) if only_methods else None
        self.only_scenarios = set(only_scenarios) if only_scenarios else None
        self.only_k = set(int(k) for k in only_k) if only_k else None
        self.results_path = self.output_root / "results.csv"
        self.ledger_path = self.output_root / "ledger.csv"
        self.manifest_path = self.output_root / "manifest.json"
        self.mc_dir = self.output_root / "mc"
        self.checkpoint_dir = self.output_root / "checkpoints"
        self.diag_dir = self.output_root / "fibmm_diagnostics"
        self.preprocessing_dir = self.output_root / "preprocessing"
        self.preprocessing_path = self.output_root / "preprocessing.json"
        for path in (self.output_root, self.mc_dir, self.checkpoint_dir,
                     self.diag_dir, self.preprocessing_dir):
            path.mkdir(parents=True, exist_ok=True)

        self.logger = self._setup_logger()
        self.df, self.G, self.nodes, self.network_meta = load_verified_network(config)
        write_network_reference(self.network_meta)
        self.simulator, self.evaluator = make_evaluator(
            self.G, self.nodes, self.network_meta, config
        )
        self.preprocessing_runtimes = {}
        self._signed_pagerank_scores = None
        self._ncge_scores = None
        self._fibmm_communities = None
        self._prior_rows = None
        self.orders = self._get_opponent_orders()
        self._prior_dir = prior_run_dir(config)
        self._write_opponent_seed_table()

    # ------------------------------------------------------------------
    def _setup_logger(self) -> logging.Logger:
        log_dir = self.output_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(f"benchmark.{id(self)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        for h in list(logger.handlers):
            logger.removeHandler(h)
            h.close()
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
        return logger

    # ------------------------------------------------------------------
    def _cache_meta(self, name: str, extra: dict | None = None) -> dict:
        payload = {
            "name": name,
            "network_file_sha256": self.network_meta["file_sha256"],
            "influence_semantic_sha256": self.network_meta["influence_semantic_sha256"],
            "config_version": self.config["version"],
        }
        if extra:
            payload.update(extra)
        return payload

    def _load_or_compute(self, name: str, filename: str, meta: dict, compute_fn):
        path = self.preprocessing_dir / filename
        meta_path = self.preprocessing_dir / f"{filename}.meta.json"
        if path.exists() and meta_path.exists():
            try:
                old_meta = _read_json(meta_path)
                if old_meta == meta:
                    payload = _read_json(path)
                    self.logger.info("Preprocessing cache hit: %s", name)
                    return payload
            except Exception:
                pass
        self.logger.info("Preprocessing cache miss: %s -> computing", name)
        t0 = time.perf_counter()
        payload = compute_fn()
        runtime = time.perf_counter() - t0
        self.preprocessing_runtimes[name] = float(runtime)
        atomic_write_json(path, payload)
        atomic_write_json(meta_path, meta)
        self.logger.info("Preprocessing %s done in %.2fs", name, runtime)
        return payload

    def _get_opponent_orders(self) -> dict[str, list[int]]:
        meta = self._cache_meta("opponent_orders", {
            "master_seed": int(self.config["master_seed"]),
            "max_k": int(max(self.config["k_values"])),
            "scenarios": list(self.config["scenarios"]),
        })
        payload = self._load_or_compute(
            "opponent_orders", "opponent_orders.json", meta,
            lambda: opponent_orders(self.G, self.nodes, self.config),
        )
        return {sid: [int(x) for x in order] for sid, order in payload.items()}

    def _write_opponent_seed_table(self) -> None:
        rows = []
        for sid in self.config["scenarios"]:
            for k in self.config["k_values"]:
                B = self.orders[sid][: int(k)]
                rows.append({
                    "scenario": sid,
                    "scenario_name": SCENARIO_NAMES[sid],
                    "K": int(k),
                    "opponent_seed_set": _json_list(B),
                    "opponent_order_signature": f"{len(self.orders[sid])}:{sha256_obj(self.orders[sid])[:16]}",
                })
        path = self.output_root / "opponent_seeds.csv"
        atomic_write_csv(pd.DataFrame(rows), path)

    def _get_signed_pagerank(self) -> dict[int, float]:
        if self._signed_pagerank_scores is not None:
            return self._signed_pagerank_scores
        meta = self._cache_meta("signed_pagerank", {
            "alpha": 0.85, "max_iter": 200, "tol": 1e-10,
            "weight": "abs_rating", "teleport": "positive_layer_only",
        })
        payload = self._load_or_compute(
            "signed_pagerank", "signed_pagerank_scores.json", meta,
            lambda: {str(u): float(v) for u, v in compute_signed_pagerank(
                self.G, self.nodes, alpha=0.85, max_iter=200, tol=1e-10
            ).items()},
        )
        self._signed_pagerank_scores = {int(k): float(v) for k, v in payload.items()}
        return self._signed_pagerank_scores

    def _get_ncge_scores(self) -> dict[int, float]:
        if self._ncge_scores is not None:
            return self._ncge_scores
        cfg = self.config["baselines"]["ncge"]
        meta = self._cache_meta("ncge_centrality", {
            "alpha": float(cfg["alpha"]), "beta": float(cfg["beta"]),
            "gamma": float(cfg["gamma"]),
            "betweenness_samples": int(cfg["betweenness_samples"]),
            "centrality_seed": int(cfg["centrality_seed"]),
        })
        payload = self._load_or_compute(
            "ncge_centrality", "ncge_centrality_scores.json", meta,
            lambda: {str(u): float(v) for u, v in compute_ncge_centrality_scores(
                self.G, self.nodes,
                alpha=float(cfg["alpha"]), beta=float(cfg["beta"]),
                gamma=float(cfg["gamma"]),
                k_approx=int(cfg["betweenness_samples"]),
                seed=int(cfg["centrality_seed"]),
            ).items()},
        )
        self._ncge_scores = {int(k): float(v) for k, v in payload.items()}
        return self._ncge_scores

    def _get_fibmm_communities(self) -> list[set[int]]:
        if self._fibmm_communities is not None:
            return self._fibmm_communities
        cfg = self.config["fibmm"]
        meta = self._cache_meta("fibmm_communities", {
            "min_fraction": float(cfg["community_min_fraction"]),
            "seed": int(cfg["community_seed"]),
            "projection": "undirected_unweighted_all_influence_edges",
        })
        payload = self._load_or_compute(
            "fibmm_communities", "fibmm_communities.json", meta,
            lambda: [sorted(int(u) for u in c) for c in detect_communities(
                self.G, self.nodes,
                min_fraction=float(cfg["community_min_fraction"]),
                seed=int(cfg["community_seed"]),
            )],
        )
        self._fibmm_communities = [set(int(u) for u in c) for c in payload]
        return self._fibmm_communities

    # ------------------------------------------------------------------
    def _load_results(self) -> pd.DataFrame:
        if self.results_path.exists():
            return pd.read_csv(self.results_path)
        return pd.DataFrame(columns=RESULT_COLUMNS)

    def _save_results(self, df: pd.DataFrame) -> None:
        df = df.reindex(columns=RESULT_COLUMNS)
        atomic_write_csv(df, self.results_path)

    def _load_ledger(self) -> pd.DataFrame:
        if self.ledger_path.exists():
            return pd.read_csv(self.ledger_path)
        return pd.DataFrame(columns=[
            "method_id", "scenario", "K", "status", "reason", "source",
            "selection_runtime_sec", "final_mc_runs", "timestamp",
        ])

    def _save_ledger(self, df: pd.DataFrame) -> None:
        atomic_write_csv(df, self.ledger_path)

    def _append_ledger(self, row: dict) -> None:
        ledger = self._load_ledger()
        key = (row.get("method_id"), row.get("scenario"), int(row.get("K")))
        if not ledger.empty:
            mask = (
                (ledger["method_id"] == key[0])
                & (ledger["scenario"] == key[1])
                & (ledger["K"].astype(int) == key[2])
            )
            ledger = ledger.loc[~mask]
        row = dict(row)
        row.setdefault("timestamp", utc_now_iso())
        ledger = pd.concat([ledger, pd.DataFrame([row])], ignore_index=True)
        self._save_ledger(ledger)

    # ------------------------------------------------------------------
    def expected_cells(self) -> list[tuple[str, str, int]]:
        methods = [m["id"] for m in self.config["methods"]]
        cells = [
            (m, s, int(k))
            for m in methods
            for s in self.config["scenarios"]
            for k in self.config["k_values"]
        ]
        if self.only_methods is not None:
            cells = [c for c in cells if c[0] in self.only_methods]
        if self.only_scenarios is not None:
            cells = [c for c in cells if c[1] in self.only_scenarios]
        if self.only_k is not None:
            cells = [c for c in cells if c[2] in self.only_k]
        return cells

    def _existing_row(self, results: pd.DataFrame, method_id: str,
                      scenario: str, k: int):
        if results.empty:
            return None
        mask = (
            (results["method_id"] == method_id)
            & (results["scenario"] == scenario)
            & (results["K"].astype(int) == int(k))
        )
        sub = results.loc[mask]
        if sub.empty:
            return None
        return sub.iloc[-1]

    # ------------------------------------------------------------------
    def _selection_budget_note(self, method_id: str) -> str:
        notes = {
            "SSA_O1": "SSA: 12 initial + 10 x 12 = 132 objective evaluations; 15 common MC seeds per evaluation.",
            "SSA_O4": "SSA: 12 initial + 10 x 12 = 132 objective evaluations; 15 common MC seeds per evaluation.",
            "SSA_O6": "SSA: 12 initial + 10 x 12 = 132 objective evaluations; 15 common MC seeds per evaluation.",
            "SSA_Notebook": "Notebook SSA: 12 initial + 10 x 12 = 132 objective evaluations; 15 common MC seeds per evaluation; notebook seed schedule.",
            "Random": "No diffusion evaluations; one uniform sample.",
            "TopDegree": "No diffusion evaluations; deterministic out-degree ranking.",
            "SignedPageRank": "No diffusion evaluations; one-time two-layer signed PageRank preprocessing.",
            "CRIEG": "2000 reverse-reachability sketches on the positive subgraph; no Monte Carlo diffusion during selection.",
            "NCGE": "One-time centrality preprocessing; lazy-greedy marginal-gain selection on the NCGE pool with 15 MC seeds per evaluation.",
            "SignedGA": "30 sparse linear-system / gradient steps; no Monte Carlo diffusion during selection.",
            "CELF": "Lazy-greedy marginal-gain selection on the weighted candidate pool with 15 MC seeds per evaluation.",
            "FIBMM": "Paper's CRP sampling and two-stage fairness-aware greedy; no Monte Carlo MT diffusion during selection.",
        }
        return notes.get(method_id, "")

    def _method_specific_budget(self, method_id: str, selection_info: dict) -> dict:
        if method_id == "CRIEG":
            return {"num_sketches": selection_info.get("num_sketches")}
        if method_id == "NCGE":
            return {"centrality": "0.5 degree + 0.3 closeness + 0.2 betweenness",
                    "pool_size": selection_info.get("candidate_pool_size")}
        if method_id == "SignedGA":
            return {"steps": selection_info.get("steps"),
                    "linear_solves": selection_info.get("linear_solves")}
        if method_id == "CELF":
            return {"candidate_pool_size": selection_info.get("candidate_pool_size"),
                    "heap_refreshes": selection_info.get("heap_refreshes")}
        if method_id == "FIBMM":
            plan = selection_info.get("sample_plan", {})
            return {
                "h": selection_info.get("h"),
                "epsilon": selection_info.get("epsilon"),
                "l": selection_info.get("l"),
                "crp_sets": selection_info.get("crp_sets"),
                "theta_raw_total": plan.get("theta_raw_total"),
                "theta_total_capped": plan.get("theta_total"),
                "stage1_count": selection_info.get("stage1_count"),
                "stage2_count": selection_info.get("stage2_count"),
                "n_communities": selection_info.get("n_communities"),
            }
        if method_id.startswith("SSA"):
            return {"population": self.config["ssa"]["population"],
                    "iterations": self.config["ssa"]["iterations"],
                    "mutation_rate": self.config["ssa"]["mutation_rate"]}
        return {}

    # ------------------------------------------------------------------
    def _select(self, method_id: str, scenario: str, k: int, B: list[int]
                ) -> tuple[list[int], dict, float, dict]:
        """Return ``(A, selection_info, selection_runtime, counters)``."""
        cfg = self.config
        opt_seeds = optimization_seeds(cfg, scenario)
        n = scenario_number(scenario)
        t0 = time.perf_counter()
        counters = {
            "objective_evaluations": np.nan,
            "simulations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        selection_info: dict = {}

        if method_id in ("SSA_O1", "SSA_O4", "SSA_O6"):
            objective = method_id.replace("SSA_", "")
            pool, _scores, pool_meta = candidate_pool_for(
                self.G, self.nodes, B, k, scenario, cfg
            )
            scorer = lambda A, Bv, sv, variant, kk: self.evaluator.evaluate_objective(
                A, Bv, sv, variant, kk
            )
            variant_index = list(VARIANTS.keys()).index(objective)
            seed = int(cfg["master_seed"]) + 500000 + n * 1000 + int(k) * 10 + variant_index
            A, score, history, info = run_discrete_ssa(
                pool, B, k, opt_seeds, objective, scorer,
                population_size=int(cfg["ssa"]["population"]),
                iterations=int(cfg["ssa"]["iterations"]),
                mutation_rate=float(cfg["ssa"]["mutation_rate"]),
                random_seed=seed,
            )
            selection_info = {
                "pool_meta": pool_meta,
                "ssa_score": float(score),
                "ssa_info": info,
                "ssa_seed": int(seed),
                "history": history.to_dict("records"),
            }
            counters["objective_evaluations"] = int(info["objective_evaluations"])
            counters["simulations"] = int(info["objective_evaluations"]) * int(cfg["optimization_mc_runs"]) * 2
            A = [int(x) for x in A]
            pool_used = pool
            pool_sig = None

        elif method_id == "SSA_Notebook":
            pool, _scores, pool_meta = candidate_pool_for(
                self.G, self.nodes, B, k, scenario, cfg
            )
            scorer = lambda A, Bv, sv, variant, kk: self.evaluator.evaluate_objective(
                A, Bv, sv, "O1", kk
            )
            seed = int(cfg["master_seed"]) + int(cfg["ssa"]["notebook_ssa_seed_offset"]) + n
            A, score, history, info = run_ssa_notebook(
                pool, B, k, opt_seeds, scorer,
                population=int(cfg["ssa"]["population"]),
                iterations=int(cfg["ssa"]["iterations"]),
                mutation_rate=float(cfg["ssa"]["mutation_rate"]),
                random_seed=seed,
            )
            selection_info = {
                "pool_meta": pool_meta,
                "ssa_score": float(score),
                "ssa_info": info,
                "ssa_seed": int(seed),
                "history": history.to_dict("records"),
            }
            counters["objective_evaluations"] = int(info["objective_evaluations"])
            counters["simulations"] = int(info["objective_evaluations"]) * int(cfg["optimization_mc_runs"]) * 2
            A = [int(x) for x in A]
            pool_used = pool
            pool_sig = None

        elif method_id == "Random":
            A = random_seeds(self.nodes, B, k, n, int(cfg["master_seed"]))
            pool_used = [int(u) for u in self.nodes if int(u) not in set(B)]
            pool_sig = None
            selection_info = {"seed_formula": "master_seed + 50000 + K*100 + scenario_number"}

        elif method_id == "TopDegree":
            A = top_degree_seeds(self.G, self.nodes, B, k)
            pool_used = [int(u) for u in self.nodes if int(u) not in set(B)]
            pool_sig = None
            selection_info = {"ranking": "influence out-degree descending, ties by node id"}

        elif method_id == "SignedPageRank":
            scores = self._get_signed_pagerank()
            A = signed_pagerank_seeds(self.nodes, B, k, scores)
            pool_used = [int(u) for u in self.nodes if int(u) not in set(B)]
            pool_sig = None
            selection_info = {"ranking": "positive-layer minus negative-layer signed PageRank"}

        elif method_id == "CRIEG":
            seed = int(cfg["master_seed"]) + int(cfg["baselines"]["crieg"]["seed_offset"]) + k * 100 + n
            A, info = crieg_seeds(
                self.G, self.nodes, B, k,
                num_sketches=int(cfg["baselines"]["crieg"]["num_sketches"]),
                random_seed=seed,
            )
            pool_used = [int(u) for u in self.nodes if int(u) not in set(B)]
            pool_sig = None
            selection_info = {**info, "crieg_seed": int(seed)}
            counters["objective_evaluations"] = 0

        elif method_id == "NCGE":
            scores = self._get_ncge_scores()
            pool_size = int(k) * int(cfg["ssa"]["pool_ratio"])
            pool = ncge_candidate_pool(self.nodes, B, pool_size, scores)
            scorer = lambda A: self.evaluator.evaluate_objective(A, B, opt_seeds, "O1", k)
            stats_before = self.evaluator.cache_stats()
            A, history, info = celf_lazy_greedy(pool, B, k, scorer, "NCGE")
            stats_after = self.evaluator.cache_stats()
            pool_used = pool
            pool_sig = None
            selection_info = {**info, "history": history}
            counters["objective_evaluations"] = int(info["objective_evaluations"])
            counters["simulations"] = int(stats_after["simulations"] - stats_before["simulations"])
            counters["cache_hits"] = int(stats_after["cache_hits"] - stats_before["cache_hits"])
            counters["cache_misses"] = int(stats_after["cache_misses"] - stats_before["cache_misses"])

        elif method_id == "SignedGA":
            A, info = signed_ga_seeds(
                self.G, self.nodes, B, k,
                steps=int(cfg["baselines"]["signed_ga"]["steps"]),
                damping=float(cfg["baselines"]["signed_ga"]["damping"]),
                ridge=float(cfg["baselines"]["signed_ga"]["ridge"]),
            )
            pool_used = [int(u) for u in self.nodes if int(u) not in set(B)]
            pool_sig = None
            selection_info = info
            counters["objective_evaluations"] = 0

        elif method_id == "CELF":
            pool, _scores, pool_meta = candidate_pool_for(
                self.G, self.nodes, B, k, scenario, cfg
            )
            scorer = lambda A: self.evaluator.evaluate_objective(A, B, opt_seeds, "O1", k)
            stats_before = self.evaluator.cache_stats()
            A, history, info = celf_lazy_greedy(pool, B, k, scorer, "CELF")
            stats_after = self.evaluator.cache_stats()
            pool_used = pool
            pool_sig = None
            selection_info = {**info, "history": history, "pool_meta": pool_meta}
            counters["objective_evaluations"] = int(info["objective_evaluations"])
            counters["simulations"] = int(stats_after["simulations"] - stats_before["simulations"])
            counters["cache_hits"] = int(stats_after["cache_hits"] - stats_before["cache_hits"])
            counters["cache_misses"] = int(stats_after["cache_misses"] - stats_before["cache_misses"])

        elif method_id == "FIBMM":
            fib_cfg = cfg["fibmm"]
            seed = int(cfg["master_seed"]) + int(fib_cfg["crp_seed_offset"]) + n * 1000 + int(k)
            fibmm_cfg = FIBMMConfig(
                h=float(fib_cfg["h"]),
                epsilon=float(fib_cfg["epsilon"]),
                l=int(fib_cfg["l"]),
                community_min_fraction=float(fib_cfg["community_min_fraction"]),
                community_seed=int(fib_cfg["community_seed"]),
                max_crp_samples=int(fib_cfg["max_crp_samples_per_config"]),
                min_samples_per_community=int(fib_cfg["min_samples_per_community"]),
                crp_seed=seed,
            )
            A, info = select_fibmm(
                self.G, self.nodes, B, k, fibmm_cfg,
                communities=self._get_fibmm_communities(),
            )
            pool_used = [int(u) for u in self.nodes if int(u) not in set(B)]
            pool_sig = None
            selection_info = info
            counters["objective_evaluations"] = 0

        else:
            raise ValueError(f"Unknown method {method_id!r}")

        selection_runtime = time.perf_counter() - t0
        return [int(x) for x in A], selection_info, selection_runtime, counters

    # ------------------------------------------------------------------
    def _validate_seed_set(self, method_id: str, A: list[int], B: list[int],
                           k: int, pool: list[int] | None) -> None:
        if len(A) != int(k):
            raise ValueError(f"{method_id}: selected {len(A)} seeds, expected {k}")
        if len(set(A)) != len(A):
            raise ValueError(f"{method_id}: duplicate seeds")
        if set(A) & set(B):
            raise ValueError(f"{method_id}: selected an opponent seed")
        if pool is not None and not set(A).issubset(set(pool)):
            raise ValueError(f"{method_id}: selected outside its candidate pool")
        if any(int(u) < 0 or int(u) >= len(self.nodes) for u in A):
            raise ValueError(f"{method_id}: seed outside node range")

    def _checkpoint_payload(self, method_id: str, scenario: str, k: int,
                            B: list[int], A: list[int], selection_info: dict,
                            metrics: dict, signature_hash: str,
                            source: str) -> dict:
        return {
            "schema_version": 1,
            "experiment_id": self.config["experiment_id"],
            "method_id": method_id,
            "scenario": scenario,
            "K": int(k),
            "opponent_seed_set": [int(x) for x in B],
            "seed_set": [int(x) for x in A],
            "selection_info": selection_info,
            "metrics": {
                key: (float(value) if isinstance(value, (int, float, np.floating)) else value)
                for key, value in metrics.items()
                if key not in ("net_advantage_runs", "spread_A_runs", "spread_B_runs",
                               "baseline_B_runs", "blocked_B_runs")
            },
            "network_file_sha256": self.network_meta["file_sha256"],
            "influence_semantic_sha256": self.network_meta["influence_semantic_sha256"],
            "config_signature_hash": signature_hash,
            "source": source,
            "created_utc": utc_now_iso(),
        }

    # ------------------------------------------------------------------
    def _reuse_prior_cell(self, method_id: str, scenario: str, k: int,
                          B: list[int], results: pd.DataFrame,
                          signature_hash: str) -> bool:
        objective = method_id.replace("SSA_", "")
        if objective not in self.config["ssa"]["objectives"]:
            return False
        if self._prior_rows is None:
            self._prior_rows = load_prior_ssa_rows(self.config)
        sub = self._prior_rows[
            (self._prior_rows["scenario"] == scenario)
            & (self._prior_rows["K"].astype(int) == int(k))
            & (self._prior_rows["objective_id"] == objective)
        ]
        if len(sub) != 1:
            raise ValueError(f"Expected one prior SSA row for {scenario}/K={k}/{objective}")
        row = sub.iloc[0]
        pool, _scores, _meta = candidate_pool_for(
            self.G, self.nodes, B, k, scenario, self.config
        )
        # The prior run used common.candidates.pool_signature; recompute it
        # exactly rather than relying on a display-only signature.
        from common.candidates import pool_signature
        pool_sig = pool_signature(pool)
        mc, metrics = validate_prior_row(
            row, self.config, self.network_meta, B, k, pool, pool_sig,
            self._prior_dir,
        )
        spec = method_spec(self.config, method_id)
        new_row = build_reused_row(
            row, mc, metrics, self.config, self.network_meta, method_id, spec,
            scenario, k, B, pool, pool_sig, signature_hash, self._prior_dir,
        )
        # Store a self-contained copy of the per-run MC table.
        new_mc_path = self.mc_dir / f"{method_id}_{scenario}_K{int(k)}.csv"
        atomic_write_csv(mc, new_mc_path)
        new_row["mc_file"] = str(new_mc_path.relative_to(self.output_root))
        checkpoint_path = self.checkpoint_dir / f"{method_id}_{scenario}_K{int(k)}.json"
        payload = self._checkpoint_payload(
            method_id, scenario, k, B,
            [int(x) for x in json.loads(new_row["seed_set"])],
            {"reused_from": new_row["source_detail"]}, metrics,
            signature_hash, "reused_prior_ssa_only",
        )
        atomic_write_json(checkpoint_path, payload)
        new_row["checkpoint_file"] = str(checkpoint_path.relative_to(self.output_root))
        results = results.loc[~(
            (results["method_id"] == method_id)
            & (results["scenario"] == scenario)
            & (results["K"].astype(int) == int(k))
        )]
        results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
        self._save_results(results)
        self._append_ledger({
            "method_id": method_id, "scenario": scenario, "K": int(k),
            "status": "reused", "reason": "validated prior SSA-only result",
            "source": new_row["source_detail"],
            "selection_runtime_sec": new_row["selection_runtime_sec"],
            "final_mc_runs": new_row["final_mc_runs"],
        })
        self.logger.info(
            "REUSED %s %s K=%d: NetAdv=%.3f (A=%.1f B=%.1f)",
            method_id, scenario, k, new_row["net_advantage"],
            new_row["spread_A"], new_row["spread_B"],
        )
        return True

    # ------------------------------------------------------------------
    def run_cell(self, method_id: str, scenario: str, k: int,
                 force: bool = False) -> dict:
        B = [int(x) for x in self.orders[scenario][:int(k)]]
        signature_hash = cell_signature(
            self.config, method_id, scenario, int(k), B, self.network_meta
        )
        results = self._load_results()
        existing = self._existing_row(results, method_id, scenario, int(k))
        if existing is not None and not force:
            if str(existing.get("config_signature_hash", "")) == signature_hash and str(
                existing.get("status", "")
            ) in ("completed_valid", "reused"):
                self._append_ledger({
                    "method_id": method_id, "scenario": scenario, "K": int(k),
                    "status": str(existing["status"]),
                    "reason": "existing result has matching signature",
                    "source": str(existing.get("source", "")),
                    "selection_runtime_sec": float(existing.get("selection_runtime_sec", np.nan)),
                    "final_mc_runs": int(existing.get("final_mc_runs", 0)),
                })
                self.logger.info("SKIP %s %s K=%d (valid existing result)", method_id, scenario, k)
                return existing.to_dict()

        # Reuse the authoritative SSA-only results when valid.
        if method_id in ("SSA_O1", "SSA_O4", "SSA_O6") and not force:
            try:
                if self._reuse_prior_cell(method_id, scenario, k, B, results, signature_hash):
                    return self._existing_row(self._load_results(), method_id, scenario, int(k)).to_dict()
            except Exception as exc:
                self.logger.warning(
                    "Prior SSA reuse rejected for %s %s K=%d: %s. Recomputing.",
                    method_id, scenario, k, exc,
                )
                self._append_ledger({
                    "method_id": method_id, "scenario": scenario, "K": int(k),
                    "status": "invalidated", "reason": f"reuse validation failed: {exc}",
                    "source": "prior_ssa_only", "selection_runtime_sec": np.nan,
                    "final_mc_runs": 0,
                })

        self.logger.info("RUN %s %s K=%d", method_id, scenario, k)
        stats_before_selection = self.evaluator.cache_stats()
        A, selection_info, selection_runtime, counters = self._select(
            method_id, scenario, k, B
        )
        stats_after_selection = self.evaluator.cache_stats()
        pool_used = None
        if method_id in ("SSA_O1", "SSA_O4", "SSA_O6", "SSA_Notebook", "CELF"):
            pool_used = selection_info.get("pool_meta") and candidate_pool_for(
                self.G, self.nodes, B, k, scenario, self.config
            )[0]
        if method_id == "NCGE":
            pool_used = ncge_candidate_pool(
                self.nodes, B, int(k) * int(self.config["ssa"]["pool_ratio"]),
                self._get_ncge_scores(),
            )
        self._validate_seed_set(method_id, A, B, k, pool_used)

        # Use measured selection-only work for the optimization budget.  This
        # avoids attributing final-evaluation simulations to the optimizer.
        counters["simulations"] = int(
            stats_after_selection["simulations"] - stats_before_selection["simulations"]
        )
        counters["cache_hits"] = int(
            stats_after_selection["cache_hits"] - stats_before_selection["cache_hits"]
        )
        counters["cache_misses"] = int(
            stats_after_selection["cache_misses"] - stats_before_selection["cache_misses"]
        )

        stats_before_eval = self.evaluator.cache_stats()
        t_eval = time.perf_counter()
        metrics, mc = evaluate_new_method(self.evaluator, A, B, self.config)
        final_eval_runtime = time.perf_counter() - t_eval
        stats_after_eval = self.evaluator.cache_stats()

        mc_path = self.mc_dir / f"{method_id}_{scenario}_K{int(k)}.csv"
        atomic_write_csv(mc, mc_path)
        spec = method_spec(self.config, method_id)
        pool_meta = selection_info.get("pool_meta", {}) if isinstance(selection_info, dict) else {}
        if method_id == "NCGE":
            pool_size = int(k) * int(self.config["ssa"]["pool_ratio"])
            pool_sig = ""
        elif pool_used is not None:
            from common.candidates import pool_signature
            pool_sig = pool_signature(pool_used)
        else:
            pool_sig = ""
        fibmm_fields = {}
        if method_id == "FIBMM":
            fibmm_fields = {
                "fibmm_pi_total": float(selection_info.get("pi_total", np.nan)),
                "fibmm_tau": float(selection_info.get("tau", np.nan)),
                "fibmm_sigma": float(selection_info.get("sigma", np.nan)),
                "fibmm_crp_sets": int(selection_info.get("crp_sets", 0)),
                "fibmm_n_communities": int(selection_info.get("n_communities", 0)),
            }
            diag_path = self.diag_dir / f"FIBMM_{scenario}_K{int(k)}.json"
            atomic_write_json(diag_path, selection_info)
        row = {
            "experiment_id": self.config["experiment_id"],
            "experiment_version": self.config["version"],
            "method_id": method_id,
            "method_name": spec["name"],
            "method_group": spec["group"],
            "method_version": {
                "SSA_O1": "common_ssa_reference_v1",
                "SSA_O4": "common_ssa_reference_v1",
                "SSA_O6": "common_ssa_reference_v1",
                "SSA_Notebook": "notebook_four_b_ssa_v1",
                "Random": "notebook_random_v1",
                "TopDegree": "notebook_top_degree_v1",
                "SignedPageRank": "notebook_signed_pagerank_v1",
                "CRIEG": "notebook_crieg_hong2021_v1",
                "NCGE": "notebook_ncge_liang2025_v1",
                "SignedGA": "notebook_signed_ga_chakraborty2025_v1",
                "CELF": "notebook_celf_o1_v1",
                "FIBMM": "paper_fibmm_xiao2025_adapted_v1",
            }.get(method_id, "unknown"),
            "scenario": scenario,
            "scenario_name": SCENARIO_NAMES[scenario],
            "K": int(k),
            "opponent_seed_set": _json_list(B),
            "seed_set": _json_list(sorted(A)),
            "seed_set_ordered": _json_list(A),
            "candidate_pool_size": int(len(pool_used)) if pool_used is not None else np.nan,
            "candidate_pool_signature": pool_sig,
            "selection_runtime_sec": float(selection_runtime),
            "final_evaluation_runtime_sec": float(final_eval_runtime),
            "total_runtime_sec": float(selection_runtime + final_eval_runtime),
            "preprocessing_runtime_sec": float(
                self.preprocessing_runtimes.get({
                    "SignedPageRank": "signed_pagerank",
                    "NCGE": "ncge_centrality",
                    "FIBMM": "fibmm_communities",
                }.get(method_id, ""), 0.0)
            ),
            "optimization_objective_evaluations": (
                int(counters["objective_evaluations"])
                if not (isinstance(counters["objective_evaluations"], float)
                        and math.isnan(counters["objective_evaluations"]))
                else np.nan
            ),
            "optimization_mc_runs": int(self.config["optimization_mc_runs"]),
            "optimization_simulations": int(counters["simulations"]),
            "optimization_cache_hits": int(counters["cache_hits"]),
            "optimization_cache_misses": int(counters["cache_misses"]),
            "final_mc_runs": int(self.config["final_mc_runs"]),
            "final_evaluation_simulations": int(
                stats_after_eval["simulations"] - stats_before_eval["simulations"]
            ),
            "spread_A": float(metrics["spread_A"]),
            "spread_B": float(metrics["spread_B"]),
            "net_advantage": float(metrics["net_advantage"]),
            "blocked_B": float(metrics["blocked_B"]),
            "baseline_B": float(metrics["baseline_B"]),
            "spread_A_se": float(metrics["spread_A_se"]),
            "spread_B_se": float(metrics["spread_B_se"]),
            "net_advantage_se": float(metrics["net_advantage_se"]),
            "net_advantage_ci_low": float(metrics["net_advantage_ci_low"]),
            "net_advantage_ci_high": float(metrics["net_advantage_ci_high"]),
            "blocked_B_se": float(metrics["blocked_B_se"]),
            "blocked_B_ci_low": float(metrics["blocked_B_ci_low"]),
            "blocked_B_ci_high": float(metrics["blocked_B_ci_high"]),
            "network_file_sha256": self.network_meta["file_sha256"],
            "influence_semantic_sha256": self.network_meta["influence_semantic_sha256"],
            "influence_direction": self.network_meta["influence_direction"],
            "config_signature_hash": signature_hash,
            "source": "new_computation",
            "source_detail": "",
            "prior_config_signature_hash": "",
            "method_specific_budget": _json_obj(
                self._method_specific_budget(method_id, selection_info)
            ),
            "optimization_budget_note": self._selection_budget_note(method_id),
            **fibmm_fields,
            "status": "completed_valid",
            "timestamp": utc_now_iso(),
            "mc_file": str(mc_path.relative_to(self.output_root)),
            "checkpoint_file": "",
        }
        checkpoint_path = self.checkpoint_dir / f"{method_id}_{scenario}_K{int(k)}.json"
        payload = self._checkpoint_payload(
            method_id, scenario, k, B, A, selection_info, metrics,
            signature_hash, "new_computation",
        )
        atomic_write_json(checkpoint_path, payload)
        row["checkpoint_file"] = str(checkpoint_path.relative_to(self.output_root))

        results = results.loc[~(
            (results["method_id"] == method_id)
            & (results["scenario"] == scenario)
            & (results["K"].astype(int) == int(k))
        )]
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
        self._save_results(results)
        self._append_ledger({
            "method_id": method_id, "scenario": scenario, "K": int(k),
            "status": "completed_valid", "reason": "new computation",
            "source": "new_computation",
            "selection_runtime_sec": float(selection_runtime),
            "final_mc_runs": int(self.config["final_mc_runs"]),
        })
        self.logger.info(
            "DONE %s %s K=%d: NetAdv=%.3f (A=%.1f B=%.1f) sel=%.2fs eval=%.2fs",
            method_id, scenario, k, row["net_advantage"], row["spread_A"],
            row["spread_B"], selection_runtime, final_eval_runtime,
        )
        return row

    # ------------------------------------------------------------------
    def warm_up_preprocessing(self) -> None:
        """Compute/cache one-time graph preprocessing before timed cells."""
        methods = (
            set(self.only_methods) if self.only_methods is not None
            else {m["id"] for m in self.config["methods"]}
        )
        if "SignedPageRank" in methods:
            self._get_signed_pagerank()
        if "NCGE" in methods:
            self._get_ncge_scores()
        if "FIBMM" in methods:
            self._get_fibmm_communities()
        self._write_preprocessing_summary()

    def run_all(self, force: bool = False) -> None:
        cells = self.expected_cells()
        self.logger.info("Expected cells: %d", len(cells))
        self.warm_up_preprocessing()
        failed = []
        for idx, (method_id, scenario, k) in enumerate(cells, start=1):
            try:
                self.run_cell(method_id, scenario, k, force=force)
            except Exception as exc:
                self.logger.exception("FAILED %s %s K=%d: %s", method_id, scenario, k, exc)
                failed.append({"method_id": method_id, "scenario": scenario,
                               "K": int(k), "error": f"{type(exc).__name__}: {exc}"})
                self._append_ledger({
                    "method_id": method_id, "scenario": scenario, "K": int(k),
                    "status": "failed", "reason": f"{type(exc).__name__}: {exc}",
                    "source": "new_computation", "selection_runtime_sec": np.nan,
                    "final_mc_runs": 0,
                })
                self._write_manifest(failed)
                raise
            if idx % 10 == 0 or idx == len(cells):
                self._write_manifest(failed)
                self.logger.info("Progress: %d/%d cells", idx, len(cells))
        self._write_manifest(failed)
        self._write_seed_sets()
        self._write_preprocessing_summary()

    def _write_manifest(self, failed: list[dict] | None = None) -> None:
        results = self._load_results()
        ledger = self._load_ledger()
        completed = [] if results.empty else [
            {"method_id": r.method_id, "scenario": r.scenario, "K": int(r.K),
             "status": r.status}
            for r in results.itertuples(index=False)
        ]
        expected = self.expected_cells()
        done_keys = {(c["method_id"], c["scenario"], c["K"]) for c in completed}
        missing = [
            {"method_id": m, "scenario": s, "K": int(k)}
            for (m, s, k) in expected if (m, s, k) not in done_keys
        ]
        status_counts = {}
        if not ledger.empty:
            status_counts = ledger["status"].value_counts().to_dict()
        manifest = {
            "experiment_id": self.config["experiment_id"],
            "experiment_version": self.config["version"],
            "generated_utc": utc_now_iso(),
            "network": {
                "path": self.network_meta["network_path"],
                "file_sha256": self.network_meta["file_sha256"],
                "trust_semantic_sha256": self.network_meta["trust_semantic_sha256"],
                "influence_semantic_sha256": self.network_meta["influence_semantic_sha256"],
                "nodes": self.network_meta["nodes"],
                "influence_arcs": self.network_meta["influence_arcs"],
                "influence_direction": self.network_meta["influence_direction"],
            },
            "expected_cells": len(expected),
            "completed_cells": len(completed),
            "missing_cells": missing,
            "failed_cells": failed or [],
            "ledger_status_counts": status_counts,
            "environment": environment_block(),
            "config_path": str(CONFIG_PATH.relative_to(EXP_DIR)),
            "results_path": str(self.results_path.relative_to(EXP_DIR)),
            "ledger_path": str(self.ledger_path.relative_to(EXP_DIR)),
        }
        atomic_write_json(self.manifest_path, manifest)

    def _write_seed_sets(self) -> None:
        results = self._load_results()
        rows = []
        for r in results.itertuples(index=False):
            rows.append({
                "method_id": r.method_id, "method_name": r.method_name,
                "scenario": r.scenario, "K": int(r.K),
                "rumour": "A", "seed_set": r.seed_set,
                "opponent_seed_set": r.opponent_seed_set,
            })
        for sid in self.config["scenarios"]:
            for k in self.config["k_values"]:
                rows.append({
                    "method_id": "OPPONENT_B", "method_name": "Rumour B opponent",
                    "scenario": sid, "K": int(k), "rumour": "B",
                    "seed_set": _json_list(self.orders[sid][:int(k)]),
                    "opponent_seed_set": _json_list(self.orders[sid][:int(k)]),
                })
        atomic_write_csv(pd.DataFrame(rows), self.output_root / "seed_sets.csv")

    def _write_preprocessing_summary(self) -> None:
        payload = {
            "runtimes_sec": self.preprocessing_runtimes,
            "environment": environment_block(),
            "network": {
                "file_sha256": self.network_meta["file_sha256"],
                "influence_semantic_sha256": self.network_meta["influence_semantic_sha256"],
            },
            "updated_utc": utc_now_iso(),
        }
        atomic_write_json(self.preprocessing_path, payload)


# ---------------------------------------------------------------------------
# Plan / verify / smoke
# ---------------------------------------------------------------------------
def print_plan(config: dict) -> None:
    print("Experiment:", config["experiment_id"], "version", config["version"])
    print("Network:", config["network"]["path"])
    print("  file sha256:", config["network"]["file_sha256"])
    print("  influence sha256:", config["network"]["influence_semantic_sha256"])
    print("  nodes/arcs:", config["network"]["nodes"], config["network"]["influence_arcs"])
    print("Scenarios:", config["scenarios"])
    print("K values:", config["k_values"])
    print("Methods:")
    for spec in config["methods"]:
        print(f"  {spec['id']:14s} {spec['group']:20s} {spec['name']}")
    print("Excluded:")
    for spec in config["excluded_methods"]:
        print(f"  {spec['id']}: {spec['reason']}")
    n = len(config["methods"]) * len(config["scenarios"]) * len(config["k_values"])
    print(f"Expected cells: {n}")
    print("Optimization MC:", config["optimization_mc_runs"],
          "Final MC:", config["final_mc_runs"])
    print("Prior SSA reuse:", config["ssa"]["reuse_prior_run"])


def verify(config: dict) -> None:
    """Validate network, opponent orders, reuse, and one SSA reproduction."""
    ensure_dirs()
    runner = BenchmarkRunner(config, output_root=RAW_DIR)
    print("Network identity: OK")
    print("  nodes:", runner.network_meta["nodes"],
          "influence arcs:", runner.network_meta["influence_arcs"])
    print("  direction check:", runner.network_meta["direction_check"])
    # Opponent consistency
    for sid in config["scenarios"]:
        for k in config["k_values"]:
            B = runner.orders[sid][:int(k)]
            assert len(B) == int(k) and len(set(B)) == int(k)
    print("Opponent nested orders: OK")
    # Prior SSA reuse validation for every cell
    prior = load_prior_ssa_rows(config)
    for sid in config["scenarios"]:
        for k in config["k_values"]:
            B = runner.orders[sid][:int(k)]
            pool, _scores, _meta = candidate_pool_for(
                runner.G, runner.nodes, B, int(k), sid, config
            )
            from common.candidates import pool_signature
            pool_sig = pool_signature(pool)
            for objective in config["ssa"]["objectives"]:
                sub = prior[
                    (prior["scenario"] == sid)
                    & (prior["K"].astype(int) == int(k))
                    & (prior["objective_id"] == objective)
                ]
                if len(sub) != 1:
                    raise AssertionError(f"Missing prior SSA row {sid}/{k}/{objective}")
                validate_prior_row(
                    sub.iloc[0], config, runner.network_meta, B, int(k),
                    pool, pool_sig, prior_run_dir(config),
                )
    print("Prior SSA O1/O4/O6 reuse validation: OK (60/60 rows)")
    # One deterministic SSA reproduction check (B1 K=10 O1).
    from common.ssa import run_discrete_ssa
    sid, k, objective = "B1", 10, "O1"
    B = runner.orders[sid][:k]
    pool, _scores, _meta = candidate_pool_for(runner.G, runner.nodes, B, k, sid, config)
    opt = optimization_seeds(config, sid)
    scorer = lambda A, Bv, sv, variant, kk: runner.evaluator.evaluate_objective(
        A, Bv, sv, variant, kk
    )
    seed = int(config["master_seed"]) + 500000 + scenario_number(sid) * 1000 + k * 10 + 0
    A, score, _hist, info = run_discrete_ssa(
        pool, B, k, opt, objective, scorer,
        population_size=int(config["ssa"]["population"]),
        iterations=int(config["ssa"]["iterations"]),
        mutation_rate=float(config["ssa"]["mutation_rate"]),
        random_seed=seed,
    )
    prior_row = prior[
        (prior["scenario"] == sid) & (prior["K"].astype(int) == k)
        & (prior["objective_id"] == objective)
    ].iloc[0]
    prior_A = {int(x) for x in json.loads(prior_row["ssa_seed_set"])}
    if {int(x) for x in A} != prior_A:
        raise AssertionError("Deterministic SSA reproduction does not match prior seed set")
    print("Deterministic SSA reproduction check (B1/K=10/O1): OK")
    print("VERIFY PASSED")


def smoke(config: dict) -> None:
    """Reduced-budget smoke test on B1/K=10 for every algorithm."""
    smoke_cfg = copy.deepcopy(config)
    smoke_cfg["scenarios"] = ["B1"]
    smoke_cfg["k_values"] = [10]
    smoke_cfg["optimization_mc_runs"] = 3
    smoke_cfg["final_mc_runs"] = 10
    smoke_cfg["ssa"]["population"] = 4
    smoke_cfg["ssa"]["iterations"] = 2
    smoke_cfg["baselines"]["crieg"]["num_sketches"] = 100
    smoke_cfg["baselines"]["ncge"]["betweenness_samples"] = 20
    smoke_cfg["fibmm"]["max_crp_samples_per_config"] = 2000
    smoke_cfg["fibmm"]["min_samples_per_community"] = 10
    smoke_cfg["baselines"]["signed_ga"]["steps"] = 3
    smoke_cfg["methods"] = [
        m for m in smoke_cfg["methods"]
        if m["id"] not in ("SSA_O1", "SSA_O4", "SSA_O6")
    ]
    ensure_dirs()
    runner = BenchmarkRunner(smoke_cfg, output_root=SMOKE_DIR / "raw_results", smoke=True)
    runner.run_all(force=True)
    print("SMOKE PASSED. Results:", runner.results_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["plan", "verify", "smoke", "full", "analyze"],
                        default="plan")
    parser.add_argument("--confirm-full", action="store_true",
                        help="Required to start the full experiment.")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Ignore existing results (logged; use sparingly).")
    parser.add_argument("--only-methods", default="",
                        help="Comma-separated method ids for a targeted run.")
    parser.add_argument("--only-scenarios", default="",
                        help="Comma-separated scenarios for a targeted run.")
    parser.add_argument("--only-k", default="",
                        help="Comma-separated K values for a targeted run.")
    args = parser.parse_args(argv)
    config = load_config()
    ensure_dirs()
    only_methods = [x.strip() for x in args.only_methods.split(",") if x.strip()]
    only_scenarios = [x.strip() for x in args.only_scenarios.split(",") if x.strip()]
    only_k = [int(x) for x in args.only_k.split(",") if x.strip()]
    if args.mode == "plan":
        print_plan(config)
        return 0
    if args.mode == "verify":
        verify(config)
        return 0
    if args.mode == "smoke":
        smoke(config)
        return 0
    if args.mode == "full":
        if not args.confirm_full:
            print("Refusing to start the full experiment without --confirm-full.")
            return 2
        runner = BenchmarkRunner(
            config, output_root=RAW_DIR,
            only_methods=only_methods or None,
            only_scenarios=only_scenarios or None,
            only_k=only_k or None,
        )
        runner.run_all(force=args.force_recompute)
        print("Full run complete. Results:", runner.results_path)
        return 0
    if args.mode == "analyze":
        from analyze_benchmark import main as analyze_main
        analyze_main()
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
