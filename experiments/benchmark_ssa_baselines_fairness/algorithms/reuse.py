"""Validation and reuse of the authoritative SSA-only results.

Only SSA-only O1/O4/O6 rows from
``experiments/exp_20261124_5000_reversed/raw_results/run_20261124_5000_reversed``
are eligible.  Nothing from the old same-direction experiments is reused.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .core import (
    final_seeds,
    metrics_from_mc,
    resolve_project_path,
    utc_now_iso,
)


def prior_run_dir(config: dict) -> Path:
    return resolve_project_path(config["ssa"]["reuse_prior_run"])


def load_prior_ssa_rows(config: dict) -> pd.DataFrame:
    run_dir = prior_run_dir(config)
    frames = []
    for scenario in config["scenarios"]:
        path = run_dir / scenario / "experiment_results.csv"
        if not path.exists():
            raise FileNotFoundError(f"Prior SSA result file not found: {path}")
        df = pd.read_csv(path)
        df = df[
            df["mode"].eq("SSA only")
            & df["objective_id"].isin(config["ssa"]["objectives"])
        ].copy()
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No prior SSA-only rows found.")
    out = pd.concat(frames, ignore_index=True)
    expected = len(config["scenarios"]) * len(config["k_values"]) * len(config["ssa"]["objectives"])
    if len(out) != expected:
        raise ValueError(f"Prior SSA rows: found {len(out)}, expected {expected}")
    return out


def _json_set(value) -> set[int]:
    if isinstance(value, str):
        return {int(x) for x in json.loads(value)}
    if isinstance(value, (list, tuple, set)):
        return {int(x) for x in value}
    raise ValueError(f"Cannot parse seed set: {value!r}")


def _json_list(value) -> list[int]:
    if isinstance(value, str):
        return [int(x) for x in json.loads(value)]
    return [int(x) for x in value]


def validate_prior_row(row, config: dict, network_meta: dict, B: list[int],
                       k: int, pool: list[int], pool_sig: str,
                       prior_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Validate one prior SSA row and its 100-row MC file.  Raises on mismatch."""
    checks = {}
    checks["scenario"] = str(row["scenario"]) in config["scenarios"]
    checks["K"] = int(row["K"]) == int(k)
    checks["mode"] = str(row["mode"]) == "SSA only"
    checks["objective"] = str(row["objective_id"]) in config["ssa"]["objectives"]
    checks["network_fingerprint"] = str(row["network_fingerprint"]) == network_meta["file_sha256"]
    checks["influence_direction"] = str(row["influence_direction"]) == "reverse"
    checks["network_nodes"] = int(row["network_nodes"]) == int(network_meta["nodes"])
    checks["network_edges"] = int(row["network_edges"]) == int(network_meta["influence_arcs"])
    checks["master_seed"] = int(row["master_seed"]) == int(config["master_seed"])
    checks["ssa_population"] = int(row["ssa_population"]) == int(config["ssa"]["population"])
    checks["ssa_iterations"] = int(row["ssa_iterations"]) == int(config["ssa"]["iterations"])
    checks["ssa_mutation_rate"] = abs(float(row["ssa_mutation_rate"]) - float(config["ssa"]["mutation_rate"])) < 1e-12
    checks["optimization_mc_runs"] = int(row["optimization_mc_runs"]) == int(config["optimization_mc_runs"])
    checks["final_mc_runs"] = int(row["final_mc_runs"]) == int(config["final_mc_runs"])
    checks["candidate_pool_signature"] = str(row["candidate_pool_signature"]) == str(pool_sig)
    checks["opponent_seed_set"] = _json_set(row["opponent_seed_set"]) == {int(x) for x in B}

    A = _json_set(row["ssa_seed_set"])
    final_A = _json_set(row["final_seed_set"])
    checks["A_size"] = len(A) == int(k)
    checks["A_unique"] = len(A) == len(set(A))
    checks["A_subset_pool"] = A.issubset({int(x) for x in pool})
    checks["A_disjoint_B"] = not (A & {int(x) for x in B})
    checks["SSA_only_same_set"] = A == final_A

    mc_rel = str(row["mc_runs_file"])
    mc_path = prior_dir / mc_rel
    checks["mc_file_exists"] = mc_path.exists()
    if not checks["mc_file_exists"]:
        failed = [k for k, v in checks.items() if not v]
        raise ValueError(f"Prior SSA row failed checks: {failed}")
    mc = pd.read_csv(mc_path)
    checks["mc_rows"] = len(mc) == int(config["final_mc_runs"])
    expected_seeds = final_seeds(config)
    checks["mc_seeds"] = list(mc["seed"].astype(int)) == expected_seeds

    metrics = metrics_from_mc(mc, int(config["final_mc_runs"]))
    for col in ("spread_A", "spread_B", "net_advantage", "blocked_B", "baseline_B"):
        checks[f"metric_{col}"] = abs(float(metrics[col]) - float(row[col])) < 1e-6

    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"Prior SSA row failed validation checks: {failed}")
    return mc, metrics


def _safe_int(value, default=np.nan):
    try:
        if value is None:
            return default
        if isinstance(value, float) and np.isnan(value):
            return default
        return int(value)
    except Exception:
        return default


def _prior_objective_evaluations(row) -> int:
    default = 132
    try:
        raw = row["ssa_info"] if "ssa_info" in row.index else None
        if isinstance(raw, str) and raw.strip():
            return int(json.loads(raw).get("objective_evaluations", default))
    except Exception:
        pass
    return default


def build_reused_row(row, mc: pd.DataFrame, metrics: dict, config: dict,
                     network_meta: dict, method_id: str, spec: dict,
                     scenario: str, k: int, B: list[int], pool: list[int],
                     pool_sig: str, signature_hash: str,
                     prior_dir: Path) -> dict:
    return {
        "experiment_id": config["experiment_id"],
        "experiment_version": config["version"],
        "method_id": method_id,
        "method_name": spec["name"],
        "method_group": spec["group"],
        "method_version": "notebook_ssa_reference_v1",
        "scenario": scenario,
        "scenario_name": {
            "B1": "Random opponent", "B2": "Degree-based opponent",
            "B3": "Positive PageRank opponent", "B4": "Community-concentrated opponent",
        }[scenario],
        "K": int(k),
        "opponent_seed_set": json.dumps(sorted(int(x) for x in B), separators=(",", ":")),
        "seed_set": json.dumps(sorted(int(x) for x in _json_set(row["ssa_seed_set"])), separators=(",", ":")),
        "seed_set_ordered": json.dumps(_json_list(row["ssa_seed_set"]), separators=(",", ":")),
        "candidate_pool_size": int(len(pool)),
        "candidate_pool_signature": str(pool_sig),
        "selection_runtime_sec": float(row["ssa_runtime_sec"]),
        "final_evaluation_runtime_sec": float(row["final_evaluation_runtime_sec"]),
        "total_runtime_sec": float(row["ssa_runtime_sec"]) + float(row["final_evaluation_runtime_sec"]),
        "preprocessing_runtime_sec": 0.0,
        "optimization_objective_evaluations": _prior_objective_evaluations(row),
        "optimization_mc_runs": int(config["optimization_mc_runs"]),
        "optimization_simulations": _safe_int(
            row.get("simulations_for_config") if "simulations_for_config" in row.index else None
        ),
        "optimization_cache_hits": _safe_int(
            row.get("cache_hits_for_config") if "cache_hits_for_config" in row.index else None
        ),
        "optimization_cache_misses": _safe_int(
            row.get("cache_misses_for_config") if "cache_misses_for_config" in row.index else None
        ),
        "final_mc_runs": int(config["final_mc_runs"]),
        "final_evaluation_simulations": np.nan,
        "method_specific_budget": json.dumps({
            "population": int(config["ssa"]["population"]),
            "iterations": int(config["ssa"]["iterations"]),
            "mutation_rate": float(config["ssa"]["mutation_rate"]),
        }, sort_keys=True),
        "optimization_budget_note": "12 initial + 10 iterations x 12 population = 132 objective evaluations; 15 MC seeds per evaluation",
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
        "network_file_sha256": network_meta["file_sha256"],
        "influence_semantic_sha256": network_meta["influence_semantic_sha256"],
        "influence_direction": network_meta["influence_direction"],
        "config_signature_hash": signature_hash,
        "source": "reused_prior_ssa_only",
        "source_detail": f"exp_20261124_5000_reversed/raw_results/run_20261124_5000_reversed/{scenario}/mc_runs/{Path(str(row['mc_runs_file'])).name}",
        "prior_config_signature_hash": str(row.get("config_signature_hash", "")),
        "status": "reused",
        "timestamp": utc_now_iso(),
    }
