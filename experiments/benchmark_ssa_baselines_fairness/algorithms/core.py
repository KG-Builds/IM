"""Shared infrastructure for the benchmark experiment.

Everything in this module is read-only with respect to prior experiments and
writes only inside ``experiments/benchmark_ssa_baselines_fairness/``.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXP_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.candidates import make_candidate_pool, pool_signature  # noqa: E402
from common.cpp_adapter import CppSimulator, cpp_available, openmp_available  # noqa: E402
from common.evaluation import MetricEvaluator, PROTOCOL_ID  # noqa: E402
from common.network import (  # noqa: E402
    direction_check,
    influence_semantic_fingerprint,
    load_influence_network,
    semantic_fingerprint,
    validate_network,
)
from common.opponents import SCENARIO_IDS, SCENARIO_NAMES, opponent_order  # noqa: E402
from common.stats import mean_ci, paired_diff  # noqa: E402
from common.utils import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    environment_info,
    sha256_file,
    stable_seed,
    utc_now_iso,
)

CONFIG_PATH = EXP_DIR / "config" / "benchmark_config.json"
RAW_DIR = EXP_DIR / "raw_results"
MC_DIR = RAW_DIR / "mc"
CHECKPOINT_DIR = RAW_DIR / "checkpoints"
FIBMM_DIAG_DIR = RAW_DIR / "fibmm_diagnostics"
PREPROCESSING_DIR = RAW_DIR / "preprocessing"
AGG_DIR = EXP_DIR / "aggregated"
FIG_DIR = EXP_DIR / "figures"
REPORT_DIR = EXP_DIR / "reports"
LOG_DIR = EXP_DIR / "logs"
SMOKE_DIR = EXP_DIR / "smoke"

RESULTS_PATH = RAW_DIR / "results.csv"
LEDGER_PATH = RAW_DIR / "ledger.csv"
MANIFEST_PATH = RAW_DIR / "manifest.json"
SEED_SETS_PATH = RAW_DIR / "seed_sets.csv"
OPPONENT_SEEDS_PATH = RAW_DIR / "opponent_seeds.csv"
PREPROCESSING_PATH = RAW_DIR / "preprocessing.json"
NETWORK_REFERENCE_PATH = EXP_DIR / "network" / "network_reference.json"
NETWORK_VALIDATION_PATH = EXP_DIR / "network" / "network_validation.json"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_project_path(relative: str | Path) -> Path:
    path = Path(relative)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def ensure_dirs() -> None:
    for path in (
        RAW_DIR, MC_DIR, CHECKPOINT_DIR, FIBMM_DIAG_DIR, PREPROCESSING_DIR,
        AGG_DIR, FIG_DIR, REPORT_DIR, LOG_DIR, SMOKE_DIR, EXP_DIR / "network",
    ):
        path.mkdir(parents=True, exist_ok=True)


def sha256_obj(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def method_spec(config: dict, method_id: str) -> dict:
    for spec in config["methods"]:
        if spec["id"] == method_id:
            return dict(spec)
    raise KeyError(f"Unknown method id {method_id!r}")


def scenario_number(scenario: str) -> int:
    return int(scenario[1:])


def optimization_seeds(config: dict, scenario: str) -> list[int]:
    n = scenario_number(scenario)
    return [
        int(config["master_seed"]) * 100000 + n * 10000 + i
        for i in range(int(config["optimization_mc_runs"]))
    ]


def final_seeds(config: dict) -> list[int]:
    start = int(config["evaluation"]["final_seed_start"])
    return [start + i for i in range(int(config["final_mc_runs"]))]


def opponent_orders(G, nodes: Iterable[int], config: dict) -> dict[str, list[int]]:
    return {
        sid: opponent_order(
            G, nodes, sid, seed=int(config["master_seed"]), max_k=max(config["k_values"])
        )
        for sid in config["scenarios"]
    }


def candidate_pool_for(G, nodes, B: Iterable[int], k: int, scenario: str,
                       config: dict) -> tuple[list[int], dict[int, float], dict]:
    pool, scores, meta = make_candidate_pool(
        G, nodes, B, k,
        seed=int(config["master_seed"]) + 600000 + scenario_number(scenario),
        pool_ratio=int(config["ssa"]["pool_ratio"]),
        random_fraction=float(config["ssa"]["random_pool_fraction"]),
    )
    return pool, scores, meta


def load_verified_network(config: dict) -> tuple[pd.DataFrame, Any, list[int], dict]:
    """Load the current 5,000-node network and prove its identity."""
    net_cfg = config["network"]
    path = resolve_project_path(net_cfg["path"])
    if not path.exists():
        raise FileNotFoundError(f"Authoritative network not found: {path}")
    actual_file_hash = sha256_file(path)
    if actual_file_hash != net_cfg["file_sha256"]:
        raise ValueError(
            "Network file hash mismatch. Refusing to run on a different graph.\n"
            f"expected={net_cfg['file_sha256']}\nactual={actual_file_hash}"
        )
    df, G, nodes = load_influence_network(
        path,
        validate=True,
        n_nodes=int(net_cfg["nodes"]),
        n_edges=int(net_cfg["trust_arcs"]),
        reverse=True,
    )
    if len(nodes) != int(net_cfg["nodes"]):
        raise ValueError(f"Expected {net_cfg['nodes']} nodes, found {len(nodes)}")
    if G.number_of_edges() != int(net_cfg["influence_arcs"]):
        raise ValueError(
            f"Expected {net_cfg['influence_arcs']} influence arcs, found {G.number_of_edges()}"
        )
    dc = direction_check(df, G)
    if not dc["all_trust_edges_reversed"]:
        raise ValueError(f"Influence direction check failed: {dc}")
    trust_sem = semantic_fingerprint(df)
    influence_sem = influence_semantic_fingerprint(df)
    if trust_sem != net_cfg["trust_semantic_sha256"]:
        raise ValueError(
            f"Trust semantic hash mismatch: {trust_sem} != {net_cfg['trust_semantic_sha256']}"
        )
    if influence_sem != net_cfg["influence_semantic_sha256"]:
        raise ValueError(
            f"Influence semantic hash mismatch: {influence_sem} != {net_cfg['influence_semantic_sha256']}"
        )
    validation = validate_network(
        df, n_nodes=int(net_cfg["nodes"]), n_edges=int(net_cfg["trust_arcs"])
    )
    meta = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "generated_utc": utc_now_iso(),
        "network_path": str(path.relative_to(PROJECT_ROOT)),
        "network_absolute_path": str(path),
        "file_sha256": actual_file_hash,
        "trust_semantic_sha256": trust_sem,
        "influence_semantic_sha256": influence_sem,
        "nodes": int(len(nodes)),
        "trust_arcs": int(len(df)),
        "influence_arcs": int(G.number_of_edges()),
        "trust_direction": "source_to_target (as recorded in the CSV)",
        "influence_direction": "reverse_of_trust (target_to_source)",
        "direction_check": dc,
        "trust_validation": validation,
        "identity_statement": (
            "This is the exact network file used by the authoritative "
            "exp_20261124_5000_reversed experiment. The file hash, trust "
            "semantic hash, influence semantic hash, node count, arc count and "
            "full reversal check all match the recorded identity."
        ),
    }
    return df, G, nodes, meta


def make_evaluator(G, nodes, network_meta: dict, config: dict) -> tuple[CppSimulator, MetricEvaluator]:
    if not cpp_available():
        raise RuntimeError("The C++ diffusion engine is not importable.")
    simulator = CppSimulator(nodes, G, n_threads=0)
    fingerprint = (
        f"{network_meta['file_sha256']}|reverse|"
        f"{network_meta['influence_semantic_sha256']}"
    )
    evaluator = MetricEvaluator(
        simulator, fingerprint, protocol=PROTOCOL_ID, n_threads=0
    )
    return simulator, evaluator


def _empty_results() -> pd.DataFrame:
    return pd.DataFrame()


def load_results() -> pd.DataFrame:
    if RESULTS_PATH.exists():
        return pd.read_csv(RESULTS_PATH)
    return _empty_results()


def save_results(df: pd.DataFrame) -> None:
    atomic_write_csv(df, RESULTS_PATH)


def load_ledger() -> pd.DataFrame:
    if LEDGER_PATH.exists():
        return pd.read_csv(LEDGER_PATH)
    return pd.DataFrame(columns=[
        "method_id", "scenario", "K", "status", "reason", "source",
        "selection_runtime_sec", "final_mc_runs", "timestamp",
    ])


def save_ledger(df: pd.DataFrame) -> None:
    atomic_write_csv(df, LEDGER_PATH)


def append_ledger(row: dict) -> None:
    ledger = load_ledger()
    row = dict(row)
    row.setdefault("timestamp", utc_now_iso())
    if not ledger.empty:
        key = (row.get("method_id"), row.get("scenario"), int(row.get("K")))
        mask = (
            (ledger["method_id"] == key[0])
            & (ledger["scenario"] == key[1])
            & (ledger["K"].astype(int) == key[2])
        )
        ledger = ledger.loc[~mask]
    ledger = pd.concat([ledger, pd.DataFrame([row])], ignore_index=True)
    save_ledger(ledger)


def cell_signature(config: dict, method_id: str, scenario: str, k: int,
                   B: list[int], network_meta: dict) -> str:
    spec = method_spec(config, method_id)
    payload = {
        "experiment_version": config["version"],
        "method_id": method_id,
        "method_name": spec["name"],
        "scenario": scenario,
        "K": int(k),
        "opponent_seed_set": sorted(int(x) for x in B),
        "network_file_sha256": network_meta["file_sha256"],
        "influence_semantic_sha256": network_meta["influence_semantic_sha256"],
        "influence_direction": network_meta["influence_direction"],
        "optimization_mc_runs": int(config["optimization_mc_runs"]),
        "final_mc_runs": int(config["final_mc_runs"]),
        "master_seed": int(config["master_seed"]),
        "ssa": config["ssa"],
        "baselines": config["baselines"],
        "fibmm": config["fibmm"],
    }
    return sha256_obj(payload)


def mc_path(method_id: str, scenario: str, k: int) -> Path:
    return MC_DIR / f"{method_id}_{scenario}_K{int(k)}.csv"


def write_mc(method_id: str, scenario: str, k: int, df: pd.DataFrame) -> Path:
    path = mc_path(method_id, scenario, k)
    atomic_write_csv(df, path)
    return path


def read_mc(method_id: str, scenario: str, k: int) -> pd.DataFrame:
    path = mc_path(method_id, scenario, k)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def metrics_from_mc(mc: pd.DataFrame, final_mc_runs: int) -> dict:
    required = ["seed", "spread_A", "spread_B", "baseline_B", "net_advantage_run"]
    missing = [c for c in required if c not in mc.columns]
    if missing:
        raise ValueError(f"MC file missing columns: {missing}")
    if len(mc) != int(final_mc_runs):
        raise ValueError(f"MC file has {len(mc)} rows, expected {final_mc_runs}")
    seeds = mc["seed"].astype(int).to_numpy()
    expected = np.arange(int(seeds.min()), int(seeds.min()) + int(final_mc_runs))
    if not np.array_equal(np.sort(seeds), expected):
        raise ValueError("MC file does not contain a contiguous common-seed list")
    sA = mc["spread_A"].astype(float).to_numpy()
    sB = mc["spread_B"].astype(float).to_numpy()
    base = mc["baseline_B"].astype(float).to_numpy()
    net = mc["net_advantage_run"].astype(float).to_numpy()
    blocked = base - sB
    ci_net = mean_ci(net)
    ci_block = mean_ci(blocked)
    return {
        "spread_A": float(sA.mean()),
        "spread_B": float(sB.mean()),
        "net_advantage": float(net.mean()),
        "blocked_B": max(0.0, float(base.mean() - sB.mean())),
        "baseline_B": float(base.mean()),
        "spread_A_se": _se(sA),
        "spread_B_se": _se(sB),
        "net_advantage_se": float(ci_net["se"]),
        "net_advantage_ci_low": float(ci_net["ci_low"]),
        "net_advantage_ci_high": float(ci_net["ci_high"]),
        "blocked_B_se": float(ci_block["se"]),
        "blocked_B_ci_low": float(ci_block["ci_low"]),
        "blocked_B_ci_high": float(ci_block["ci_high"]),
        "net_advantage_runs": net,
        "spread_A_runs": sA,
        "spread_B_runs": sB,
        "baseline_B_runs": base,
        "blocked_B_runs": blocked,
    }


def _se(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    return float(values.std(ddof=1) / np.sqrt(values.size))


def evaluate_new_method(evaluator: MetricEvaluator, A: Iterable[int], B: Iterable[int],
                        config: dict) -> tuple[dict, pd.DataFrame]:
    seeds = final_seeds(config)
    metrics = evaluator.evaluate(A, B, seeds)
    mc = pd.DataFrame({
        "seed": list(metrics["seeds"]),
        "spread_A": metrics["spread_A_runs"],
        "spread_B": metrics["spread_B_runs"],
        "ignorant": metrics["ignorant_runs"],
        "rounds": metrics["rounds_runs"],
        "baseline_B": metrics["baseline_B_runs"],
        "blocked_B_run": metrics["blocked_B_runs"],
        "net_advantage_run": metrics["net_advantage_runs"],
    })
    summary = metrics_from_mc(mc, len(seeds))
    summary["ignorant"] = float(metrics["ignorant"])
    summary["rounds"] = float(metrics["rounds"])
    summary["from_cache"] = bool(metrics.get("from_cache", False))
    return summary, mc


def write_network_reference(meta: dict) -> None:
    atomic_write_json(NETWORK_REFERENCE_PATH, meta)
    atomic_write_json(NETWORK_VALIDATION_PATH, {
        "generated_utc": meta["generated_utc"],
        "network_path": meta["network_path"],
        "file_sha256": meta["file_sha256"],
        "trust_semantic_sha256": meta["trust_semantic_sha256"],
        "influence_semantic_sha256": meta["influence_semantic_sha256"],
        "nodes": meta["nodes"],
        "trust_arcs": meta["trust_arcs"],
        "influence_arcs": meta["influence_arcs"],
        "direction_check": meta["direction_check"],
        "validation": meta["trust_validation"],
    })


def environment_block() -> dict:
    return {
        "environment": environment_info(),
        "cpp_available": bool(cpp_available()),
        "openmp_available": bool(openmp_available()),
    }


def write_preprocessing(payload: dict) -> None:
    existing = {}
    if PREPROCESSING_PATH.exists():
        try:
            existing = json.loads(PREPROCESSING_PATH.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    existing.update(payload)
    existing["updated_utc"] = utc_now_iso()
    atomic_write_json(PREPROCESSING_PATH, existing)
