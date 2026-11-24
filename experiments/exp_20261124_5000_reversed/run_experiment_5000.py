"""Dedicated runner for the 5,000-node, reverse-influence experiment.

Usage (from the project root):

    python experiments/exp_20261124_5000_reversed/run_experiment_5000.py --mode plan
    python experiments/exp_20261124_5000_reversed/run_experiment_5000.py --mode smoke
    python experiments/exp_20261124_5000_reversed/run_experiment_5000.py --mode full --confirm-full --run-id run_20261124_5000_reversed

The script never deletes anything. It writes only inside the experiment
directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[1]
for p in (ROOT, ROOT / "common", ROOT / "cpp_engine"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiment import ExperimentConfig, ExperimentRunner, MODE_SSA, MODE_SWAP  # noqa: E402
from utils import atomic_write_json, local_now_iso  # noqa: E402

NETWORK = EXP_DIR / "network" / "synthetic_network_5000_37500.csv"
RESULTS_ROOT = EXP_DIR / "raw_results"
LOG_DIR = EXP_DIR / "logs"
CONFIG_DIR = EXP_DIR / "config"

FULL_RUN_ID = "run_20261124_5000_reversed"
SMOKE_RUN_ID = "SMOKE_5000_reversed_20261124"


def base_config(run_id: str, run_kind: str, *, smoke: bool, threads: int,
                resume: bool, results_root: Path) -> ExperimentConfig:
    if smoke:
        scenarios = ("B1",)
        k_values = (10,)
        objectives = ("O2", "O3")
        ssa_population, ssa_iterations = 4, 2
        opt_mc, final_mc, swap_tests = 5, 10, 10
    else:
        scenarios = ("B1", "B2", "B3", "B4")
        k_values = (10, 20, 30, 40, 50)
        objectives = ("O1", "O2", "O3", "O4", "O5", "O6")
        ssa_population, ssa_iterations = 12, 10
        opt_mc, final_mc, swap_tests = 15, 100, 100
    return ExperimentConfig(
        run_id=run_id,
        run_kind=run_kind,
        scenarios=scenarios,
        k_values=k_values,
        objective_ids=objectives,
        modes=(MODE_SSA, MODE_SWAP),
        master_seed=42,
        ssa_population=ssa_population,
        ssa_iterations=ssa_iterations,
        ssa_mutation_rate=0.20,
        optimization_mc_runs=opt_mc,
        final_mc_runs=final_mc,
        pool_ratio=10,
        random_pool_fraction=0.30,
        swap_max_tests=swap_tests,
        n_threads=threads,
        network_path=NETWORK,
        results_root=results_root,
        log_dir=LOG_DIR,
        resume=resume,
        influence_direction="reverse",
        expected_nodes=5000,
        expected_edges=37500,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["plan", "smoke", "full"], default="plan")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--confirm-full", action="store_true")
    args = parser.parse_args()

    if args.mode == "plan":
        cfg = base_config(FULL_RUN_ID, "full_5000_reversed", smoke=False,
                          threads=args.threads, resume=True, results_root=RESULTS_ROOT)
        plan = {
            "network": str(NETWORK),
            "influence_direction": cfg.influence_direction,
            "expected_nodes": cfg.expected_nodes,
            "expected_edges": cfg.expected_edges,
            "scenarios": list(cfg.scenarios),
            "k_values": list(cfg.k_values),
            "objective_ids": list(cfg.objective_ids),
            "modes": list(cfg.modes),
            "expected_configurations": len(cfg.expected_keys()),
            "ssa_population": cfg.ssa_population,
            "ssa_iterations": cfg.ssa_iterations,
            "optimization_mc_runs": cfg.optimization_mc_runs,
            "final_mc_runs": cfg.final_mc_runs,
            "swap_max_tests": cfg.swap_max_tests,
            "results_root": str(cfg.results_root),
            "log_dir": str(cfg.log_dir),
        }
        print(json.dumps(plan, indent=2))
        return 0

    if args.mode == "full" and not args.confirm_full:
        raise SystemExit("Refusing full run without --confirm-full. Run smoke first.")

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode == "smoke":
        run_id = args.run_id or SMOKE_RUN_ID
        cfg = base_config(run_id, "SMOKE_TEST_5000_REVERSED", smoke=True,
                          threads=args.threads, resume=not args.no_resume,
                          results_root=EXP_DIR / "validation" / "smoke_results")
    else:
        run_id = args.run_id or FULL_RUN_ID
        cfg = base_config(run_id, "full_5000_reversed", smoke=False,
                          threads=args.threads, resume=not args.no_resume,
                          results_root=RESULTS_ROOT)

    config_payload = cfg.to_dict()
    config_payload["created_local"] = local_now_iso()
    atomic_write_json(CONFIG_DIR / f"{run_id}_config.json", config_payload)
    print(json.dumps(config_payload, indent=2))
    runner = ExperimentRunner(cfg)
    manifest = runner.run_all()
    print(json.dumps({
        "run_id": manifest["run_id"],
        "run_kind": manifest["run_kind"],
        "expected_configurations": manifest["expected_configurations"],
        "completed_configurations": manifest["completed_configurations"],
        "per_scenario": manifest["per_scenario"],
        "status": manifest["status"],
    }, indent=2))
    print("Results:", cfg.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
