"""Command-line experiment runner.

Examples (from the project root)::

    python common/run_experiment.py --print-plan
    python common/run_experiment.py --smoke
    python common/run_experiment.py --run-id run_20261124_010000
    python common/run_experiment.py --scenarios B1 --k 10,20 --objectives O2

The runner is checkpointed and resumable.  It never runs more than one scenario
at a time and never launches competing experiment jobs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.experiment import (  # noqa: E402
    ExperimentConfig,
    ExperimentRunner,
    MODE_SSA,
    MODE_SWAP,
    default_run_id,
)
from common.utils import NETWORK_CSV, RESULTS_DIR  # noqa: E402


def _csv_list(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Signed competitive influence maximization experiment runner")
    parser.add_argument("--run-id", default=None, help="Unique run id (default: generated timestamp)")
    parser.add_argument("--kind", default="full", choices=["full", "SMOKE_TEST"],
                        help="Run kind label stored in results")
    parser.add_argument("--scenarios", default="B1,B2,B3,B4")
    parser.add_argument("--k", default="10,20,30,40,50", dest="k_values")
    parser.add_argument("--objectives", default="O1,O2,O3,O4,O5,O6")
    parser.add_argument("--modes", default="ssa,swap", help="ssa, swap (comma separated)")
    parser.add_argument("--threads", type=int, default=0, help="OpenMP threads; 0 = runtime default")
    parser.add_argument("--no-resume", action="store_true", help="Refuse to reuse existing rows/checkpoints")
    parser.add_argument("--results-root", default=str(RESULTS_DIR))
    parser.add_argument("--network", default=str(NETWORK_CSV))
    parser.add_argument("--smoke", action="store_true",
                        help="Small SMOKE_TEST configuration: B1, K=10, O2+O3, both modes, tiny budgets")
    parser.add_argument("--print-plan", action="store_true", help="Print the configuration and exit")
    parser.add_argument("--master-seed", type=int, default=42)
    return parser


def config_from_args(args) -> ExperimentConfig:
    if args.smoke:
        run_id = args.run_id or default_run_id("SMOKE_TEST")
        results_root = Path(args.results_root) if args.results_root != str(RESULTS_DIR) else RESULTS_DIR / "smoke"
        return ExperimentConfig(
            run_id=run_id,
            run_kind="SMOKE_TEST",
            scenarios=("B1",),
            k_values=(10,),
            objective_ids=("O2", "O3"),
            modes=(MODE_SSA, MODE_SWAP),
            master_seed=args.master_seed,
            ssa_population=4,
            ssa_iterations=2,
            optimization_mc_runs=5,
            final_mc_runs=10,
            swap_max_tests=10,
            n_threads=args.threads,
            network_path=Path(args.network),
            results_root=results_root,
            resume=not args.no_resume,
        )

    modes = []
    for token in _csv_list(args.modes):
        if token.lower() in ("ssa", "ssa_only", "ssa-only"):
            modes.append(MODE_SSA)
        elif token.lower() in ("swap", "ssa+swap", "ssa_swap"):
            modes.append(MODE_SWAP)
        else:
            raise SystemExit(f"Unknown mode token {token!r}")
    run_kind = args.kind
    run_id = args.run_id or default_run_id(run_kind)
    return ExperimentConfig(
        run_id=run_id,
        run_kind=run_kind,
        scenarios=_csv_list(args.scenarios),
        k_values=tuple(int(x) for x in _csv_list(args.k_values)),
        objective_ids=_csv_list(args.objectives),
        modes=tuple(modes),
        master_seed=args.master_seed,
        n_threads=args.threads,
        network_path=Path(args.network),
        results_root=Path(args.results_root),
        resume=not args.no_resume,
    )


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    plan = {
        "run_id": config.run_id,
        "run_kind": config.run_kind,
        "run_dir": str(config.run_dir),
        "scenarios": list(config.scenarios),
        "k_values": list(config.k_values),
        "objective_ids": list(config.objective_ids),
        "modes": list(config.modes),
        "expected_configurations": len(config.expected_keys()),
        "ssa_population": config.ssa_population,
        "ssa_iterations": config.ssa_iterations,
        "optimization_mc_runs": config.optimization_mc_runs,
        "final_mc_runs": config.final_mc_runs,
        "swap_max_tests": config.swap_max_tests,
        "n_threads": config.n_threads,
    }
    if args.print_plan:
        print(json.dumps(plan, indent=2))
        return 0
    print(json.dumps(plan, indent=2))
    runner = ExperimentRunner(config)
    manifest = runner.run_all()
    print(json.dumps({k: manifest[k] for k in (
        "run_id", "run_kind", "expected_configurations", "completed_configurations",
        "per_scenario", "status"
    )}, indent=2))
    print(f"Results: {config.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
