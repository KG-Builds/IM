#!/usr/bin/env python3
"""Run/resume the separate 1,000-node synthetic benchmark.

This runner deliberately disables reuse of the 5,000-node SSA results; the
1,000-node graph is a different network.  Existing 1,000-node cells are still
reused by the normal checkpoint logic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_benchmark import BenchmarkRunner

EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config" / "benchmark_1000_synthetic_config.json"
RESULTS_ROOT = EXP_DIR / "benchmark_1000_synthetic" / "raw_results"


class Synthetic1000Runner(BenchmarkRunner):
    def _reuse_prior_cell(self, *args, **kwargs):
        return False


def _refresh_manifest(config: dict) -> None:
    """Keep the manifest scoped to the full 1,000-node matrix.

    A filtered run would otherwise report only the selected subset as the
    expected cell count, which makes the notebook's reuse audit misleading.
    """
    import pandas as pd

    manifest_path = RESULTS_ROOT / "manifest.json"
    results_path = RESULTS_ROOT / "results.csv"
    if not manifest_path.exists() or not results_path.exists():
        return

    results = pd.read_csv(results_path)
    present = {
        (str(row.method_id), str(row.scenario), int(row.K))
        for row in results.itertuples(index=False)
    }
    expected = [
        (str(method["id"]), str(scenario), int(k))
        for method in config["methods"]
        for scenario in config["scenarios"]
        for k in config["k_values"]
    ]
    missing = [
        {"method_id": method_id, "scenario": scenario, "K": k}
        for (method_id, scenario, k) in expected
        if (method_id, scenario, k) not in present
    ]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["expected_cells"] = len(expected)
    manifest["completed_cells"] = len(present)
    manifest["missing_cells"] = missing
    manifest["config_path"] = str(CONFIG_PATH.relative_to(EXP_DIR))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Recompute selected cells; existing results are not deleted.")
    parser.add_argument("--only-methods", default="")
    parser.add_argument("--only-scenarios", default="")
    parser.add_argument("--only-k", default="")
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    only_methods = [x.strip() for x in args.only_methods.split(",") if x.strip()]
    only_scenarios = [x.strip() for x in args.only_scenarios.split(",") if x.strip()]
    only_k = [int(x) for x in args.only_k.split(",") if x.strip()]

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    runner = Synthetic1000Runner(
        config,
        output_root=RESULTS_ROOT,
        only_methods=only_methods or None,
        only_scenarios=only_scenarios or None,
        only_k=only_k or None,
    )
    runner.run_all(force=bool(args.force))
    _refresh_manifest(config)
    print("1,000-node run complete:", runner.results_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
