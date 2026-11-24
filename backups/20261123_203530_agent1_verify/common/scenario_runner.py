"""Deprecated compatibility wrapper around the new experiment runner.

The old ``run_scenario(scenario_id, results_dir, network_path, engine_simulator)``
signature is kept so existing scripts do not fail on import, but the new runner
uses a persistent C++ engine and a run-id directory layout.  New code should use
``common.experiment.ExperimentRunner`` or ``common/run_experiment.py``.
"""
from __future__ import annotations

import time
from pathlib import Path

try:  # package import
    from .experiment import ExperimentConfig, ExperimentRunner, MODE_SSA, MODE_SWAP
except ImportError:  # top-level import
    from experiment import ExperimentConfig, ExperimentRunner, MODE_SSA, MODE_SWAP  # type: ignore


def run_scenario(scenario_id: str, results_dir, network_path, engine_simulator=None,
                 run_swap: bool = True):
    """Run one scenario with the new checkpointed runner.

    ``results_dir`` is treated as the intended scenario folder; a unique run
    directory is created next to it.  ``engine_simulator`` is ignored (the
    persistent C++ engine is used) and kept only for signature compatibility.
    """
    results_dir = Path(results_dir)
    run_id = f"{results_dir.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    config = ExperimentConfig(
        run_id=run_id,
        run_kind="legacy",
        scenarios=(scenario_id,),
        modes=(MODE_SSA, MODE_SWAP) if run_swap else (MODE_SSA,),
        network_path=Path(network_path),
        results_root=results_dir.parent,
    )
    runner = ExperimentRunner(config)
    runner.run_all()
    import pandas as pd

    return pd.read_csv(config.scenario_dir(scenario_id) / "experiment_results.csv")
