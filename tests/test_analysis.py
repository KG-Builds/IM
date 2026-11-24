"""Read-only analysis pipeline tests."""
from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from _helpers import ROOT, persistent_tmp_dir

from analysis import (
    _safe_mc_path,
    build_figures,
    build_tables,
    discover_runs,
    load_run,
    select_run,
    validate_run,
)
from experiment import ExperimentConfig, ExperimentRunner, MODE_SSA, MODE_SWAP


class TestAnalysisPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = persistent_tmp_dir("analysis")
        root = cls.tmp / "results"
        cls.config = ExperimentConfig(
            run_id="analysis_unit_run",
            run_kind="unit_test",
            scenarios=("B1",),
            k_values=(10,),
            objective_ids=("O2",),
            modes=(MODE_SSA, MODE_SWAP),
            ssa_population=3,
            ssa_iterations=1,
            optimization_mc_runs=3,
            final_mc_runs=6,
            swap_max_tests=3,
            n_threads=2,
            results_root=root,
            log_dir=cls.tmp / "logs",
            resume=True,
        )
        ExperimentRunner(cls.config).run_all()
        cls.df = load_run(cls.config.run_dir)

    def test_validation(self):
        report = validate_run(self.config.run_dir, self.df)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["n_rows"], 2)
        self.assertEqual(report["mc_files_loaded"], 2)

    def test_tables(self):
        tables = build_tables(self.df, self.config.run_dir, n_boot=50)
        for name in ("overall", "per_scenario", "per_k", "mode_comparison",
                     "objective_ranking", "runtime", "variability", "best_config"):
            self.assertIn(name, tables)
            self.assertIsInstance(tables[name], pd.DataFrame)
        self.assertFalse(tables["overall"].empty)
        self.assertFalse(tables["mode_comparison"].empty)
        self.assertEqual(len(tables["mode_comparison"]), 1)
        self.assertFalse(tables["best_config"].empty)
        self.assertAlmostEqual(
            float(tables["overall"]["net_advantage_mean"].mean()),
            float(self.df["net_advantage"].mean()),
        )

    def test_figures(self):
        tables = build_tables(self.df, self.config.run_dir, n_boot=20)
        out = self.tmp / "figures"
        paths = build_figures(self.df, tables, out)
        self.assertGreaterEqual(len(paths), 3)
        for path in paths:
            self.assertTrue(Path(path).exists())
            self.assertGreater(Path(path).stat().st_size, 1000)

    def test_discovery_and_selection(self):
        root = self.tmp / "results"
        runs = discover_runs(root, include_smoke=True)
        self.assertTrue(any(r["run_id"] == self.config.run_id for r in runs))
        selected = select_run(self.config.run_id, results_root=root, include_smoke=True)
        self.assertEqual(selected, self.config.run_dir.resolve())

    def test_corruption_detection(self):
        bad = self.df.copy()
        bad.loc[0, "net_advantage"] = bad.loc[0, "net_advantage"] + 1.0
        report = validate_run(self.config.run_dir, bad)
        self.assertFalse(report["ok"])

        dup = pd.concat([self.df, self.df.iloc[[0]]], ignore_index=True)
        self.assertFalse(validate_run(self.config.run_dir, dup)["ok"])

    def test_path_escape_rejected(self):
        with self.assertRaises(ValueError):
            _safe_mc_path(self.config.run_dir, "../outside.csv")


if __name__ == "__main__":
    unittest.main()
