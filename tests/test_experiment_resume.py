"""Experiment checkpoint/resume, manifest and result-validation tests."""
from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from _helpers import ROOT, persistent_tmp_dir

from experiment import (
    CheckpointMismatch,
    ExperimentConfig,
    ExperimentRunner,
    MODE_SSA,
    MODE_SWAP,
    RESULT_COLUMNS,
    derive_seeds,
    load_run_results,
    validate_results,
)


def tiny_config(results_root: Path, run_id: str, objective_ids=("O2",), **overrides):
    params = dict(
        run_id=run_id,
        run_kind="unit_test",
        scenarios=("B1",),
        k_values=(10,),
        objective_ids=objective_ids,
        modes=(MODE_SSA, MODE_SWAP),
        master_seed=42,
        ssa_population=3,
        ssa_iterations=1,
        optimization_mc_runs=3,
        final_mc_runs=5,
        swap_max_tests=3,
        n_threads=2,
        results_root=results_root,
        log_dir=Path(results_root).parent / "logs",
        resume=True,
    )
    params.update(overrides)
    return ExperimentConfig(**params)


class TestExperimentRunner(unittest.TestCase):
    def setUp(self):
        self.tmp = persistent_tmp_dir("exp_resume")
        self.root = self.tmp / "results"
        self.config = tiny_config(self.root, "unit_run_1")

    def test_full_tiny_run_and_resume(self):
        runner = ExperimentRunner(self.config)
        manifest = runner.run_all()
        self.assertEqual(manifest["expected_configurations"], 2)
        self.assertEqual(manifest["completed_configurations"], 2)
        self.assertEqual(manifest["status"], "completed")

        df = load_run_results(self.config.run_dir)
        self.assertEqual(len(df), 2)
        for col in RESULT_COLUMNS:
            self.assertIn(col, df.columns)
        self.assertEqual(set(df["mode"]), {MODE_SSA, MODE_SWAP})
        self.assertEqual(df["objective_id"].nunique(), 1)
        self.assertTrue((df["status"] == "completed").all())
        core = ["scenario", "K", "objective_id", "mode", "spread_A", "spread_B",
                "net_advantage", "blocked_B", "final_seed_set", "opponent_seed_set"]
        self.assertFalse(df[core].isna().all(axis=0).any())

        values_before = df[["mode", "spread_A", "spread_B", "net_advantage", "final_seed_set"]].copy()
        runner2 = ExperimentRunner(self.config)
        manifest2 = runner2.run_all()
        self.assertEqual(manifest2["completed_configurations"], 2)
        df2 = load_run_results(self.config.run_dir)
        self.assertEqual(len(df2), 2)
        values_after = df2[["mode", "spread_A", "spread_B", "net_advantage", "final_seed_set"]].copy()
        pd.testing.assert_frame_equal(
            values_before.sort_values("mode").reset_index(drop=True),
            values_after.sort_values("mode").reset_index(drop=True),
        )

        scenario_dir = self.config.scenario_dir("B1")
        self.assertTrue((scenario_dir / "checkpoints" / "B1_K10_O2.json").exists())
        self.assertTrue((scenario_dir / "mc_runs" / "B1_K10_O2_ssa_only.csv").exists())
        self.assertTrue((scenario_dir / "mc_runs" / "B1_K10_O2_ssa_swap.csv").exists())
        self.assertEqual(list(scenario_dir.glob(".*.tmp")), [])

    def test_missing_mode_uses_saved_ssa_checkpoint(self):
        ExperimentRunner(self.config).run_all()
        path = self.config.scenario_dir("B1") / "experiment_results.csv"
        df = pd.read_csv(path)
        swap_row = df[df["mode"] == MODE_SWAP].copy()
        self.assertEqual(len(swap_row), 1)
        df = df[df["mode"] != MODE_SWAP].reset_index(drop=True)
        df.to_csv(path, index=False)
        self.assertEqual(len(pd.read_csv(path)), 1)

        runner = ExperimentRunner(self.config)
        runner.run_all()
        df2 = pd.read_csv(path)
        self.assertEqual(len(df2), 2)
        swap_row2 = df2[df2["mode"] == MODE_SWAP].iloc[0]
        self.assertTrue(bool(swap_row2["ssa_from_checkpoint"]))

    def test_config_change_is_rejected(self):
        ExperimentRunner(self.config).run_all()
        changed = tiny_config(self.root, "unit_run_1", final_mc_runs=6)
        with self.assertRaises(CheckpointMismatch):
            ExperimentRunner(changed).run_all()

    def test_manifest_reports_missing_objectives(self):
        ExperimentRunner(self.config).run_all()
        wider = tiny_config(self.root, "unit_run_1", objective_ids=("O2", "O3"))
        runner = ExperimentRunner(wider)
        manifest = runner.build_manifest()
        self.assertEqual(manifest["expected_configurations"], 4)
        self.assertEqual(manifest["completed_configurations"], 2)
        self.assertEqual(len(manifest["missing"]), 2)
        self.assertEqual({m["objective_id"] for m in manifest["missing"]}, {"O3"})

    def test_validate_results_detects_corruption(self):
        ExperimentRunner(self.config).run_all()
        df = load_run_results(self.config.run_dir)
        report = validate_results(df, self.config)
        self.assertTrue(report["ok"], report)

        dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        self.assertFalse(validate_results(dup)["ok"])

        bad = df.copy()
        bad.loc[0, "net_advantage"] = bad.loc[0, "net_advantage"] + 1.0
        self.assertFalse(validate_results(bad)["ok"])

        negative = df.copy()
        negative.loc[0, "blocked_B"] = -1.0
        self.assertFalse(validate_results(negative)["ok"])

    def test_seed_derivation(self):
        seeds = derive_seeds(self.config, "B1", 10, "O2")
        self.assertEqual(seeds["optimization_seeds"], [42 * 100000 + 10000 + i for i in range(3)])
        self.assertEqual(seeds["final_seeds"], [900000000 + i for i in range(5)])
        self.assertEqual(seeds["ssa_random_seed"], 42 + 500000 + 1000 + 100 + 1)
        self.assertEqual(seeds["swap_random_seed"], 770000 + 1000 + 10 + 1)

    def test_full_plan_count_is_240(self):
        config = ExperimentConfig(
            run_id="plan_check",
            scenarios=("B1", "B2", "B3", "B4"),
            k_values=(10, 20, 30, 40, 50),
            objective_ids=("O1", "O2", "O3", "O4", "O5", "O6"),
            modes=(MODE_SSA, MODE_SWAP),
        )
        self.assertEqual(len(config.expected_keys()), 240)

    def test_run_id_validation(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(run_id="../escape")
        with self.assertRaises(ValueError):
            ExperimentConfig(run_id="a/b")


if __name__ == "__main__":
    unittest.main()
