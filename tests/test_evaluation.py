"""Metric evaluation, cache-key and Blocked_B protocol tests."""
from __future__ import annotations

import unittest

from _helpers import chain_graph, make_graph

from cpp_adapter import CppSimulator
from evaluation import MetricEvaluator, PROTOCOL_ID
from objectives import objective_score


class TestMetricEvaluator(unittest.TestCase):
    def setUp(self):
        # 0 -> 1 -> 2 positive chain, plus isolated node 3.
        self.G = make_graph(
            [0, 1, 2, 3],
            [(0, 1, 1.0, 1, 5), (1, 2, 1.0, 1, 5)],
        )
        self.nodes = [0, 1, 2, 3]
        self.sim = CppSimulator(self.nodes, self.G, n_threads=1)
        self.ev = MetricEvaluator(self.sim, "test-fingerprint", n_threads=1)
        self.B = [0]
        self.seeds = [11, 12, 13]

    def test_metrics_and_blocked_b(self):
        m = self.ev.evaluate([3], self.B, self.seeds)
        self.assertEqual(m["spread_A"], 1.0)
        self.assertEqual(m["spread_B"], 3.0)
        self.assertEqual(m["baseline_B"], 3.0)
        self.assertEqual(m["net_advantage"], -2.0)
        self.assertEqual(m["blocked_B"], 0.0)
        self.assertEqual(m["ignorant"], 0.0)
        self.assertEqual(m["n_runs"], 3)
        self.assertEqual(len(m["spread_A_runs"]), 3)
        self.assertEqual(len(m["baseline_B_runs"]), 3)
        self.assertTrue((m["net_advantage_runs"] == m["spread_A_runs"] - m["spread_B_runs"]).all())

    def test_blocking_seed_reduces_baseline(self):
        m = self.ev.evaluate([1], self.B, self.seeds)
        self.assertEqual(m["spread_A"], 2.0)
        self.assertEqual(m["spread_B"], 1.0)
        self.assertEqual(m["baseline_B"], 3.0)
        self.assertEqual(m["net_advantage"], 1.0)
        self.assertEqual(m["blocked_B"], 2.0)
        self.assertTrue((m["blocked_B_runs"] == 2).all())

    def test_blocked_b_formula(self):
        m = self.ev.evaluate([1], self.B, self.seeds)
        expected = max(0.0, m["baseline_B"] - m["spread_B"])
        self.assertAlmostEqual(m["blocked_B"], expected)

    def test_metric_cache_and_objective_cache_are_independent(self):
        m1 = self.ev.evaluate([1], self.B, self.seeds)
        m2 = self.ev.evaluate([1], self.B, self.seeds)
        self.assertFalse(m1["from_cache"])
        self.assertTrue(m2["from_cache"])
        self.assertEqual(self.ev.cache_hits, 1)
        # Same metrics must produce different objective values per variant.
        o1 = objective_score(m2, "O1", 10)
        o2 = objective_score(m2, "O2", 10)
        o3 = objective_score(m2, "O3", 10)
        self.assertAlmostEqual(o1, 2.0 + 1.5 * 2.0)
        self.assertAlmostEqual(o2, 1.0)
        self.assertAlmostEqual(o3, 2.0)
        self.assertNotEqual(o1, o2)
        self.assertEqual(self.ev.evaluate_objective([1], self.B, self.seeds, "O1", 10), o1)
        self.assertEqual(self.ev.evaluate_objective([1], self.B, self.seeds, "O2", 10), o2)

    def test_baseline_cache_shared_across_a_sets(self):
        self.ev.evaluate([3], self.B, self.seeds)
        self.ev.evaluate([1], self.B, self.seeds)
        self.assertEqual(len(self.ev.baseline_cache), 1)
        self.assertEqual(len(self.ev.metric_cache), 2)

    def test_protocol_id(self):
        self.assertEqual(self.ev.protocol, PROTOCOL_ID)

    def test_overlap_and_empty_seeds(self):
        with self.assertRaises(ValueError):
            self.ev.evaluate([0], [0], self.seeds)
        with self.assertRaises(ValueError):
            self.ev.evaluate([1], self.B, [])

    def test_cache_stats(self):
        self.ev.evaluate([1], self.B, self.seeds)
        stats = self.ev.cache_stats()
        self.assertGreaterEqual(stats["simulations"], 6)  # 3 A/B + 3 baseline
        self.assertEqual(stats["cache_misses"], 1)
        self.assertEqual(stats["cache_hits"], 0)


if __name__ == "__main__":
    unittest.main()
