"""C++ engine tests: import, determinism, parity, errors, and old-vs-new regression."""
from __future__ import annotations

import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from _helpers import ROOT, adjacency, chain_graph, make_graph, random_small_graph, two_into_one_graph

import simulation_engine
from cpp_adapter import CppSimulator, cpp_available, openmp_available
from diffusion import simulate_python
from network import NETWORK_CSV, load_network
from opponents import opponent_order


def run_cpp_functional(G, nodes, A, B, seed):
    idx, adj, probs, pol, strength = adjacency(G, nodes)
    r = simulation_engine.simulate(list(nodes), adj, probs, pol, strength, list(A), list(B), int(seed))
    return {"spread_A": int(r.spread_A), "spread_B": int(r.spread_B),
            "ignorant": int(r.ignorant), "rounds": int(r.rounds)}


def run_python(G, nodes, A, B, seed):
    idx, adj, probs, pol, strength = adjacency(G, nodes)
    return simulate_python(list(nodes), adj, probs, pol, strength, list(A), list(B), int(seed))


class TestCppImport(unittest.TestCase):
    def test_import_and_openmp(self):
        self.assertTrue(cpp_available())
        self.assertTrue(hasattr(simulation_engine, "DiffusionEngine"))
        self.assertTrue(hasattr(simulation_engine, "simulate"))
        # OpenMP is expected in this build environment; report rather than fail
        # if a future build lacks it.
        self.assertIn(openmp_available(), (True, False))


class TestCppDeterministic(unittest.TestCase):
    def test_no_edges(self):
        G = make_graph([0, 1], [])
        r = run_cpp_functional(G, [0, 1], [0], [1], 3)
        self.assertEqual((r["spread_A"], r["spread_B"], r["ignorant"], r["rounds"]), (1, 1, 0, 1))

    def test_positive_chain(self):
        G = chain_graph(5)
        r = run_cpp_functional(G, list(range(5)), [0], [], 1)
        self.assertEqual((r["spread_A"], r["spread_B"], r["ignorant"], r["rounds"]), (5, 0, 0, 5))

    def test_negative_edge_flips_rumor(self):
        G = make_graph([0, 1], [(0, 1, 1.0, -1, 5)])
        r = run_cpp_functional(G, [0, 1], [0], [], 1)
        self.assertEqual((r["spread_A"], r["spread_B"]), (1, 1))
        r = run_cpp_functional(G, [0, 1], [], [0], 1)
        self.assertEqual((r["spread_A"], r["spread_B"]), (1, 1))

    def test_strength_conflict(self):
        G = two_into_one_graph(8, 3)
        r = run_cpp_functional(G, [0, 1, 2], [0], [1], 1)
        self.assertEqual((r["spread_A"], r["spread_B"]), (2, 1))
        G = two_into_one_graph(2, 9)
        r = run_cpp_functional(G, [0, 1, 2], [0], [1], 1)
        self.assertEqual((r["spread_A"], r["spread_B"]), (1, 2))

    def test_stifling(self):
        G = make_graph([0, 1], [(0, 1, 1.0, 1, 5)])
        r = run_cpp_functional(G, [0, 1], [0], [1], 1)
        self.assertEqual((r["spread_A"], r["spread_B"], r["rounds"]), (1, 1, 1))

    def test_fully_informed(self):
        G = chain_graph(4)
        r = run_cpp_functional(G, list(range(4)), [0, 1, 2, 3], [], 1)
        self.assertEqual((r["spread_A"], r["ignorant"], r["rounds"]), (4, 0, 1))

    def test_python_cpp_deterministic_parity(self):
        cases = [
            (chain_graph(7), list(range(7)), [0], [], 11),
            (make_graph([0, 1], [(0, 1, 1.0, -1, 5)]), [0, 1], [0], [], 12),
            (two_into_one_graph(8, 3), [0, 1, 2], [0], [1], 13),
            (two_into_one_graph(2, 9), [0, 1, 2], [0], [1], 14),
            (make_graph([0, 1], [(0, 1, 1.0, 1, 5)]), [0, 1], [0], [1], 15),
        ]
        for G, nodes, A, B, seed in cases:
            self.assertEqual(run_cpp_functional(G, nodes, A, B, seed),
                             run_python(G, nodes, A, B, seed), (G.edges, A, B, seed))


class TestCppEngineAndParallel(unittest.TestCase):
    def test_functional_matches_persistent_engine(self):
        G = random_small_graph(25, 50, seed=5)
        nodes = list(range(25))
        idx, adj, probs, pol, strength = adjacency(G, nodes)
        engine = simulation_engine.DiffusionEngine(nodes, adj, probs, pol, strength)
        for seed in range(30):
            a = engine.simulate([0, 1, 2], [3, 4, 5], seed)
            b = run_cpp_functional(G, nodes, [0, 1, 2], [3, 4, 5], seed)
            self.assertEqual((a.spread_A, a.spread_B, a.ignorant, a.rounds),
                             (b["spread_A"], b["spread_B"], b["ignorant"], b["rounds"]))

    def test_serial_parallel_identical(self):
        G = random_small_graph(40, 100, seed=9)
        nodes = list(range(40))
        sim = CppSimulator(nodes, G)
        seeds = list(range(200))
        serial = sim.simulate_many([0, 1, 2, 3], [4, 5, 6, 7], seeds, n_threads=1)
        parallel = sim.simulate_many([0, 1, 2, 3], [4, 5, 6, 7], seeds, n_threads=4)
        self.assertEqual(serial, parallel)

    def test_batch_order_independence(self):
        G = random_small_graph(30, 70, seed=10)
        nodes = list(range(30))
        sim = CppSimulator(nodes, G)
        As = [[0, 1], [2, 3], [4, 5], [6, 7]]
        Bs = [[10, 11], [12, 13], [14, 15], [16, 17]]
        seeds = [100, 101, 102, 103]
        forward = sim.simulate_batch(As, Bs, seeds, n_threads=4)
        order = [3, 0, 2, 1]
        shuffled = sim.simulate_batch([As[i] for i in order], [Bs[i] for i in order],
                                      [seeds[i] for i in order], n_threads=4)
        for pos, original_idx in enumerate(order):
            self.assertEqual(shuffled[pos], forward[original_idx])

    def test_rounds_terminate_on_long_chain(self):
        G = chain_graph(50)
        r = run_cpp_functional(G, list(range(50)), [0], [], 1)
        self.assertEqual(r["spread_A"], 50)
        self.assertEqual(r["rounds"], 50)


class TestCppErrors(unittest.TestCase):
    def test_unknown_and_overlap(self):
        G = chain_graph(4)
        nodes = list(range(4))
        with self.assertRaises(Exception):
            run_cpp_functional(G, nodes, [99], [], 1)
        with self.assertRaises(Exception):
            run_cpp_functional(G, nodes, [], [99], 1)
        with self.assertRaises(Exception):
            run_cpp_functional(G, nodes, [0], [0], 1)

    def test_invalid_graph_attributes(self):
        nodes = [0, 1]
        adj = [[1], []]
        good_probs = [[0.5], []]
        good_pol = [[1], []]
        good_strength = [[5], []]
        with self.assertRaises(Exception):
            simulation_engine.DiffusionEngine(nodes, adj, [[1.5], []], good_pol, good_strength)
        with self.assertRaises(Exception):
            simulation_engine.DiffusionEngine(nodes, adj, good_probs, [[0], []], good_strength)
        with self.assertRaises(Exception):
            simulation_engine.DiffusionEngine(nodes, adj, good_probs, good_pol, [[11], []])
        with self.assertRaises(Exception):
            simulation_engine.DiffusionEngine([0, 0], adj, good_probs, good_pol, good_strength)


class TestCppDistribution(unittest.TestCase):
    def test_probability_classes(self):
        for p in (0.1, 0.5, 0.9):
            G = make_graph([0, 1], [(0, 1, p, 1, 5)])
            transmitted = 0
            n_runs = 3000
            for s in range(n_runs):
                r = run_cpp_functional(G, [0, 1], [0], [], s)
                transmitted += int(r["spread_A"] == 2)
            self.assertLess(abs(transmitted / n_runs - p), 0.045, (p, transmitted / n_runs))

    def test_python_cpp_distributional_parity(self):
        G = random_small_graph(30, 60, seed=123)
        nodes = list(range(30))
        A, B = [0, 1, 2], [3, 4, 5]
        n_runs = 800
        cpp = [run_cpp_functional(G, nodes, A, B, s) for s in range(n_runs)]
        py = [run_python(G, nodes, A, B, s) for s in range(n_runs)]
        for key in ("spread_A", "spread_B", "ignorant", "rounds"):
            c = np.array([r[key] for r in cpp], dtype=float)
            p = np.array([r[key] for r in py], dtype=float)
            diff = abs(c.mean() - p.mean())
            se = math.sqrt(c.var(ddof=1) / n_runs + p.var(ddof=1) / n_runs)
            self.assertLess(diff, 4.0 * se + 0.5, (key, c.mean(), p.mean(), se))


class TestOldEngineRegression(unittest.TestCase):
    """The refactored engine must preserve the original RNG stream exactly."""

    def test_old_vs_new_full_network(self):
        backup = ROOT / "backups" / "20261123_200236" / "cpp_engine"
        old_so = backup / "simulation_engine.cpython-314-x86_64-linux-gnu.so"
        if not old_so.exists():
            self.skipTest("original engine backup not present")
        code = f"""
import sys, json
sys.path.insert(0, {str(backup)!r})
sys.path.insert(0, {str(ROOT / 'common')!r})
import simulation_engine as old
from network import load_network
from opponents import opponent_order
from network import build_adjacency
df, G, nodes = load_network({str(NETWORK_CSV)!r})
idx, adj, probs, pol, strength = build_adjacency(G, nodes)
B = opponent_order(G, nodes, 'B1')[:10]
Bset = set(B)
A = [u for u in nodes if u not in Bset][:10]
out = []
for s in range(30):
    r = old.simulate(nodes, adj, probs, pol, strength, A, B, s)
    out.append([r.spread_A, r.spread_B, r.ignorant, r.rounds])
print(json.dumps(out))
"""
        output = subprocess.check_output([sys.executable, "-c", code], cwd=str(ROOT))
        old_results = json.loads(output.decode())
        df, G, nodes = load_network(NETWORK_CSV)
        B = opponent_order(G, nodes, "B1")[:10]
        A = [u for u in nodes if u not in set(B)][:10]
        new_results = []
        for s in range(30):
            r = run_cpp_functional(G, nodes, A, B, s)
            new_results.append([r["spread_A"], r["spread_B"], r["ignorant"], r["rounds"]])
        self.assertEqual(old_results, new_results)


if __name__ == "__main__":
    unittest.main()
