"""Deterministic tests for trust->influence reversal and the 5000-node network."""
from __future__ import annotations

import unittest
from pathlib import Path

import networkx as nx
import pandas as pd

from _helpers import persistent_tmp_dir

from cpp_adapter import CppSimulator
from diffusion import simulate_python
from network import (
    build_adjacency,
    direction_check,
    influence_semantic_fingerprint,
    load_influence_network,
    load_network,
    semantic_fingerprint,
    validate_network,
)
from network_generator_large import generate_large_synthetic_network
from opponents import opponent_order

EXP = Path(__file__).resolve().parents[1] / "experiments" / "exp_20261124_5000_reversed"
LARGE_NET = EXP / "network" / "synthetic_network_5000_37500.csv"


def write_trust_csv(path: Path, edges):
    rows = []
    for u, v, prob, pol, strength in edges:
        rows.append({
            "source": int(u), "target": int(v),
            "rating": int(pol) * int(strength),
            "probability": float(prob),
            "strength": int(strength),
            "polarity": int(pol),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def cpp_run(G, nodes, A, B, seed=1):
    sim = CppSimulator(nodes, G, n_threads=1)
    return sim.simulate(A, B, seed)


def py_run(G, nodes, A, B, seed=1):
    idx, adj, probs, pol, strength = build_adjacency(G, nodes)
    return simulate_python(nodes, adj, probs, pol, strength, A, B, seed)


class TestTinyDirection(unittest.TestCase):
    def setUp(self):
        self.tmp = persistent_tmp_dir("direction")
        self.path = self.tmp / "trust.csv"

    def _load(self, edges, nodes):
        write_trust_csv(self.path, edges)
        return load_influence_network(self.path, validate=False)

    def test_trust_edge_reversed(self):
        df, G, nodes = self._load(
            [(0, 1, 1.0, 1, 5), (1, 2, 1.0, -1, 5)], [0, 1, 2]
        )
        self.assertTrue(G.has_edge(1, 0))
        self.assertTrue(G.has_edge(2, 1))
        self.assertFalse(G.has_edge(0, 1))
        self.assertFalse(G.has_edge(1, 2))
        self.assertEqual(G[1][0]["polarity"], 1)
        self.assertEqual(G[2][1]["polarity"], -1)
        self.assertEqual(G[2][1]["strength"], 5)

    def test_positive_propagation_reverse_only(self):
        _df, G, nodes = self._load([(0, 1, 1.0, 1, 5)], [0, 1])
        # Seed at 1 should reach 0; seed at 0 should not reach 1.
        cpp = cpp_run(G, nodes, [1], [], 1)
        py = py_run(G, nodes, [1], [], 1)
        self.assertEqual(cpp, py)
        self.assertEqual((cpp["spread_A"], cpp["spread_B"]), (2, 0))
        cpp = cpp_run(G, nodes, [0], [], 1)
        py = py_run(G, nodes, [0], [], 1)
        self.assertEqual(cpp, py)
        self.assertEqual((cpp["spread_A"], cpp["spread_B"]), (1, 0))

    def test_negative_propagation_reverse_only(self):
        _df, G, nodes = self._load([(0, 1, 1.0, -1, 5)], [0, 1])
        cpp = cpp_run(G, nodes, [1], [], 1)
        py = py_run(G, nodes, [1], [], 1)
        self.assertEqual(cpp, py)
        self.assertEqual((cpp["spread_A"], cpp["spread_B"]), (1, 1))
        cpp = cpp_run(G, nodes, [0], [], 1)
        py = py_run(G, nodes, [0], [], 1)
        self.assertEqual(cpp, py)
        self.assertEqual((cpp["spread_A"], cpp["spread_B"]), (1, 0))

    def test_multiple_edges_reverse_chain(self):
        _df, G, nodes = self._load(
            [(0, 1, 1.0, 1, 5), (1, 2, 1.0, 1, 5)], [0, 1, 2]
        )
        cpp = cpp_run(G, nodes, [2], [], 1)
        py = py_run(G, nodes, [2], [], 1)
        self.assertEqual(cpp, py)
        self.assertEqual((cpp["spread_A"], cpp["spread_B"]), (3, 0))
        cpp = cpp_run(G, nodes, [0], [], 1)
        self.assertEqual((cpp["spread_A"], cpp["spread_B"]), (1, 0))

    def test_candidate_pool_uses_influence_direction(self):
        from candidates import candidate_scores
        _df, G, nodes = self._load([(0, 1, 0.9, 1, 9)], [0, 1])
        scores = candidate_scores(G, nodes)
        self.assertAlmostEqual(scores[1], 0.9)
        self.assertAlmostEqual(scores[0], 0.0)

    def test_b2_degree_uses_influence_out_degree(self):
        _df, G, nodes = self._load(
            [(0, 1, 0.9, 1, 9), (0, 2, 0.9, 1, 9), (1, 2, 0.9, 1, 9)],
            [0, 1, 2],
        )
        order = opponent_order(G, nodes, "B2", max_k=3)
        self.assertEqual(order, [2, 1, 0])

    def test_b3_pagerank_uses_influence_direction(self):
        edges = [(0, 1, 0.9, 1, 9), (0, 2, 0.9, 1, 9), (1, 2, 0.9, 1, 9)]
        write_trust_csv(self.path, edges)
        _df, trust_G, nodes = load_network(self.path, validate=False)
        _df, G, nodes = load_influence_network(self.path, validate=False)
        order_inf = opponent_order(G, nodes, "B3", max_k=3)
        order_trust = opponent_order(trust_G, nodes, "B3", max_k=3)
        # Reversal swaps the high-PageRank endpoint.
        self.assertEqual(order_inf[0], 0)
        self.assertEqual(order_trust[0], 2)


class TestLargeNetwork(unittest.TestCase):
    def test_large_network_counts_and_direction(self):
        self.assertTrue(LARGE_NET.exists(), LARGE_NET)
        df, trust_G, nodes = load_network(LARGE_NET, validate=True, n_nodes=5000, n_edges=37500)
        _df, G, nodes = load_influence_network(
            LARGE_NET, validate=True, n_nodes=5000, n_edges=37500
        )
        self.assertEqual(len(nodes), 5000)
        self.assertEqual(len(df), 37500)
        self.assertEqual(G.number_of_edges(), 37500)
        dc = direction_check(df, G)
        self.assertTrue(dc["all_trust_edges_reversed"])
        self.assertEqual(dc["missing_reversed_edges"], 0)
        self.assertEqual(dc["extra_influence_edges"], 0)
        self.assertEqual(dc["same_direction_edges"], 0)
        self.assertEqual(dc["reciprocal_trust_pairs"], 0)

    def test_large_network_validation(self):
        df, _G, nodes = load_network(LARGE_NET, validate=True, n_nodes=5000, n_edges=37500)
        report = validate_network(df, n_nodes=5000, n_edges=37500)
        self.assertEqual(report["n_nodes"], 5000)
        self.assertEqual(report["n_directed_arcs"], 37500)
        self.assertEqual(report["n_self_loops"], 0)
        self.assertEqual(report["n_duplicate_arcs"], 0)
        self.assertEqual(report["n_positive"] + report["n_negative"], 37500)

    def test_seed42_regeneration_deterministic(self):
        a = generate_large_synthetic_network(5000, 37500, seed=42)
        b = generate_large_synthetic_network(5000, 37500, seed=42)
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
        self.assertEqual(semantic_fingerprint(a), semantic_fingerprint(b))
        self.assertEqual(influence_semantic_fingerprint(a), influence_semantic_fingerprint(b))

    def test_cpp_python_consistency_on_large_influence_graph(self):
        _df, G, nodes = load_influence_network(
            LARGE_NET, validate=True, n_nodes=5000, n_edges=37500
        )
        B = opponent_order(G, nodes, "B1")[:10]
        A = [u for u in nodes if u not in set(B)][:10]
        cpp = cpp_run(G, nodes, A, B, seed=123)
        py = py_run(G, nodes, A, B, seed=123)
        # Deterministic cases differ by RNG across languages; compare the
        # invariant identities rather than exact stochastic paths.
        self.assertEqual(cpp["spread_A"] + cpp["spread_B"] + cpp["ignorant"], 5000)
        self.assertEqual(py["spread_A"] + py["spread_B"] + py["ignorant"], 5000)
        self.assertGreaterEqual(cpp["spread_A"], len(A))
        self.assertGreaterEqual(cpp["spread_B"], len(B))


if __name__ == "__main__":
    unittest.main()
