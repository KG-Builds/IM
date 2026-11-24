"""Pure-Python diffusion reference tests (deterministic cases + distributions)."""
from __future__ import annotations

import unittest

from _helpers import adjacency, chain_graph, make_graph, two_into_one_graph

from diffusion import simulate_python, simulate_python_graph


def run(G, nodes, A, B, seed=1):
    idx, adj, probs, pol, strength = adjacency(G, nodes)
    return simulate_python(nodes, adj, probs, pol, strength, A, B, seed)


class TestPythonDiffusionDeterministic(unittest.TestCase):
    def test_no_edges(self):
        G = make_graph([0, 1], [])
        r = run(G, [0, 1], [0], [1], seed=3)
        self.assertEqual((r["spread_A"], r["spread_B"], r["ignorant"], r["rounds"]), (1, 1, 0, 2))

    def test_positive_chain(self):
        G = chain_graph(5, prob=1.0, polarity=1)
        r = run(G, list(range(5)), [0], [], seed=1)
        self.assertEqual(r["spread_A"], 5)
        self.assertEqual(r["spread_B"], 0)
        self.assertEqual(r["ignorant"], 0)
        self.assertEqual(r["rounds"], 6)

    def test_negative_edge_flips_rumor(self):
        G = make_graph([0, 1], [(0, 1, 1.0, -1, 5)])
        r = run(G, [0, 1], [0], [], seed=1)
        self.assertEqual(r["spread_A"], 1)  # seed 0 only
        self.assertEqual(r["spread_B"], 1)  # node 1 received flipped rumor
        self.assertEqual(r["ignorant"], 0)

    def test_negative_edge_flips_B_to_A(self):
        G = make_graph([0, 1], [(0, 1, 1.0, -1, 5)])
        r = run(G, [0, 1], [], [0], seed=1)
        self.assertEqual(r["spread_A"], 1)
        self.assertEqual(r["spread_B"], 1)

    def test_strength_wins_conflict(self):
        G = two_into_one_graph(strength_a=8, strength_b=3)
        r = run(G, [0, 1, 2], [0], [1], seed=1)
        self.assertEqual(r["spread_A"], 2)
        self.assertEqual(r["spread_B"], 1)

    def test_stronger_B_wins_conflict(self):
        G = two_into_one_graph(strength_a=2, strength_b=9)
        r = run(G, [0, 1, 2], [0], [1], seed=1)
        self.assertEqual(r["spread_A"], 1)
        self.assertEqual(r["spread_B"], 2)

    def test_stifling_on_non_ignorant_contact(self):
        G = make_graph([0, 1], [(0, 1, 1.0, 1, 5)])
        r = run(G, [0, 1], [0], [1], seed=1)
        self.assertEqual((r["spread_A"], r["spread_B"], r["rounds"]), (1, 1, 2))

    def test_fully_informed(self):
        G = chain_graph(4, prob=1.0, polarity=1)
        r = run(G, list(range(4)), [0, 1, 2, 3], [], seed=1)
        self.assertEqual(r["spread_A"], 4)
        self.assertEqual(r["ignorant"], 0)
        self.assertEqual(r["rounds"], 2)

    def test_unknown_seed_and_overlap(self):
        G = chain_graph(3)
        with self.assertRaises(ValueError):
            run(G, [0, 1, 2], [99], [], seed=1)
        with self.assertRaises(ValueError):
            run(G, [0, 1, 2], [0], [0], seed=1)

    def test_termination_on_medium_chain(self):
        G = chain_graph(50, prob=1.0, polarity=1)
        r = run(G, list(range(50)), [0], [], seed=1)
        self.assertEqual(r["spread_A"], 50)
        self.assertEqual(r["rounds"], 51)


class TestPythonDiffusionStochastic(unittest.TestCase):
    def test_probability_classes(self):
        for p in (0.1, 0.5, 0.9):
            G = make_graph([0, 1], [(0, 1, p, 1, 5)])
            transmitted = 0
            n_runs = 3000
            for s in range(n_runs):
                r = run(G, [0, 1], [0], [], seed=s)
                transmitted += int(r["spread_A"] == 2)
            observed = transmitted / n_runs
            self.assertLess(abs(observed - p), 0.045, (p, observed))

    def test_negative_probability_class(self):
        p = 0.5
        G = make_graph([0, 1], [(0, 1, p, -1, 5)])
        transmitted = 0
        n_runs = 3000
        for s in range(n_runs):
            r = run(G, [0, 1], [0], [], seed=s)
            transmitted += int(r["spread_B"] == 1)
        self.assertLess(abs(transmitted / n_runs - p), 0.045)

    def test_conflict_tie_is_balanced(self):
        G = two_into_one_graph(strength_a=5, strength_b=5)
        a_wins = 0
        n_runs = 2000
        for s in range(n_runs):
            r = run(G, [0, 1, 2], [0], [1], seed=s)
            a_wins += int(r["spread_A"] == 2)
        self.assertLess(abs(a_wins / n_runs - 0.5), 0.05)


if __name__ == "__main__":
    unittest.main()
