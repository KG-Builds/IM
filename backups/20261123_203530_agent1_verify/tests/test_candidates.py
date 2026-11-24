"""Candidate-pool composition, polarity-awareness and edge-case tests."""
from __future__ import annotations

import unittest

from _helpers import make_graph

from candidates import candidate_scores, make_candidate_pool, pool_signature
from network import NETWORK_CSV, load_network
from opponents import opponent_order


class TestCandidateScores(unittest.TestCase):
    def test_negative_edges_reduce_score(self):
        G = make_graph(
            [0, 1, 2, 3, 4],
            [
                (0, 3, 0.9, 1, 9),
                (1, 3, 0.9, 1, 9),
                (1, 4, 0.9, -1, 9),
                (2, 4, 0.9, -1, 9),
            ],
        )
        scores = candidate_scores(G, [0, 1, 2, 3, 4])
        self.assertAlmostEqual(scores[0], 0.9)
        self.assertAlmostEqual(scores[1], 0.0)
        self.assertAlmostEqual(scores[2], -0.9)
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])

    def test_negative_only_node_is_not_rewarded(self):
        G = make_graph([0, 1], [(0, 1, 0.5, -1, 5)])
        scores = candidate_scores(G, [0, 1])
        self.assertEqual(scores[0], -0.5)
        self.assertEqual(scores[1], 0.0)


class TestCandidatePool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.G, cls.nodes = load_network(NETWORK_CSV)

    def test_pool_size_and_composition_full_network(self):
        B = opponent_order(self.G, self.nodes, "B1")[:10]
        pool, scores, meta = make_candidate_pool(self.G, self.nodes, B, 10, seed=42)
        self.assertEqual(len(pool), 100)
        self.assertEqual(meta["pool_size_target"], 100)
        self.assertEqual(meta["n_ranked"], 70)
        self.assertEqual(meta["n_random"], 30)
        self.assertFalse(set(pool) & set(B))
        self.assertEqual(len(pool), len(set(pool)))
        self.assertTrue(set(pool).issubset(set(self.nodes)))

    def test_determinism(self):
        B = opponent_order(self.G, self.nodes, "B1")[:10]
        a = make_candidate_pool(self.G, self.nodes, B, 10, seed=123)
        b = make_candidate_pool(self.G, self.nodes, B, 10, seed=123)
        c = make_candidate_pool(self.G, self.nodes, B, 10, seed=124)
        self.assertEqual(a[0], b[0])
        self.assertNotEqual(a[0], c[0])
        self.assertEqual(pool_signature(a[0]), pool_signature(b[0]))

    def test_small_network_truncation(self):
        G = make_graph(list(range(20)), [])
        pool, scores, meta = make_candidate_pool(G, list(range(20)), [], 10, seed=1)
        self.assertEqual(len(pool), 20)
        self.assertTrue(meta["truncated"])
        self.assertEqual(meta["n_random"], 0)

    def test_too_few_eligible_nodes_raises(self):
        G = make_graph(list(range(20)), [])
        B = list(range(15))
        with self.assertRaises(ValueError):
            make_candidate_pool(G, list(range(20)), B, 10, seed=1)

    def test_opponent_seeds_never_in_pool(self):
        B = opponent_order(self.G, self.nodes, "B2")[:50]
        pool, _, _ = make_candidate_pool(self.G, self.nodes, B, 50, seed=42)
        self.assertFalse(set(pool) & set(B))
        self.assertEqual(len(pool), 500)

    def test_invalid_arguments(self):
        G = make_graph(list(range(10)), [])
        with self.assertRaises(ValueError):
            make_candidate_pool(G, list(range(10)), [], 0, seed=1)
        with self.assertRaises(ValueError):
            make_candidate_pool(G, list(range(10)), [], 5, seed=1, random_fraction=1.5)


if __name__ == "__main__":
    unittest.main()
