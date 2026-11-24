"""Opponent B1-B4 strategy tests."""
from __future__ import annotations

import random
import unittest

import networkx as nx

from _helpers import make_graph

from network import NETWORK_CSV, load_network
from opponents import (
    SCENARIO_IDS,
    all_opponent_orders,
    opponent_order,
    opponent_seed_set,
)


class TestOpponentsSynthetic(unittest.TestCase):
    def test_b1_is_exact_starter_sample(self):
        nodes = list(range(50))
        G = make_graph(nodes, [])
        order = opponent_order(G, nodes, "B1", seed=42, max_k=20)
        expected = random.Random(42 + 810001).sample(nodes, 20)
        self.assertEqual(order, expected)

    def test_b2_degree_then_id(self):
        G = make_graph(
            list(range(6)),
            [
                (0, 1, 0.5, 1, 5), (0, 2, 0.5, 1, 5), (0, 3, 0.5, 1, 5),
                (1, 2, 0.5, 1, 5), (1, 3, 0.5, 1, 5),
                (2, 3, 0.5, 1, 5),
                (3, 4, 0.5, 1, 5),
            ],
        )
        order = opponent_order(G, list(range(6)), "B2", max_k=6)
        self.assertEqual(order, [0, 1, 2, 3, 4, 5])

    def test_b3_uses_positive_subgraph_only(self):
        G = make_graph(
            [0, 1, 2, 3],
            [
                (0, 1, 0.9, 1, 9),
                (2, 1, 0.9, 1, 9),
                (3, 1, 0.9, 1, 9),
                (1, 0, 0.9, -1, 9),
                (2, 0, 0.9, -1, 9),
                (3, 0, 0.9, -1, 9),
            ],
        )
        order = opponent_order(G, [0, 1, 2, 3], "B3", max_k=4)
        self.assertEqual(order[0], 1)
        # Node 0 receives only negative edges and must not rank first.
        self.assertNotEqual(order[0], 0)

    def test_b4_small_community_raises(self):
        nodes = list(range(10))
        edges = []
        for group in (range(0, 5), range(5, 10)):
            group = list(group)
            for i in group:
                for j in group:
                    if i != j:
                        edges.append((i, j, 0.5, 1, 5))
        edges.append((4, 5, 0.5, 1, 5))
        G = make_graph(nodes, edges)
        with self.assertRaises(ValueError):
            opponent_order(G, nodes, "B4", max_k=10)

    def test_unknown_scenario(self):
        G = make_graph([0, 1], [])
        with self.assertRaises(ValueError):
            opponent_order(G, [0, 1], "B9", max_k=2)


class TestOpponentsFullNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.G, cls.nodes = load_network(NETWORK_CSV)
        cls.orders = all_opponent_orders(cls.G, cls.nodes, seed=42, max_k=50)

    def test_all_orders_valid(self):
        for sid in SCENARIO_IDS:
            order = self.orders[sid]
            self.assertEqual(len(order), 50)
            self.assertEqual(len(set(order)), 50)
            self.assertTrue(set(order).issubset(set(self.nodes)))

    def test_nested_across_k(self):
        for sid in SCENARIO_IDS:
            order = self.orders[sid]
            for k in (10, 20, 30, 40, 50):
                self.assertEqual(opponent_seed_set(order, k), order[:k])

    def test_b2_matches_degree_sort(self):
        expected = sorted(self.nodes, key=lambda u: (-self.G.out_degree(u), int(u)))[:50]
        self.assertEqual(self.orders["B2"], expected)

    def test_b4_is_largest_community_by_degree(self):
        und = nx.Graph()
        und.add_nodes_from(self.nodes)
        und.add_edges_from((u, v) for u, v in self.G.edges() if u != v)
        comms = list(nx.community.louvain_communities(und, seed=42, weight=None))
        largest = sorted(comms, key=lambda c: (-len(c), min(c)))[0]
        self.assertEqual(len(largest), 227)
        expected = sorted(largest, key=lambda u: (-self.G.out_degree(u), int(u)))[:50]
        self.assertEqual(self.orders["B4"], expected)

    def test_determinism(self):
        again = all_opponent_orders(self.G, self.nodes, seed=42, max_k=50)
        self.assertEqual(self.orders, again)


if __name__ == "__main__":
    unittest.main()
