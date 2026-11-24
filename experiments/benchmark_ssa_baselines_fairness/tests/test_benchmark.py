"""Unit and integration tests for the benchmark experiment.

Run from the experiment directory:

    python -m unittest discover -s tests -p 'test_*.py' -v
"""
from __future__ import annotations

import json
import random
import sys
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXP_DIR.parents[1]
for path in (EXP_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from algorithms.core import (  # noqa: E402
    candidate_pool_for,
    cell_signature,
    load_config,
    load_verified_network,
    metrics_from_mc,
    opponent_orders,
    optimization_seeds,
)
from algorithms.fibmm import (  # noqa: E402
    FIBMMConfig,
    detect_communities,
    select_fibmm,
    two_hop_lower_bound,
)
from algorithms.notebook_baselines import (  # noqa: E402
    celf_lazy_greedy,
    compute_ncge_centrality_scores,
    compute_signed_pagerank,
    crieg_seeds,
    ncge_candidate_pool,
    random_seeds,
    signed_ga_seeds,
    signed_pagerank_seeds,
    top_degree_seeds,
)
from algorithms.reuse import load_prior_ssa_rows, prior_run_dir, validate_prior_row  # noqa: E402


def tiny_graph(n: int = 40, seed: int = 7) -> nx.DiGraph:
    rng = random.Random(seed)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for u in range(n):
        targets = rng.sample([v for v in range(n) if v != u], 4)
        for v in targets:
            polarity = 1 if rng.random() < 0.7 else -1
            strength = rng.choice([2, 5, 9])
            probability = {2: 0.1, 5: 0.5, 9: 0.9}[strength]
            G.add_edge(
                u, v, probability=probability, polarity=polarity,
                strength=strength, rating=polarity * strength,
            )
    return G


class TestNetworkIdentity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.df, cls.G, cls.nodes, cls.meta = load_verified_network(cls.config)

    def test_counts_and_direction(self):
        self.assertEqual(len(self.nodes), 5000)
        self.assertEqual(self.G.number_of_edges(), 37500)
        self.assertEqual(self.meta["direction_check"]["missing_reversed_edges"], 0)
        self.assertEqual(self.meta["direction_check"]["extra_influence_edges"], 0)
        self.assertEqual(self.meta["direction_check"]["same_direction_edges"], 0)
        self.assertTrue(self.meta["direction_check"]["all_trust_edges_reversed"])

    def test_hashes(self):
        self.assertEqual(
            self.meta["file_sha256"],
            "d001616cb88d73311874367774cafccc635e14d83f7d008ffcfaabea13e7a18e",
        )
        self.assertEqual(
            self.meta["influence_semantic_sha256"],
            "b3911df08912a2356c9ebf12c549b59c49049e877fb61541ab3b9d17108184b9",
        )

    def test_opponent_orders_nested(self):
        orders = opponent_orders(self.G, self.nodes, self.config)
        for sid, order in orders.items():
            self.assertEqual(len(order), 50)
            self.assertEqual(len(set(order)), 50)
            for k in self.config["k_values"]:
                B = order[:k]
                self.assertEqual(len(B), k)
                self.assertEqual(len(set(B)), k)

    def test_candidate_pool(self):
        B = opponent_orders(self.G, self.nodes, self.config)["B1"][:10]
        pool, _scores, meta = candidate_pool_for(
            self.G, self.nodes, B, 10, "B1", self.config
        )
        self.assertEqual(len(pool), 100)
        self.assertEqual(len(set(pool)), 100)
        self.assertFalse(set(pool) & set(B))
        self.assertEqual(meta["n_ranked"], 70)
        self.assertEqual(meta["n_random"], 30)


class TestPriorSSAReuse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.df, cls.G, cls.nodes, cls.meta = load_verified_network(cls.config)
        cls.prior = load_prior_ssa_rows(cls.config)
        cls.orders = opponent_orders(cls.G, cls.nodes, cls.config)

    def test_prior_rows_complete(self):
        self.assertEqual(len(self.prior), 60)
        self.assertEqual(set(self.prior["objective_id"]), {"O1", "O4", "O6"})
        self.assertEqual(set(self.prior["mode"]), {"SSA only"})

    def test_one_row_validates(self):
        sid, k, objective = "B1", 10, "O1"
        B = self.orders[sid][:k]
        pool, _scores, _meta = candidate_pool_for(
            self.G, self.nodes, B, k, sid, self.config
        )
        from common.candidates import pool_signature
        row = self.prior[
            (self.prior["scenario"] == sid)
            & (self.prior["K"].astype(int) == k)
            & (self.prior["objective_id"] == objective)
        ].iloc[0]
        mc, metrics = validate_prior_row(
            row, self.config, self.meta, B, k, pool, pool_signature(pool),
            prior_run_dir(self.config),
        )
        self.assertEqual(len(mc), 100)
        self.assertAlmostEqual(metrics["net_advantage"], float(row["net_advantage"]), places=6)


class TestSmallAlgorithms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.G = tiny_graph()
        cls.nodes = list(range(40))
        cls.B = [0, 1, 2, 3, 4]
        cls.k = 8

    def _check(self, A):
        self.assertEqual(len(A), self.k)
        self.assertEqual(len(set(A)), self.k)
        self.assertFalse(set(A) & set(self.B))
        self.assertTrue(set(A).issubset(self.nodes))

    def test_random(self):
        A = random_seeds(self.nodes, self.B, self.k, 1, 42)
        self._check(A)
        self.assertEqual(A, random_seeds(self.nodes, self.B, self.k, 1, 42))

    def test_top_degree(self):
        A = top_degree_seeds(self.G, self.nodes, self.B, self.k)
        self._check(A)

    def test_signed_pagerank(self):
        scores = compute_signed_pagerank(self.G, self.nodes)
        A = signed_pagerank_seeds(self.nodes, self.B, self.k, scores)
        self._check(A)
        self.assertEqual(len(scores), 40)

    def test_crieg(self):
        A1, info1 = crieg_seeds(self.G, self.nodes, self.B, self.k,
                                num_sketches=80, random_seed=11)
        A2, info2 = crieg_seeds(self.G, self.nodes, self.B, self.k,
                                num_sketches=80, random_seed=11)
        self._check(A1)
        self.assertEqual(A1, A2)
        self.assertEqual(info1["num_sketches"], 80)

    def test_ncge_pool(self):
        scores = compute_ncge_centrality_scores(self.G, self.nodes, k_approx=10, seed=42)
        pool = ncge_candidate_pool(self.nodes, self.B, 20, scores)
        self.assertEqual(len(pool), 20)
        self.assertFalse(set(pool) & set(self.B))

    def test_signed_ga(self):
        A1, _ = signed_ga_seeds(self.G, self.nodes, self.B, self.k, steps=3)
        A2, _ = signed_ga_seeds(self.G, self.nodes, self.B, self.k, steps=3)
        self._check(A1)
        self.assertEqual(A1, A2)

    def test_celf_matches_greedy_for_additive_scorer(self):
        pool = [u for u in self.nodes if u not in self.B][:25]
        A, history, info = celf_lazy_greedy(
            pool, self.B, self.k, lambda S: float(sum(S)), "test"
        )
        expected = sorted(pool, key=lambda u: (-u, u))[:self.k]
        self.assertEqual(sorted(A), sorted(expected))
        self.assertEqual(len(history), self.k)
        self.assertGreaterEqual(info["objective_evaluations"], len(pool))

    def test_fibmm(self):
        communities = detect_communities(self.G, self.nodes, min_fraction=0.1, seed=42)
        lb = two_hop_lower_bound(self.G, self.nodes, self.B, self.k)
        self.assertGreaterEqual(lb["LB"], 1.0)
        cfg = FIBMMConfig(
            h=0.5, epsilon=0.1, l=1, community_min_fraction=0.1,
            community_seed=42, max_crp_samples=500,
            min_samples_per_community=5, crp_seed=123,
        )
        A1, info1 = select_fibmm(
            self.G, self.nodes, self.B, self.k, cfg, communities=communities
        )
        A2, info2 = select_fibmm(
            self.G, self.nodes, self.B, self.k, cfg, communities=communities
        )
        self._check(A1)
        self.assertEqual(A1, A2)
        self.assertGreaterEqual(info1["pi_total"], 0.0)
        self.assertLessEqual(info1["pi_total"], 1.0)
        self.assertGreaterEqual(info1["tau"], 0.0)
        self.assertLessEqual(info1["tau"], 1.0)
        self.assertAlmostEqual(
            info1["sigma"], 0.5 * info1["pi_total"] + 0.5 * info1["tau"], places=10
        )
        expected_stage1 = self.k - int(np.floor(self.k * 0.5))
        self.assertEqual(info1["stage1_count"], expected_stage1)
        self.assertEqual(info1["stage1_count"] + info1["stage2_count"], self.k)


class TestResultUtilities(unittest.TestCase):
    def test_metrics_identity(self):
        mc = pd.DataFrame({
            "seed": list(range(900000000, 900000010)),
            "spread_A": np.arange(10) + 100,
            "spread_B": np.arange(10) + 90,
            "baseline_B": np.arange(10) + 95,
            "net_advantage_run": (np.arange(10) + 100) - (np.arange(10) + 90),
        })
        metrics = metrics_from_mc(mc, 10)
        self.assertAlmostEqual(metrics["net_advantage"], 10.0)
        self.assertAlmostEqual(metrics["spread_A"], 104.5)
        self.assertAlmostEqual(metrics["spread_B"], 94.5)
        self.assertAlmostEqual(metrics["blocked_B"], 5.0)
        self.assertEqual(len(metrics["net_advantage_runs"]), 10)

    def test_signature_changes(self):
        config = load_config()
        _df, G, nodes, meta = load_verified_network(config)
        B = opponent_orders(G, nodes, config)["B1"][:10]
        s1 = cell_signature(config, "SSA_O1", "B1", 10, B, meta)
        s2 = cell_signature(config, "SSA_O4", "B1", 10, B, meta)
        s3 = cell_signature(config, "SSA_O1", "B2", 10, B, meta)
        self.assertNotEqual(s1, s2)
        self.assertNotEqual(s1, s3)

    def test_smoke_results_if_present(self):
        path = EXP_DIR / "smoke" / "raw_results" / "results.csv"
        if not path.exists():
            self.skipTest("smoke results not present")
        df = pd.read_csv(path)
        self.assertEqual(len(df), 9)
        self.assertEqual(df["method_id"].nunique(), 9)
        self.assertEqual(df.groupby(["scenario", "K"])["opponent_seed_set"].nunique().max(), 1)
        for r in df.itertuples(index=False):
            A = {int(x) for x in json.loads(r.seed_set)}
            B = {int(x) for x in json.loads(r.opponent_seed_set)}
            self.assertEqual(len(A), int(r.K))
            self.assertFalse(A & B)


if __name__ == "__main__":
    unittest.main()
