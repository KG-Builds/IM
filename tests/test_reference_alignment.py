"""Direct checks against the supplied reference notebooks.

Reference files:
  old work/competition_blocking_ssa_6_variants_optimized_shared_cache.ipynb
  old work/competitive_im_no_celf_csv_persist.ipynb

These tests re-implement the reference formulas/logic locally and compare exact
outputs.  They do not execute the notebooks.
"""
from __future__ import annotations

import math
import random
import unittest
from collections import defaultdict

import numpy as np

from _helpers import ROOT, make_graph

from candidates import make_candidate_pool
from diffusion import simulate_python
from experiment import ExperimentConfig, derive_seeds
from network import NETWORK_CSV, load_network
from objectives import objective_score
from opponents import opponent_order
from ssa import run_discrete_ssa
from swap import sampled_swap_refinement


# ---------------------------------------------------------------------------
# Reference implementations copied from the notebooks
# ---------------------------------------------------------------------------
def reference_candidate_details(G, nodes):
    details = {}
    for u in nodes:
        pos = 0.0
        neg = 0.0
        for v in G.successors(u):
            d = G[u][v]
            p = float(d["probability"])
            pol = int(d["polarity"])
            if pol > 0:
                pos += p
            elif pol < 0:
                neg += p
        details[u] = (pos, neg, pos - neg)
    return details


def reference_pool(G, nodes, B, k, seed, pool_ratio=10, random_fraction=0.30):
    details = reference_candidate_details(G, nodes)
    B = set(B)
    eligible = [u for u in nodes if u not in B]
    ranked = sorted(
        eligible,
        key=lambda u: (-details[u][2], -details[u][0], details[u][1], int(u)),
    )
    pool_size = int(pool_ratio * k)
    n_random = int(round(pool_size * random_fraction))
    n_ranked = pool_size - n_random
    top_ranked = ranked[:n_ranked]
    remaining = ranked[n_ranked:]
    rng = random.Random(int(seed))
    random_nodes = rng.sample(remaining, n_random) if n_random else []
    return top_ranked + random_nodes


def reference_ssa(pool, B, k, value, population_size=8, iterations=5,
                  mutation_rate=0.2, random_seed=123):
    """Reference discrete_ssa logic with a simple additive scorer."""
    rng = random.Random(random_seed)
    cache = {}

    def score(sol):
        key = tuple(sorted(set(sol)))
        if key not in cache:
            cache[key] = float(sum(value[u] for u in sol))
        return cache[key]

    population = [rng.sample(pool, k) for _ in range(population_size)]
    scores = [score(s) for s in population]
    best_i = int(np.argmax(scores))
    food, food_score = list(population[best_i]), float(scores[best_i])
    hist = [{"iteration": 0, "best_objective_score": food_score,
             "mean_objective_score": float(np.mean(scores)),
             "unique_solutions": len({tuple(sorted(s)) for s in population})}]

    def repair(nodes):
        pool_set = set(pool)
        result, seen = [], set()
        for u in nodes:
            if u in pool_set and u not in seen:
                result.append(u)
                seen.add(u)
        remaining = [u for u in pool if u not in seen]
        if len(result) < k:
            result.extend(rng.sample(remaining, k - len(result)))
        return result[:k]

    for iteration in range(1, iterations + 1):
        c1 = 2.0 * math.exp(-((4.0 * iteration / iterations) ** 2))
        exploit = 1.0 - min(c1 / 2.0, 1.0)
        new_pop = []
        leader, current = [], population[0]
        for _pos in range(k):
            if rng.random() < exploit:
                choices = [u for u in food if u not in leader]
                if choices:
                    leader.append(rng.choice(choices))
                    continue
            choices = [u for u in current if u not in leader]
            if choices and rng.random() < 0.5:
                leader.append(rng.choice(choices))
            else:
                leader.append(rng.choice([u for u in pool if u not in leader]))
        new_pop.append(repair(leader))
        for i in range(1, population_size):
            updated = list(population[i])
            previous = new_pop[i - 1]
            for pos in rng.sample(range(k), rng.randint(1, k)):
                used = set(updated[:pos] + updated[pos + 1:])
                choices = [u for u in previous if u not in used]
                if choices:
                    updated[pos] = rng.choice(choices)
            for pos in range(k):
                if rng.random() < mutation_rate:
                    used = set(updated[:pos] + updated[pos + 1:])
                    choices = [u for u in pool if u not in used]
                    if choices:
                        updated[pos] = rng.choice(choices)
            new_pop.append(repair(updated))
        new_scores = [score(s) for s in new_pop]
        ib = int(np.argmax(new_scores))
        if new_scores[ib] > food_score:
            food, food_score = list(new_pop[ib]), float(new_scores[ib])
        iw = int(np.argmin(new_scores))
        new_pop[iw] = list(food)
        population = new_pop
        hist.append({"iteration": iteration, "best_objective_score": food_score,
                     "mean_objective_score": float(np.mean(new_scores)),
                     "unique_solutions": len({tuple(sorted(s)) for s in population})})
    return food, food_score, hist


def reference_swap(initial, pool, k, value, max_tests=100, random_seed=3):
    rng = random.Random(random_seed)
    selected = list(initial)
    cache = {}

    def score(sol):
        key = tuple(sorted(set(sol)))
        if key not in cache:
            cache[key] = float(sum(value[u] for u in sol))
        return cache[key]

    current = score(selected)
    outside = [node for node in pool if node not in set(selected)]
    total = len(selected) * len(outside)
    if total <= max_tests:
        pairs = [(old, new) for old in selected for new in outside]
    else:
        pair_set = set()
        while len(pair_set) < max_tests:
            pair_set.add((rng.choice(selected), rng.choice(outside)))
        pairs = sorted(pair_set)
    best_score, best_pair = current, None
    for old, new in pairs:
        trial = [node for node in selected if node != old] + [new]
        value_ = score(trial)
        if value_ > best_score + 1e-12:
            best_score, best_pair = value_, (old, new)
    if best_pair:
        old, new = best_pair
        selected.remove(old)
        selected.append(new)
        current = best_score
    return sorted(selected), {"swap_tests": len(pairs), "accepted_swap": best_pair,
                              "objective_before": float(score(initial)),
                              "objective_after": float(current)}


def reference_diffusion(G, nodes, A, B, seed, max_rounds=100):
    """Reference simulator logic for tiny deterministic graphs."""
    nodes = tuple(sorted(nodes))
    ADJ = {u: tuple(sorted(G.successors(u))) for u in nodes}
    states = {u: "I" for u in nodes}
    for u in sorted(set(A)):
        states[u] = "A_S"
    for u in sorted(set(B)):
        states[u] = "B_S"
    contacted = {u: set() for u in nodes}
    rng = random.Random(seed)
    for round_no in range(1, max_rounds + 1):
        old = states
        new = old.copy()
        spreaders = [u for u in nodes if old[u] in ("A_S", "B_S")]
        if not spreaders:
            break
        proposals = defaultdict(list)
        stiflers = set()
        for source in spreaders:
            available = [v for v in ADJ[source] if v not in contacted[source]]
            if not available:
                stiflers.add(source)
                continue
            target = rng.choice(available)
            contacted[source].add(target)
            if old[target] != "I":
                stiflers.add(source)
                continue
            d = G[source][target]
            if rng.random() < float(d["probability"]):
                rumour = "A" if old[source] == "A_S" else "B"
                if int(d["polarity"]) < 0:
                    rumour = "B" if rumour == "A" else "A"
                proposals[target].append((rumour, int(d["strength"])))
        for target, attempts in proposals.items():
            strongest = max(s for _, s in attempts)
            contenders = {r for r, s in attempts if s == strongest}
            new[target] = rng.choice(sorted(contenders)) + "_S"
        for u in stiflers:
            if old[u] == "A_S":
                new[u] = "A_T"
            elif old[u] == "B_S":
                new[u] = "B_T"
        states = new
    return {
        "spread_A": sum(x in ("A_S", "A_T") for x in states.values()),
        "spread_B": sum(x in ("B_S", "B_T") for x in states.values()),
        "ignorant": sum(x == "I" for x in states.values()),
        "rounds": round_no,
    }


# ---------------------------------------------------------------------------
class TestReferenceCandidatePool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.G, cls.nodes = load_network(NETWORK_CSV)

    def test_full_network_pool_exact(self):
        for sid, k in (("B1", 10), ("B2", 30), ("B3", 50), ("B4", 50)):
            B = opponent_order(self.G, self.nodes, sid)[:k]
            seed = 42 + 600000 + int(sid[1:])
            expected = reference_pool(self.G, self.nodes, B, k, seed)
            actual, _, meta = make_candidate_pool(self.G, self.nodes, B, k, seed=seed)
            self.assertEqual(actual, expected, (sid, k))
            self.assertEqual(meta["n_ranked"], 10 * k - int(round(10 * k * 0.30)))
            self.assertEqual(meta["n_random"], int(round(10 * k * 0.30)))

    def test_four_level_tie_break(self):
        # Node 1: net .9, positive 1.8, negative .9 -> first
        # Node 0: net .9, positive .9, negative 0  -> second (lower id than 4)
        # Node 4: net .9, positive .9, negative 0  -> third
        # Node 2: net .5                            -> fourth
        # Node 3: net 0                             -> fifth
        G = make_graph(
            list(range(5)),
            [
                (0, 4, 0.9, 1, 9),
                (1, 4, 0.9, 1, 9),
                (1, 3, 0.9, -1, 9),
                (2, 4, 0.5, 1, 5),
                (4, 0, 0.9, 1, 9),
            ],
        )
        # Add a second positive edge for node 1 so pos=1.8, neg=.9.
        G.add_edge(1, 2, probability=0.9, polarity=1, strength=9, rating=9)
        pool, _, _ = make_candidate_pool(G, list(range(5)), [], 1, seed=1,
                                         pool_ratio=5, random_fraction=0.0)
        self.assertEqual(pool, [1, 0, 4, 2, 3])

    def test_random_part_comes_from_ranked_remainder(self):
        nodes = list(range(20))
        G = make_graph(nodes, [])  # all scores zero -> ranked by id
        pool, _, meta = make_candidate_pool(G, nodes, [], 1, seed=7,
                                            pool_ratio=10, random_fraction=0.30)
        expected = list(range(7)) + random.Random(7).sample(list(range(7, 20)), 3)
        self.assertEqual(pool, expected)
        self.assertEqual(meta["n_ranked"], 7)
        self.assertEqual(meta["n_random"], 3)


class TestReferenceSeeds(unittest.TestCase):
    def test_seed_derivation(self):
        cfg = ExperimentConfig(run_id="seed_check")
        seeds = derive_seeds(cfg, "B3", 40, "O5")
        self.assertEqual(seeds["optimization_seeds"],
                         [42 * 100000 + 3 * 10000 + i for i in range(15)])
        self.assertEqual(seeds["final_seeds"], [900000000 + i for i in range(100)])
        self.assertEqual(seeds["ssa_random_seed"], 42 + 500000 + 3 * 1000 + 40 * 10 + 4)
        self.assertEqual(seeds["swap_random_seed"], 770000 + 3 * 1000 + 40 + 4)
        self.assertEqual(seeds["candidate_pool_seed"], 42 + 600000 + 3)


class TestReferenceObjectives(unittest.TestCase):
    def test_all_six_formulas(self):
        m = {"spread_A": 100.0, "spread_B": 40.0, "net_advantage": 60.0,
             "blocked_B": 25.0}
        self.assertAlmostEqual(objective_score(m, "O1", 10), 137.5)
        self.assertAlmostEqual(objective_score(m, "O2", 10), 60.0)
        self.assertAlmostEqual(objective_score(m, "O3", 10), 25.0)
        self.assertAlmostEqual(objective_score(m, "O4", 10), 150.0)
        self.assertAlmostEqual(objective_score(m, "O5", 10), 110.0)
        self.assertAlmostEqual(objective_score(m, "O6", 10), 131.25)
        self.assertAlmostEqual(objective_score(m, "O6", 30), 143.75)
        self.assertAlmostEqual(objective_score(m, "O6", 50), 162.5)


class TestReferenceSSA(unittest.TestCase):
    def test_exact_ssa_trajectory(self):
        pool = list(range(30))
        B = []
        k = 5
        value = {u: float(u) for u in range(30)}
        expected_food, expected_score, expected_hist = reference_ssa(
            pool, B, k, value, population_size=8, iterations=5,
            mutation_rate=0.2, random_seed=123,
        )
        food, score, hist, info = run_discrete_ssa(
            pool, B, k, [1, 2, 3], "O2",
            lambda sol, Bv, sv, v, kk: float(sum(value[u] for u in sol)),
            population_size=8, iterations=5, mutation_rate=0.2, random_seed=123,
        )
        self.assertEqual(food, expected_food)
        self.assertAlmostEqual(score, expected_score)
        for got, exp in zip(hist.to_dict("records"), expected_hist):
            self.assertAlmostEqual(got["best_objective_score"], exp["best_objective_score"])
            self.assertAlmostEqual(got["mean_objective_score"], exp["mean_objective_score"])
            self.assertEqual(got["unique_solutions"], exp["unique_solutions"])


class TestReferenceSwap(unittest.TestCase):
    def test_exact_swap_result(self):
        pool = list(range(20))
        k = 5
        initial = [0, 1, 2, 3, 4]
        value = {u: float(u) for u in range(20)}
        expected_final, expected_info = reference_swap(
            initial, pool, k, value, max_tests=100, random_seed=3
        )
        final, info = sampled_swap_refinement(
            initial, pool, [], k, [1, 2, 3], "O2",
            lambda sol, Bv, sv, v, kk: float(sum(value[u] for u in sol)),
            max_tests=100, random_seed=3,
        )
        self.assertEqual(final, expected_final)
        self.assertEqual(info["swap_tests"], expected_info["swap_tests"])
        self.assertEqual(info["accepted_swap"], list(expected_info["accepted_swap"]))
        self.assertAlmostEqual(info["objective_before"], expected_info["objective_before"])
        self.assertAlmostEqual(info["objective_after"], expected_info["objective_after"])


class TestReferenceRounds(unittest.TestCase):
    def test_rounds_match_reference(self):
        cases = [
            (make_graph(list(range(5)), [(i, i + 1, 1.0, 1, 5) for i in range(4)]),
             list(range(5)), [0], []),
            (make_graph([0, 1], []), [0, 1], [0], [1]),
            (make_graph(list(range(4)), [(i, i + 1, 1.0, 1, 5) for i in range(3)]),
             list(range(4)), [0, 1, 2, 3], []),
            (make_graph([0, 1], [(0, 1, 1.0, 1, 5)]), [0, 1], [0], []),
        ]
        for G, nodes, A, B in cases:
            expected = reference_diffusion(G, nodes, A, B, seed=1)
            from network import build_adjacency
            idx, adj, probs, pol, strength = build_adjacency(G, nodes)
            got = simulate_python(nodes, adj, probs, pol, strength, A, B, 1)
            self.assertEqual(got, expected, (G.edges, A, B))


if __name__ == "__main__":
    unittest.main()
