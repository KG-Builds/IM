"""SSA invariants and Swap refinement tests."""
from __future__ import annotations

import unittest

from _helpers import ROOT  # noqa: F401  (bootstraps sys.path)

from ssa import repair_solution, run_discrete_ssa
from swap import sampled_swap_refinement

VALUES = {i: float(i) for i in range(30)}


def value_scorer(A, B, seeds, variant, k):
    return float(sum(VALUES[int(u)] for u in A))


class TestRepair(unittest.TestCase):
    def test_deduplicate_filter_and_pad(self):
        import random

        rng = random.Random(1)
        result = repair_solution([1, 1, 2, 99], list(range(6)), 4, rng)
        self.assertEqual(len(result), 4)
        self.assertEqual(len(set(result)), 4)
        self.assertTrue(set(result).issubset(set(range(6))))
        self.assertIn(1, result)
        self.assertIn(2, result)

    def test_pool_too_small_raises(self):
        import random

        with self.assertRaises(ValueError):
            repair_solution([1], [1, 2], 5, random.Random(1))


class TestSSA(unittest.TestCase):
    def _run(self, **kwargs):
        params = dict(
            pool=list(range(20)), B=[], k=5, seeds=[1, 2, 3], variant="O2",
            scorer=value_scorer, population_size=8, iterations=10,
            mutation_rate=0.2, random_seed=7,
        )
        params.update(kwargs)
        return run_discrete_ssa(**params)

    def test_invariants_and_determinism(self):
        food, score, history, info = self._run()
        self.assertEqual(len(food), 5)
        self.assertEqual(len(set(food)), 5)
        self.assertTrue(set(food).issubset(set(range(20))))
        self.assertFalse(set(food) & set())
        self.assertTrue(all(b >= a for a, b in zip(
            history["best_objective_score"], history["best_objective_score"][1:]
        )))
        self.assertEqual(info["objective_evaluations"], 8 * 11)
        food2, score2, history2, info2 = self._run()
        self.assertEqual(food, food2)
        self.assertEqual(score, score2)
        self.assertEqual(info["objective_evaluations"], info2["objective_evaluations"])

    def test_k_equals_pool_size(self):
        food, score, history, info = self._run(pool=list(range(10)), k=10,
                                               population_size=4, iterations=2)
        self.assertEqual(sorted(food), list(range(10)))

    def test_b_seeds_excluded(self):
        food, score, history, info = self._run(pool=list(range(2, 20)), B=[0, 1], k=5)
        self.assertFalse(set(food) & {0, 1})
        self.assertTrue(set(food).issubset(set(range(2, 20))))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            self._run(pool=list(range(3)), k=5)
        with self.assertRaises(ValueError):
            self._run(pool=list(range(20)), B=[0], k=5)
        with self.assertRaises(ValueError):
            self._run(population_size=1)
        with self.assertRaises(ValueError):
            self._run(iterations=0)
        with self.assertRaises(ValueError):
            self._run(mutation_rate=1.5)
        with self.assertRaises(ValueError):
            self._run(pool=[0, 0, 1, 2, 3, 4], k=2)


class TestSwap(unittest.TestCase):
    def _run(self, initial, **kwargs):
        params = dict(
            pool=list(range(20)), B=[], k=5, seeds=[1, 2, 3], variant="O2",
            scorer=value_scorer, max_tests=100, random_seed=3,
        )
        params.update(kwargs)
        return sampled_swap_refinement(initial, **params)

    def test_improving_swap_is_accepted(self):
        final, info = self._run([0, 1, 2, 3, 4])
        self.assertTrue(info["accepted"])
        self.assertEqual(info["accepted_swap"], [0, 19])
        self.assertEqual(sorted(final), [1, 2, 3, 4, 19])
        self.assertEqual(info["objective_before"], 10.0)
        self.assertEqual(info["objective_after"], 29.0)
        self.assertEqual(info["swap_tests"], 75)
        self.assertEqual(info["objective_evaluations"], 76)

    def test_elitist_no_improvement(self):
        initial = [15, 16, 17, 18, 19]
        final, info = self._run(initial)
        self.assertFalse(info["accepted"])
        self.assertIsNone(info["accepted_swap"])
        self.assertEqual(sorted(final), sorted(initial))
        self.assertEqual(info["objective_after"], info["objective_before"])

    def test_max_tests_bound(self):
        final, info = self._run([0, 1, 2, 3, 4], max_tests=10)
        self.assertLessEqual(info["swap_tests"], 10)
        self.assertEqual(info["swap_tests"], 10)
        self.assertGreaterEqual(info["objective_after"], info["objective_before"])
        self.assertEqual(len(final), 5)
        self.assertEqual(len(set(final)), 5)

    def test_b_seeds_never_selected(self):
        final, info = self._run([0, 1, 2, 3, 4], B=[18, 19])
        self.assertFalse(set(final) & {18, 19})
        self.assertNotIn(info["accepted_swap"][1] if info["accepted_swap"] else -1, {18, 19})

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            self._run([0, 1, 2, 3])
        with self.assertRaises(ValueError):
            self._run([0, 0, 1, 2, 3])
        with self.assertRaises(ValueError):
            self._run([0, 1, 2, 3, 99])
        with self.assertRaises(ValueError):
            self._run([0, 1, 2, 3, 18], B=[18])


if __name__ == "__main__":
    unittest.main()
