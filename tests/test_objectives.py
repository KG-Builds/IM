"""Hand-calculated objective formula tests."""
from __future__ import annotations

import unittest

from _helpers import ROOT  # noqa: F401  (bootstraps sys.path)

from objectives import (
    VARIANTS,
    budget_coefficients,
    objective_score,
    objective_validity,
    requires_blocked_b,
    resolve_variant,
    variant_metadata,
)

METRICS = {
    "spread_A": 100.0,
    "spread_B": 40.0,
    "net_advantage": 60.0,
    "blocked_B": 25.0,
    "baseline_B": 65.0,
    "ignorant": 900.0,
    "rounds": 10.0,
}


class TestObjectives(unittest.TestCase):
    def test_hand_calculated_values(self):
        self.assertAlmostEqual(objective_score(METRICS, "O1", 10), 137.5)
        self.assertAlmostEqual(objective_score(METRICS, "O2", 10), 60.0)
        self.assertAlmostEqual(objective_score(METRICS, "O3", 10), 25.0)
        self.assertAlmostEqual(objective_score(METRICS, "O4", 10), 150.0)
        self.assertAlmostEqual(objective_score(METRICS, "O5", 10), 110.0)
        self.assertAlmostEqual(objective_score(METRICS, "O6", 10), 131.25)
        self.assertAlmostEqual(objective_score(METRICS, "O6", 20), 131.25)
        self.assertAlmostEqual(objective_score(METRICS, "O6", 30), 143.75)
        self.assertAlmostEqual(objective_score(METRICS, "O6", 40), 143.75)
        self.assertAlmostEqual(objective_score(METRICS, "O6", 50), 162.5)

    def test_o4_and_o5_are_distinct(self):
        self.assertNotAlmostEqual(objective_score(METRICS, "O4", 10),
                                  objective_score(METRICS, "O5", 10))

    def test_budget_coefficients(self):
        self.assertEqual(budget_coefficients(10), (0.75, 2.25))
        self.assertEqual(budget_coefficients(20), (0.75, 2.25))
        self.assertEqual(budget_coefficients(30), (1.00, 1.75))
        self.assertEqual(budget_coefficients(40), (1.00, 1.75))
        self.assertEqual(budget_coefficients(50), (1.25, 1.50))
        with self.assertRaises(ValueError):
            budget_coefficients(15)

    def test_names_and_ids_resolve(self):
        self.assertEqual(resolve_variant("O2")["name"], "Competition-aware")
        self.assertEqual(resolve_variant("Competition-aware")["id"], "O2")
        with self.assertRaises(ValueError):
            resolve_variant("not-a-variant")

    def test_validity_labels(self):
        for oid in ("O1", "O2", "O3", "O4", "O5", "O6"):
            self.assertEqual(objective_validity(oid), "validated")
        self.assertFalse(requires_blocked_b("O2"))
        for oid in ("O1", "O3", "O4", "O5", "O6"):
            self.assertTrue(requires_blocked_b(oid))

    def test_all_six_present(self):
        self.assertEqual(list(VARIANTS), ["O1", "O2", "O3", "O4", "O5", "O6"])
        self.assertEqual(len(variant_metadata()), 6)


if __name__ == "__main__":
    unittest.main()
