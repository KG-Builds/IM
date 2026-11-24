"""Structural validation for the 1,000-node synthetic notebook."""
from __future__ import annotations

import unittest
from pathlib import Path

import nbformat as nbf

EXP_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK = EXP_DIR / "competitive_im_benchmark_1000_with_fibm.ipynb"


class TestNotebook1000Structure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb = nbf.read(NOTEBOOK, as_version=4)
        nbf.validate(cls.nb)
        cls.text = "\n".join(cell.source for cell in cls.nb.cells)

    def test_exists_and_valid(self):
        self.assertTrue(NOTEBOOK.exists())
        self.assertGreaterEqual(len(self.nb.cells), 40)

    def test_synthetic_1000_configuration(self):
        self.assertIn("benchmark_1000_synthetic_config.json", self.text)
        self.assertIn("synthetic_network_1000_7500.csv", self.text)
        self.assertIn("1,000 nodes", self.text)
        self.assertIn("7,500", self.text)

    def test_required_sections(self):
        required = [
            "Research overview", "Research objectives", "Experimental methodology",
            "Imports and configuration", "Network loading and verification",
            "Network characteristics", "Diffusion model", "Opponent strategies",
            "Algorithm implementations", "SSA O1", "SSA O4", "SSA O6",
            "Traditional baseline algorithms", "FIBM fairness baseline",
            "Experiment execution", "Results loading", "Results aggregation",
            "Net Advantage analysis", "Comparison across B1", "Comparison across K",
            "Win/tie/loss", "Runtime analysis", "Visualizations",
            "Findings and interpretation", "Limitations", "Conclusion",
        ]
        for phrase in required:
            self.assertIn(phrase, self.text, f"Missing section: {phrase}")

    def test_no_placeholders(self):
        self.assertNotIn("TODO", self.text)
        self.assertNotIn("Findings will be inserted", self.text)
        self.assertNotIn("Conclusion will be finalized", self.text)
        self.assertNotIn("NotImplementedError", self.text)

    def test_fibm_documented(self):
        self.assertIn("Fairness-Aware Influence Blocking Maximization", self.text)
        self.assertIn("positive-edge CIC surrogate", self.text)
        self.assertIn("rewritten into O1, O4 or O6", self.text)

    def test_safe_default_execution(self):
        self.assertIn('EXECUTION_MODE = "load_existing"', self.text)
        self.assertIn("ALLOW_MISSING_RUNS = False", self.text)
        self.assertIn("ALLOW_FORCE_RERUN = False", self.text)
        self.assertIn("SIMULATION_LAUNCHED = False", self.text)

    def test_no_swap_execution(self):
        self.assertNotIn("sampled_swap_refinement(", self.text)

    def test_code_cells_compile(self):
        code_cells = [c for c in self.nb.cells if c.cell_type == "code"]
        self.assertGreaterEqual(len(code_cells), 15)
        for i, cell in enumerate(code_cells):
            self.assertNotIn("\\n", cell.source, f"Literal escaped newline in code cell {i}")
            compile(cell.source.replace("%matplotlib inline", ""), f"<cell {i}>", "exec")


if __name__ == "__main__":
    unittest.main()
