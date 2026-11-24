"""Notebook structure, syntax and safety tests."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import nbformat

from _helpers import ROOT

NOTEBOOK_DIR = ROOT / "notebooks"
EXPECTED = [
    "00_Setup_Synthetic_Network.ipynb",
    "B1_Random_opponent.ipynb",
    "B2_Degree-based_opponent.ipynb",
    "B3_Positive_PageRank_opponent.ipynb",
    "B4_Community-concentrated_opponent.ipynb",
    "05_Analysis_Charts.ipynb",
]


class TestNotebooks(unittest.TestCase):
    def test_expected_notebooks_exist(self):
        for name in EXPECTED:
            self.assertTrue((NOTEBOOK_DIR / name).exists(), name)

    def test_json_and_nbformat_valid(self):
        for name in EXPECTED:
            path = NOTEBOOK_DIR / name
            nb = json.loads(path.read_text(encoding="utf-8"))
            nbformat.validate(nb)

    def test_code_cells_compile_and_have_real_line_breaks(self):
        for name in EXPECTED:
            path = NOTEBOOK_DIR / name
            nb = json.loads(path.read_text(encoding="utf-8"))
            for i, cell in enumerate(nb["cells"]):
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell["source"])
                self.assertNotIn("\\n", source, f"{name} cell {i} contains literal backslash-n")
                compile(source, f"{path}:cell{i}", "exec")

    def test_scenario_labels(self):
        for sid in ("B1", "B2", "B3", "B4"):
            path = NOTEBOOK_DIR / (f"{sid}_" + {
                "B1": "Random_opponent.ipynb",
                "B2": "Degree-based_opponent.ipynb",
                "B3": "Positive_PageRank_opponent.ipynb",
                "B4": "Community-concentrated_opponent.ipynb",
            }[sid])
            text = path.read_text(encoding="utf-8")
            self.assertIn(f"SCENARIO = '{sid}'", text)

    def test_full_run_requires_confirmation(self):
        for sid in ("B1", "B2", "B3", "B4"):
            path = next(NOTEBOOK_DIR.glob(f"{sid}_*.ipynb"))
            text = path.read_text(encoding="utf-8")
            self.assertIn("CONFIRM_FULL_RUN = False", text)
            self.assertIn("RUN_MODE = 'none'", text)

    def test_analysis_notebook_is_read_only(self):
        path = NOTEBOOK_DIR / "05_Analysis_Charts.ipynb"
        text = path.read_text(encoding="utf-8")
        self.assertIn("run_analysis", text)
        self.assertNotIn("ExperimentRunner", text)
        self.assertNotIn("run_experiment.py --scenarios", text)
        self.assertNotIn("CppSimulator", text)


if __name__ == "__main__":
    unittest.main()
