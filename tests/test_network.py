"""Network generation, validation, fingerprint and metadata tests."""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import pandas as pd

from _helpers import ROOT, persistent_tmp_dir

from network import (
    NETWORK_CSV,
    build_metadata_v2,
    load_network,
    regenerate_network_df,
    semantic_fingerprint,
    validate_network,
    verify_fingerprint,
    write_metadata_v2,
)


class TestSharedNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.G, cls.nodes = load_network(NETWORK_CSV)
        cls.report = validate_network(cls.df)

    def test_node_and_edge_counts(self):
        self.assertEqual(len(self.nodes), 1000)
        self.assertEqual(self.nodes, list(range(1000)))
        self.assertEqual(len(self.df), 7500)
        self.assertEqual(self.G.number_of_edges(), 7500)

    def test_unique_directed_arcs_and_no_self_loops(self):
        pairs = set(zip(self.df["source"].astype(int), self.df["target"].astype(int)))
        self.assertEqual(len(pairs), 7500)
        self.assertEqual(int((self.df["source"] == self.df["target"]).sum()), 0)

    def test_sign_and_probability_mapping(self):
        strength = self.df["strength"].astype(int)
        polarity = self.df["polarity"].astype(int)
        rating = self.df["rating"].astype(int)
        prob = self.df["probability"].astype(float)
        self.assertTrue(((polarity == 1) | (polarity == -1)).all())
        self.assertTrue((rating == polarity * strength).all())
        expected = strength.map(lambda s: 0.1 if s <= 3 else (0.5 if s <= 7 else 0.9))
        self.assertTrue((prob - expected).abs().max() < 1e-12)
        self.assertEqual(int((polarity > 0).sum()), 3720)
        self.assertEqual(int((polarity < 0).sum()), 3780)

    def test_degree_and_connectivity_summary(self):
        self.assertEqual(self.report["n_isolated"], 0)
        self.assertEqual(self.report["n_weak_components"], 1)
        self.assertEqual(self.report["n_directed_arcs"], 7500)
        self.assertGreater(self.report["out_degree"]["max"], 1)
        self.assertEqual(self.report["n_self_loops"], 0)

    def test_semantic_fingerprint_is_order_independent(self):
        shuffled = self.df.sample(frac=1.0, random_state=7).reset_index(drop=True)
        self.assertEqual(semantic_fingerprint(self.df), semantic_fingerprint(shuffled))

    def test_seed_42_regeneration_is_byte_identical(self):
        regenerated = regenerate_network_df()
        pd.testing.assert_frame_equal(
            regenerated.reset_index(drop=True),
            self.df.reset_index(drop=True),
            check_dtype=True,
        )
        tmp = persistent_tmp_dir("network_regen")
        path = tmp / "regenerated.csv"
        regenerated.to_csv(path, index=False)
        original = hashlib.sha256(NETWORK_CSV.read_bytes()).hexdigest()
        new = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(original, new)

    def test_metadata_v2_and_fingerprint(self):
        meta = build_metadata_v2(self.df, NETWORK_CSV)
        self.assertEqual(meta["schema_version"], 2)
        self.assertEqual(meta["validation"]["n_nodes"], 1000)
        self.assertEqual(meta["validation"]["n_directed_arcs"], 7500)
        self.assertEqual(meta["generation"]["seed"], 42)
        self.assertEqual(meta["file_sha256"], hashlib.sha256(NETWORK_CSV.read_bytes()).hexdigest())
        tmp = persistent_tmp_dir("network_meta")
        out = tmp / "meta_v2.json"
        write_metadata_v2(self.df, NETWORK_CSV, out)
        check = verify_fingerprint(NETWORK_CSV, out)
        self.assertTrue(check["ok"], check)

    def test_validation_rejects_bad_rows(self):
        bad = self.df.copy()
        bad.loc[0, "probability"] = 0.7
        with self.assertRaises(ValueError):
            validate_network(bad)
        bad = self.df.copy()
        bad.loc[0, "strength"] = 11
        bad.loc[0, "rating"] = bad.loc[0, "polarity"] * 11
        with self.assertRaises(ValueError):
            validate_network(bad)
        bad = self.df.copy()
        bad.loc[0, "target"] = bad.loc[0, "source"]
        with self.assertRaises(ValueError):
            validate_network(bad)
        bad = pd.concat([self.df, self.df.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            validate_network(bad)

    def test_metadata_v2_file_is_written_in_project(self):
        # The real v2 metadata is generated as part of the project setup.
        from network import NETWORK_METADATA_V2

        if NETWORK_METADATA_V2.exists():
            meta = json.loads(NETWORK_METADATA_V2.read_text(encoding="utf-8"))
            self.assertEqual(meta["file_sha256"], hashlib.sha256(NETWORK_CSV.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
