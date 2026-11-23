"""Tests for indexing, row alignment and leakage controls, with synthetic embeddings."""
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from human_alignment.navon_followup import (
    component_distances, letter_sensitivity, load_embeddings, outer_folds,
    probe, substitution_indices,
)


class FollowupTests(unittest.TestCase):
    def test_components_and_balanced_margin_identity(self):
        k = 5
        cube = np.zeros((k, k, 2 * k))
        for g in range(k):
            for l in range(k):
                cube[g, l, g] = 2 / np.sqrt(5)
                cube[g, l, k + l] = 1 / np.sqrt(5)
        values = component_distances(cube, substitution_indices(k))
        np.testing.assert_allclose(values[:, 0], 0.8)
        np.testing.assert_allclose(values[:, 1], 0.2)
        margins = [cube[g, l] @ cube[g, t] - cube[g, l] @ cube[t, l]
                   for g in range(k) for l in range(k) for t in range(k) if len({g, l, t}) == 3]
        self.assertAlmostEqual(np.mean(margins), -values[:, 2].mean())

    def test_letter_omissions_cover_all_three_roles(self):
        xyz = substitution_indices(6)
        values = (xyz == 0).any(axis=0).astype(float)[:, None]
        leave, se = letter_sensitivity(values, xyz, 6)
        self.assertEqual(leave[0, 0], 0)
        self.assertTrue(np.all(leave[1:, 0] > 0))
        self.assertGreater(se[0], 0)

    def test_outer_folds_hold_out_complete_nuisance_identities(self):
        groups = np.repeat(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 7)
        seen = np.zeros(len(groups), dtype=int)
        for _, train, test in outer_folds(groups, 0):
            self.assertFalse(set(groups[train]) & set(groups[test]))
            seen[test] += 1
        np.testing.assert_array_equal(seen, 1)

    def test_probe_recovers_separable_letters_with_held_out_nuisance(self):
        letters = list("ABCDEFGHIJ")
        rows, features = [], []
        for g, global_letter in enumerate(letters):
            for l, local_letter in enumerate(letters):
                if g == l:
                    continue
                rows.append(dict(image_id=f"{global_letter}{local_letter}", global_letter=global_letter,
                                 local_letter=local_letter, condition="test", pos_idx=0))
                features.append(np.r_[np.eye(10)[g], np.eye(10)[l]] / np.sqrt(2))
        for target in ("global", "local"):
            predictions, folds = probe(np.asarray(features), rows, target, 0)
            self.assertTrue(all(r["correct"] for r in predictions))
            self.assertEqual(len(folds), 5)

    def test_embeddings_align_by_id_and_reject_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "embeddings-dino_base.parquet"
            vectors = np.eye(2, 768)
            table = pa.table(dict(image_id=["b", "a"], family=["dino"] * 2, variant=["base"] * 2,
                                  pre_logits=vectors.tolist()))
            pq.write_table(table, path)
            aligned = load_embeddings(path, [{"image_id": "a"}, {"image_id": "b"}], "pre_logits")
            np.testing.assert_array_equal(aligned, vectors[::-1])
            table = table.set_column(0, "image_id", pa.array(["a", "a"]))
            pq.write_table(table, path)
            with self.assertRaises(ValueError):
                load_embeddings(path, [{"image_id": "a"}, {"image_id": "b"}], "pre_logits")


if __name__ == "__main__":
    unittest.main()
