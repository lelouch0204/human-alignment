import sys
from pathlib import Path
import unittest
import numpy as np
from itertools import combinations
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from human_alignment.navon_shapes import image_folds, pair_features, fit_metric, adapt


class ShapeTests(unittest.TestCase):
    def test_disjoint_image_splits(self):
        pairs = np.array(list(combinations(range(49), 2)))
        seen = []
        for train, val, test, masks in image_folds(pairs, 49):
            self.assertFalse(set(train) & set(val))
            self.assertFalse(set(train) & set(test))
            self.assertFalse(set(val) & set(test))
            for ids, mask in zip((train, val, test), masks):
                self.assertTrue(set(pairs[mask].flat).issubset(set(ids)))
            seen.extend(np.flatnonzero(masks[2]))
        self.assertEqual(len(seen), 147)
        self.assertEqual(len(set(seen)), 147)

    def test_pair_features_invariant_to_positive_image_scale(self):
        x = np.array([[1., 2.], [3., 1.], [1., -1.]])
        pairs = np.array([[0, 1], [1, 2]])
        np.testing.assert_allclose(pair_features(x, pairs), pair_features(x * np.array([2, 5, 3])[:, None], pairs))

    def test_metric_recovers_anisotropic_distances(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(size=(100, 3))
        y = x @ np.array([.1, 1., 4.])
        w, xs, ys = fit_metric(x[:70], y[:70], 1e-6)
        pred = x[70:] @ w / xs * ys
        self.assertTrue((w >= 0).all())
        self.assertLess(np.mean((pred - y[70:]) ** 2), 1e-6)

    def test_outer_test_targets_do_not_change_fitted_metric(self):
        rng = np.random.default_rng(2)
        pairs = np.array(list(combinations(range(49), 2)))
        x = pair_features(rng.normal(size=(49, 4)), pairs)
        y = x @ np.array([.1, .4, 2., 3.]) + .01
        _, _, _, masks = next(image_folds(pairs, 49))
        _, weights, predictions = adapt(x, y, pairs, 49, penalties=(.01, .1))
        altered = y.copy()
        altered[masks[2]] *= 10
        _, changed, _ = adapt(x, altered, pairs, 49, penalties=(.01, .1))
        np.testing.assert_array_equal(weights[0], changed[0])
        self.assertEqual(len(predictions), 147)


if __name__ == '__main__':
    unittest.main()
