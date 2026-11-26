"""Checks for the cached V2 objective and the paired study partitions."""
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold

import baselines
from design import partitions


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "Id": np.arange(100, 112),
            "Area": [10., 12., np.nan, 15., 17., 18., 20., 23., 26., 30., 32., 41.],
            "Neighborhood": ["A", "B", "A", "B", "C", "A", "A", "C", "D", "A", "B", "A"],
        })
        self.target = pd.Series(np.arange(90_000., 210_000., 10_000.))
        self.splits = list(KFold(3, shuffle=True, random_state=42).split(self.frame))

    def test_cached_features_reproduce_published_estimator_objective(self):
        X = baselines._without_id(self.frame)
        folds = baselines._prepare_folds(X, self.target, self.splits, engineered=False)
        predictions, scores = baselines._evaluate("ridge", {"alpha": 3.}, folds, len(X))
        config = {"model": "ridge", "params": {"alpha": 3.}, "engineered": False}
        for fold, (train, valid) in enumerate(self.splits):
            pipeline = baselines.fit_v2(config, self.frame.iloc[train], self.target.iloc[train])
            ref = pipeline.predict(self.frame.iloc[valid])
            np.testing.assert_allclose(predictions[valid], ref, rtol=1e-10)
            wanted = np.sqrt(mean_squared_log_error(self.target.iloc[valid], ref))
            self.assertAlmostEqual(scores[fold], wanted, places=12)

    def test_changing_validation_features_or_labels_cannot_modify_training_matrix(self):
        train, valid = self.splits[0]
        X = baselines._without_id(self.frame)
        y = self.target.copy()
        original = baselines._prepare_folds(X, y, [(train, valid)], engineered=True)[0]
        X.loc[valid, "Area"] = 100000000.
        X.loc[valid, "Neighborhood"] = "UNSEEN"
        y.loc[valid] = 1.
        changed = baselines._prepare_folds(X, y, [(train, valid)], engineered=True)[0]
        np.testing.assert_array_equal(original[0], changed[0])
        np.testing.assert_array_equal(original[2], changed[2])
        self.assertFalse(np.array_equal(original[1], changed[1]))

    def test_inner_splits_reject_duplicates_and_overlap(self):
        train, valid = self.splits[0]
        with self.assertRaisesRegex(ValueError, "cover"):
            baselines.select_v2(self.frame, self.target, [(train, valid)] * 3, trials=0)
        contaminated = [(np.append(train, valid[0]), valid), *self.splits[1:]]
        with self.assertRaisesRegex(ValueError, "overlap"):
            baselines.select_v2(self.frame, self.target, contaminated, trials=0)

    def test_reference_order_and_outer_and_forward_partitions(self):
        frame = pd.DataFrame({"YrSold": 2006 + np.arange(1460) % 5})
        tasks = partitions(frame)
        development, historical, _, _ = tasks["final"]
        self.assertEqual(baselines.legacy.split_fingerprint(development),
                         "fbb0ef6b3187b2d632840fec65c796d3cf706eaed200f2f8952534d2e7137e41")
        self.assertEqual(baselines.legacy.split_fingerprint(historical),
                         "471ba7654e8fd463e2fd39e521b7b05863c8692941e04b5f54f2525b2bb68ead")
        for repeat in range(2):
            evaluations = []
            for fold in range(5):
                train, validation, _, _ = tasks[f"r{repeat}f{fold}"]
                self.assertEqual(len(np.intersect1d(train, validation)), 0)
                self.assertEqual(set(train) | set(validation), set(development))
                evaluations.extend(validation)
            np.testing.assert_array_equal(np.sort(evaluations), np.sort(development))
        past, future, _, _ = tasks["temporal"]
        self.assertTrue(frame.YrSold.iloc[past].max() < frame.YrSold.iloc[future].min())
        self.assertEqual(len(np.intersect1d(np.union1d(past, future), historical)), 0)


if __name__ == "__main__":
    unittest.main()
