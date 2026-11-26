"""The aggregate must pair the same homes across repeats before resampling."""
import unittest

import numpy as np
import pandas as pd

from report import _bootstrap, summarize


class ReportTests(unittest.TestCase):
    def setUp(self):
        n = 50
        row = np.arange(n)
        self.dev = pd.DataFrame({
            "Id": row, "repeat": 0, "fold": row % 5,
            "SalePrice": 10000. * (1 + row),
            "YrSold": 2008 + row % 2,
            "Neighborhood": np.where(row < 10, "Rare", "Common"),
        })
        self.dev["v2"] = self.dev.SalePrice * (1.005 + row / 100)
        self.dev["v3"] = self.dev.SalePrice * (1.005 + row / 200)
        self.held = self.dev.iloc[:20].assign(Id=lambda f: f.Id + n)
        self.outer = pd.concat([self.dev, self.dev.assign(repeat=1, v3=self.dev.SalePrice * .9)])

    def test_pooled_score_and_unique_house_counts(self):
        r = summarize(self.outer, self.dev, self.held)
        expected = np.sqrt(np.mean((np.log1p(self.outer.v3) - np.log1p(self.outer.SalePrice)) ** 2))
        self.assertAlmostEqual(r["outer"]["v3"]["rmsle"], expected)
        self.assertEqual(r["outer"]["properties"], 50)
        self.assertEqual(r["outer"]["predictions"], 100)
        self.assertEqual(set(r["outer"]["by_neighborhood_min20"]), {"Common"})
        self.assertEqual(r["promote_v3"], r["outer"]["rmsle_gain_pct"] >= 2 and
                         r["outer"]["mae_change_pct"] < 1)

    def test_bootstrap_rejects_incomplete_pairs(self):
        with self.assertRaisesRegex(ValueError, "two paired repeats"):
            _bootstrap(self.outer.iloc[1:])

    def test_duplicate_and_overlapping_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "one prediction per property"):
            summarize(pd.concat([self.outer, self.outer.iloc[:1]]), self.dev, self.held)
        with self.assertRaisesRegex(ValueError, "overlap"):
            summarize(self.outer, self.dev, self.dev)

    def test_bootstrap_is_deterministic_under_row_permutation(self):
        a = _bootstrap(self.outer)
        b = _bootstrap(self.outer.sample(frac=1, random_state=42))
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
