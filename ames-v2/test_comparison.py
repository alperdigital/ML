import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from compare_v1_v2 import TreeFeatures, fill_validation, metrics, split_fingerprint


class ComparisonTests(unittest.TestCase):
    def test_legacy_imputation_uses_training_statistics(self):
        train = pd.DataFrame({"Neighborhood": ["A", "A", "B"], "LotFrontage": [10., 30., 90.],
                              "Electrical": ["X", "X", "Y"], "Other": [1., 2., 9.]})
        val = pd.DataFrame({"Neighborhood": ["A", "C", "A"], "LotFrontage": [np.nan, np.nan, 1000.],
                            "Electrical": [None, None, "Z"], "Other": [np.nan, np.nan, 999.]})
        pp = SimpleNamespace(strategies={"LotFrontage": "NeighborhoodMedian", "Electrical": "Mode"})
        result = fill_validation(pp, train, val, train)
        self.assertEqual(result["LotFrontage"].tolist(), [20., 30., 1000.])
        self.assertEqual(result["Electrical"].tolist(), ["X", "X", "Z"])
        self.assertEqual(result["Other"].tolist(), [2., 2., 999.])
        self.assertTrue(pd.isna(val.loc[0, "LotFrontage"]))

    def test_tree_features_use_real_years_and_room_counts(self):
        raw = pd.DataFrame({
            "YrSold": [2008, 2009, 2009], "YearBuilt": [2003, 1990, 1998],
            "YearRemodAdd": [2003, 2000, 2007],
            "TotRmsAbvGrd": [8, 4, 6], "BedroomAbvGr": [3, 2, 2],
            "Neighborhood": ["A", "A", "B"], "LotFrontage": [10., 30., 60.],
            "GrLivArea": [1500, 4800, 1800],
        })
        features = TreeFeatures()
        tr, target = features.fit_transform(raw, pd.Series([200_000, 180_000, 190_000]))
        self.assertEqual(target.index.tolist(), [0, 2])
        self.assertEqual(tr.loc[0, ["HouseAge", "RemodelAge", "TotalRooms"]].tolist(), [5, 5, 8])

        # A sale year and neighborhood unseen at training time arrive together.
        holdout = raw.iloc[[1]].assign(YrSold=2010, Neighborhood="C", LotFrontage=np.nan)
        result = features.transform(holdout)
        self.assertEqual(result.index.tolist(), [1])
        self.assertEqual(result.columns.tolist(), tr.columns.tolist())
        self.assertEqual(result.iloc[0][["YrSold", "HouseAge", "LotFrontage"]].tolist(), [2010, 20, 30])
        self.assertEqual(result.filter(like="Neighborhood_").sum(axis=1).iloc[0], 0)
        # The same house is encoded identically if other sales are scored alongside it.
        mixed = pd.concat([holdout, raw.iloc[[2]].assign(LotFrontage=50_000.)])
        pd.testing.assert_frame_equal(result, features.transform(mixed).iloc[:1])

    def test_common_metrics_and_split_identity(self):
        self.assertEqual(metrics(np.array([10., 20.]), np.array([10., 20.])),
                         {"rmsle": 0., "r2": 1., "mae": 0.})
        self.assertEqual(split_fingerprint([1, 2]), split_fingerprint(np.array([1, 2])))
        self.assertNotEqual(split_fingerprint([1, 2]), split_fingerprint([2, 1]))


if __name__ == "__main__":
    unittest.main()
