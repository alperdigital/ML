import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from compare_v1_v2 import fill_validation, metrics, split_fingerprint


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

    def test_common_metrics_and_split_identity(self):
        self.assertEqual(metrics(np.array([10., 20.]), np.array([10., 20.])),
                         {"rmsle": 0., "r2": 1., "mae": 0.})
        self.assertEqual(split_fingerprint([1, 2]), split_fingerprint(np.array([1, 2])))
        self.assertNotEqual(split_fingerprint([1, 2]), split_fingerprint([2, 1]))


if __name__ == "__main__":
    unittest.main()
