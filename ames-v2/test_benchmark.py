import unittest
import numpy as np
import pandas as pd
from benchmark import estimator, rmsle, Features

class BenchmarkTests(unittest.TestCase):
    def test_rmsle_raw_prices(self):
        self.assertAlmostEqual(rmsle([100, 200], [100, 400]),
                               abs(np.log1p(400) - np.log1p(200)) / np.sqrt(2))
        self.assertEqual(rmsle([0], [-10]), 0)

    def test_fold_local_and_unknown_category(self):
        X = pd.DataFrame({"area": [1., 3., np.nan], "zone": ["a", "b", None]}).replace({None: np.nan})
        model = estimator("ridge").fit(X, [10., 20., 30.])
        prep = model.regressor_.named_steps["prep"]
        self.assertEqual(prep.named_transformers_["num"].named_steps["impute"].statistics_[0], 2)
        pred = model.predict(pd.DataFrame({"area": [1000.], "zone": ["unseen"]}))
        self.assertTrue(np.isfinite(pred).all())
        self.assertEqual(prep.named_transformers_["num"].named_steps["impute"].statistics_[0], 2)

    def test_feature_engineering_does_not_mutate_input(self):
        X = pd.DataFrame({"YrSold": [2010], "YearBuilt": [1990]})
        self.assertEqual(Features().fit_transform(X)["HouseAge"].iloc[0], 20)
        self.assertNotIn("HouseAge", X)
        self.assertNotIn("HouseAge", Features(False).fit_transform(X))

if __name__ == "__main__":
    unittest.main()
