import unittest
from io import StringIO
import numpy as np
import pandas as pd
from benchmark import estimator, rmsle, Features, read_ames_csv

class BenchmarkTests(unittest.TestCase):
    def test_rmsle_raw_prices(self):
        self.assertAlmostEqual(rmsle([100, 200], [100, 400]),
                               abs(np.log1p(400) - np.log1p(200)) / np.sqrt(2))
        self.assertEqual(rmsle([0], [-10]), 0)

    def test_rmsle_requires_exactly_one_price_pair_per_row(self):
        labels = pd.Series([100., 200., 80.])
        pred = pd.Series([120., 200., 72.])
        reference = rmsle(labels, pred)
        for actual in (labels, labels.to_frame()):
            for estimated in (pred, pred.to_frame()):
                self.assertEqual(rmsle(actual, estimated), reference)
        for invalid in (np.column_stack((pred, pred)), pred.iloc[1:], [], 100., [np.inf] * 3, [np.nan] * 3):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                rmsle(labels, invalid)
        with self.assertRaises(ValueError):
            rmsle([-1], [1])

    def test_masonry_absence_survives_csv_in_a_batch_and_single_house(self):
        rows = read_ames_csv(StringIO("Id,MasVnrType,MasVnrArea\n1,None,0\n2,NA,NA\n3,,\n"))
        self.assertEqual(rows.MasVnrType.iloc[0], "None")
        self.assertTrue(rows.MasVnrType.iloc[1:].isna().all())
        self.assertTrue(rows.MasVnrArea.iloc[1:].isna().all())
        model = estimator("ridge").fit(rows[["MasVnrType"]], [100., 200., 300.])
        prep = model.regressor_.named_steps["prep"]
        np.testing.assert_array_equal(prep.transform(rows[["MasVnrType"]]), [[1., 0.], [0., 1.], [0., 1.]])
        house = read_ames_csv(StringIO("MasVnrType\nNone\n"), dtype=str)
        self.assertEqual(house.iloc[0, 0], "None")

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
