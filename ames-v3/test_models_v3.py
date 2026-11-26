import unittest

import numpy as np
import pandas as pd

from models_v3 import ABSENT, MISSING, AmesFeatures, blend_predict, blend_weights, candidates, fit_model


def houses():
    return pd.DataFrame({
        "Id": [1, 2, 3, 4], "Neighborhood": ["A", "A", "B", "B"],
        "LotFrontage": [10., 30., 90., np.nan], "Fireplaces": [0, 1, 1, 0],
        "FireplaceQu": [None, "TA", None, "Gd"], "GarageArea": [0, 10, 0, 20],
        "GarageCars": [0, 1, 0, 1], "GarageType": [None, "Attchd", None, "Attchd"],
        "GarageFinish": [None, "Fin", None, "Fin"], "GarageQual": [None, "TA", None, "TA"],
        "GarageCond": [None, "TA", None, "TA"], "GarageYrBlt": [np.nan, 1980., np.nan, 2000.],
        "TotalBsmtSF": [0, 60, 0, 50], "BsmtQual": [None, "TA", None, "TA"],
        "BsmtCond": [None, "TA", None, "TA"], "BsmtExposure": [None, "Av", None, "Av"],
        "BsmtFinType1": [None, "LwQ", None, "GLQ"], "BsmtFinType2": [None, "Unf", None, "Unf"],
        "PoolArea": [0, 0, 0, 0], "PoolQC": [None]*4,
        "1stFlrSF": [100, 110, 120, 200], "2ndFlrSF": [0, 80, 0, 50],
        "FullBath": [1, 2, 1, 2], "HalfBath": [0, 0, 1, 1], "BsmtFullBath": [0, 1, 0, 1],
        "BsmtHalfBath": [0]*4, "YrSold": [2008]*4, "YearBuilt": [1970, 1980, 1990, 2000],
        "YearRemodAdd": [1970, 1980, 1990, 2000], "MSSubClass": [20, 60, 20, 60],
    })


class AmesModelTests(unittest.TestCase):
    def test_structural_vs_unknown_and_conflict(self):
        x = houses()
        values = AmesFeatures(candidates()["native_full"]).fit(x).transform(x)
        self.assertEqual(values.FireplaceQu.tolist(), [ABSENT, "TA", MISSING, "Gd"])
        self.assertEqual(values.GarageType.iloc[0], ABSENT)
        self.assertEqual(values.GarageYrBlt.iloc[0], 0)
        self.assertNotIn("Id", values)

    def test_frontage_uses_training_neighborhood_with_global_fallback(self):
        x = houses()
        val = x.iloc[:3].copy()
        val["Neighborhood"] = ["A", "unseen", "A"]
        val["LotFrontage"] = [np.nan, np.nan, 9000.]
        fe = AmesFeatures(candidates()["native_full"]).fit(x)
        result = fe.transform(val)
        self.assertEqual(result.LotFrontage.tolist(), [20., 30., 9000.])
        self.assertEqual(result.LotFrontageMissing.tolist(), [1., 1., 0.])
        self.assertEqual(fe.transform(val.iloc[:1]).LotFrontage.iloc[0], 20.)

    def test_new_category_and_inference_roundtrip(self):
        import pickle
        x = houses()
        val = x.iloc[:1].assign(Neighborhood="held_out")
        for name in ("native_full", "ridge_full"):
            fitted = fit_model(candidates()[name], x, [1000, 2000, 3000, 4000])
            restored = pickle.loads(pickle.dumps(fitted))
            pred = restored.predict(val)
            self.assertTrue(np.isfinite(pred).all() and (pred >= 0).all())
            np.testing.assert_allclose(pred, fitted.predict(val))

    def test_convex_blend_log_scale(self):
        prices = np.array([[30., 105.], [90., 50.], [220., 260.]])
        expected = np.expm1(np.log1p(prices) @ np.array([0.25, 0.75]))
        weights = blend_weights(expected, prices)
        self.assertTrue((weights >= 0).all())
        self.assertAlmostEqual(weights.sum(), 1.)
        np.testing.assert_allclose(blend_predict(prices, weights), expected, rtol=1e-6)
        np.testing.assert_allclose(blend_weights(expected[:, None], prices), weights)
        with self.assertRaises(ValueError):
            blend_predict(prices, weights[:, None])
        with self.assertRaises(ValueError):
            blend_weights([-0.5, 90., 220.], prices)


if __name__ == "__main__":
    unittest.main()
