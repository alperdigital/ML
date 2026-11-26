import unittest

import numpy as np
import pandas as pd

from predict import prepare_row, price, selected_recipe


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, row):
        return np.full(len(row), self.value)


class PredictionTests(unittest.TestCase):
    columns = ["Id", "count", "frontage", "zone"]
    dtypes = {"Id": "int64", "count": "int64", "frontage": "float64", "zone": "object"}

    def frame(self):
        return pd.DataFrame({"zone": ["01", None], "frontage": ["50", None],
                             "count": ["3", "1"], "Id": ["1", "2"]})

    def test_single_input_schema_and_coercion(self):
        row = prepare_row(self.frame(), self.columns, self.dtypes, house_id=1)
        self.assertEqual(row.columns.tolist(), self.columns)
        self.assertEqual(row.zone.iloc[0], "01")
        self.assertEqual(row.frontage.dtype, "float64")
        self.assertEqual(row["count"].dtype, "int64")
        missing = prepare_row(self.frame(), self.columns, self.dtypes, house_id=2)
        self.assertTrue(missing.frontage.isna().all() and missing.zone.isna().all())

    def test_invalid_or_ambiguous_inputs(self):
        for data in [self.frame().drop(columns="count"), self.frame().assign(SalePrice=10),
                     self.frame().assign(count="unknown"), self.frame().assign(count="1.5"),
                     self.frame().assign(frontage="inf")]:
            with self.subTest(columns=data.columns.tolist()), self.assertRaises(ValueError):
                prepare_row(data, self.columns, self.dtypes, 1)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            prepare_row(self.frame(), self.columns, self.dtypes)

    def test_full_bundle_blend_and_recommended_v2_member(self):
        audit = {"recipe": "weighted_blend", "member_order": ["v1_xgb", "v2_cat", "best_new"],
                 "weights": [0.5, 0., 0.5]}
        bundle = {"audit": audit, "models": {"v1_xgb": ConstantModel(99.), "v2_cat": ConstantModel(100.),
                  "best_new": ConstantModel(399.), "v2": ConstantModel(123.)}}
        row = prepare_row(self.frame(), self.columns, self.dtypes, 1)
        self.assertEqual(price(bundle, row, "v3")["predicted_sale_price_usd"], 199.)
        self.assertEqual(price(bundle, row, "v2")["predicted_sale_price_usd"], 123.)
        self.assertEqual(selected_recipe(audit, "v2"), "v2")


if __name__ == "__main__":
    unittest.main()
