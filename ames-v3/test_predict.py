import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from baselines import V1Predictor
from design import load_data, partitions
from predict import ROOT, fit_recipe, main, prepare_row, price, recommended_version, selected_recipe
from src.ames_data import INPUT_SCHEMA


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

    def test_external_cli_preserves_masonry_absence_and_unknown(self):
        class MasonryModel:
            def predict(self, row):
                veneer = row.MasVnrType.iloc[0]
                return np.array([100. if veneer == "None" else 200.])

        bundle = {"input_schema": INPUT_SCHEMA, "audit": {}, "fit_version": "v2",
                  "models": {"v2_release": MasonryModel()}, "columns": ["Id", "MasVnrType"],
                  "dtypes": {"Id": "int64", "MasVnrType": "object"}}
        with TemporaryDirectory() as folder:
            path = Path(folder) / "one.csv"
            for literal, expected in (("None", 100.), ("NA", 200.)):
                path.write_text(f"Id,MasVnrType\n1,{literal}\n")
                output = StringIO()
                with self.subTest(literal=literal), patch("predict.joblib.load", return_value=bundle), \
                        patch("sys.argv", ["predict.py", "--version", "v2", "--bundle", "local.joblib",
                                           "--input", str(path)]), redirect_stdout(output):
                    main()
                    self.assertEqual(json.loads(output.getvalue())["predicted_sale_price_usd"], expected)

    def test_historical_tabpfn_replays_ordered_unlabeled_cohort_and_external_input_stays_single(self):
        data = load_data(ROOT / "data/train.csv")
        _, holdout, _, _ = partitions(data)["final"]
        houses = data.iloc[holdout].drop(columns="SalePrice")
        house_id = int(houses.Id.iloc[137])
        calls = []

        class BatchSensitiveModel:
            def predict(self, frame):
                calls.append((frame.Id.tolist(), frame.columns.tolist()))
                return 100.0 + len(frame) + np.arange(len(frame)) / 100

        model = BatchSensitiveModel()
        member_order = ["v1_xgb", "best_new"]
        audit = {"best_new": "candidate", "member_order": member_order, "weights": [0.4, 0.6]}
        bundle = {"input_schema": INPUT_SCHEMA, "audit": audit, "columns": houses.columns.tolist(),
                  "dtypes": houses.dtypes.astype(str).to_dict(),
                  "models": {name: model for name in (*member_order, "v2", "v2_release")}}
        with TemporaryDirectory() as folder:
            path = Path(folder) / "house.csv"
            houses.iloc[[137]].to_csv(path, index=False, na_rep="NA")
            for version, recipe, candidate, external, replay in (
                    ("v3", "best_new", "tabpfn", False, True),
                    ("v3", "weighted_blend", "tabpfn", False, True),
                    ("v3", "equal_blend", "tabpfn", False, True),
                    ("v3", "weighted_blend", "tabpfn", True, False),
                    ("v3", "best_new", "tabpfn", True, False),
                    ("v3", "best_new", "catboost", False, False),
                    ("v2", "best_new", "tabpfn", False, False),
                    ("v2-nested", "best_new", "tabpfn", False, False)):
                audit.update(recipe=recipe, specs={"candidate": {"model": candidate}})
                source = ["--input", str(path)] if external else ["--holdout-id", str(house_id)]
                calls.clear()
                output = StringIO()
                with self.subTest(version=version, recipe=recipe, candidate=candidate, external=external), \
                        patch("predict.joblib.load", return_value=bundle), \
                        patch("sys.argv", ["predict.py", "--bundle", "local.joblib", "--version", version, *source]), \
                        redirect_stdout(output):
                    main()
                    payload = json.loads(output.getvalue())
                    self.assertEqual(payload["Id"], house_id)
                    self.assertEqual(payload["predicted_sale_price_usd"], 393.37 if replay else 101.)
                    expected_ids = houses.Id.tolist() if replay else [house_id]
                    n_models = 2 if version == "v3" and recipe in ("equal_blend", "weighted_blend") else 1
                    self.assertEqual(calls, [(expected_ids, houses.columns.tolist())] * n_models)

    def test_full_bundle_blend_and_separate_nested_v2_member(self):
        audit = {"recipe": "weighted_blend", "member_order": ["v1_xgb", "v2_cat", "best_new"],
                 "weights": [0.5, 0., 0.5]}
        bundle = {"input_schema": INPUT_SCHEMA, "audit": audit, "models": {"v1_xgb": ConstantModel(99.), "v2_cat": ConstantModel(100.),
                  "best_new": ConstantModel(399.), "v2": ConstantModel(123.)}}
        row = prepare_row(self.frame(), self.columns, self.dtypes, 1)
        self.assertEqual(price(bundle, row, "v3")["predicted_sale_price_usd"], 199.)
        self.assertEqual(price(bundle, row, "v2-nested")["predicted_sale_price_usd"], 123.)
        with self.assertRaisesRegex(ValueError, "lacks the published V2"):
            price(bundle, row, "v2")
        self.assertEqual(selected_recipe(audit, "v2"), "v2_release")
        self.assertEqual(recommended_version({"input_schema": INPUT_SCHEMA, "promote_v3": False}), "v2-nested")
        self.assertEqual(recommended_version({"input_schema": INPUT_SCHEMA, "promote_v3": True}), "v3")
        with self.assertRaisesRegex(ValueError, "incompatible Ames input schema"):
            recommended_version({"promote_v3": True})

    def test_legacy_lightweight_v2_bundle_and_release_bundle(self):
        row = prepare_row(self.frame(), self.columns, self.dtypes, 1)
        legacy = {"fit_version": "v2", "audit": {}, "models": {"v2": ConstantModel(123.)}}
        release = {"input_schema": INPUT_SCHEMA, "fit_version": "v2", "audit": {},
                   "models": {"v2_release": ConstantModel(456.)}}
        for version in ("v2", "v2-nested", "v3"):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "incompatible Ames input schema"):
                price(legacy, row, version)
        for version in ("v2-nested", "v3"):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "incompatible Ames input schema"):
                fit_recipe(row, {}, version)
        self.assertEqual(price(release, row, "v2")["predicted_sale_price_usd"], 456.)
        with self.assertRaisesRegex(ValueError, "Bundle fitted for v2, requested v2-nested"):
            price(release, row, "v2-nested")

    def test_legacy_v1_member_requires_refit_only_when_used(self):
        # Unpickling an old V1Predictor restores no feature_schema_version.
        legacy = {"input_schema": INPUT_SCHEMA, "models": {"v1_xgb": V1Predictor(), "v2": ConstantModel(123.),
                             "best_new": ConstantModel(456.)},
                  "audit": {"recipe": "best_new", "member_order": ["v1_xgb", "best_new"]}}
        row = prepare_row(self.frame(), self.columns, self.dtypes, 1)
        self.assertEqual(price(legacy, row, "v2-nested")["predicted_sale_price_usd"], 123.)
        self.assertEqual(price(legacy, row, "v3")["predicted_sale_price_usd"], 456.)
        legacy["audit"]["recipe"] = "equal_blend"
        with self.assertRaisesRegex(ValueError, "Refit without --bundle"):
            price(legacy, row, "v3")

    def test_single_house_v2_matches_all_portfolio_prices_to_the_cent(self):
        data = load_data(ROOT / "data/train.csv")
        development, holdout, _, _ = partitions(data)["final"]
        bundle = fit_recipe(data.iloc[development], {}, "v2")
        houses = data.iloc[holdout].drop(columns="SalePrice")
        site_rows = json.loads((ROOT / "docs/house-results.json").read_text())
        self.assertEqual(len(site_rows), 292)
        self.assertCountEqual((row[0] for row in site_rows), houses.Id.tolist())
        for house_id, _, _, _, _, site_v2, _ in site_rows:
            with self.subTest(Id=house_id):
                row = prepare_row(houses, bundle["columns"], bundle["dtypes"], house_id)
                self.assertEqual(price(bundle, row, "v2")["predicted_sale_price_usd"], site_v2)


if __name__ == "__main__":
    unittest.main()
