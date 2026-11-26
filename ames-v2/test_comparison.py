import json
import unittest
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pandas as pd
from compare_v1_v2 import TreeFeatures, fill_validation, metrics, split_fingerprint
from sklearn.model_selection import KFold, train_test_split
from src.ames_data import INPUT_SCHEMA, read_ames_csv
from src.data_preprocessing import DataPreprocessor


class ComparisonTests(unittest.TestCase):
    def test_published_search_has_twenty_matching_trials_per_family(self):
        folder = Path(__file__).parent
        benchmark = json.loads((folder / "results.json").read_text())
        history = json.loads((folder / "optuna-trials-20.json").read_text())
        self.assertEqual(benchmark["input_schema"], INPUT_SCHEMA)
        for key in ("seed", "folds", "input_schema", "dataset_sha256"):
            self.assertEqual(history[key], benchmark[key])
        self.assertEqual(history["trials_per_model"], benchmark["trials"])
        self.assertEqual(history["trials_per_model"], 20)
        self.assertEqual(set(history["trials"]), {"ridge", "xgboost", "lightgbm", "catboost"})
        for family, trials in history["trials"].items():
            with self.subTest(family=family):
                self.assertEqual([row["trial"] for row in trials], list(range(20)))
                winning_trial = min(trials, key=lambda row: row["cv_rmsle"])
                engineered = next(c for c in benchmark["candidates"] if c["engineered"] and c["model"] == family)
                self.assertEqual(winning_trial["params"], engineered["params"])
                self.assertAlmostEqual(winning_trial["cv_rmsle"], np.mean(engineered["folds"]), places=9)

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

    def test_unseen_quality_is_reserved_without_refitting(self):
        raw = read_ames_csv(Path(__file__).resolve().parents[1] / "data/train.csv")
        dev, _ = train_test_split(raw, test_size=.2, random_state=42)
        train, validation = list(KFold(5, shuffle=True, random_state=42).split(dev))[4]
        fit = dev.iloc[train]
        probe = dev.iloc[validation].set_index("Id").loc[[515, 247], ["GarageQual"]]
        self.assertEqual(probe.GarageQual.tolist(), ["Po", "Po"])
        pp = DataPreprocessor()
        pp.encode_categorical(fit[["GarageQual"]])
        original = pp.label_encoders["GarageQual"].classes_.copy()
        self.assertNotIn("Po", original)
        self.assertIn("Ex", original)
        self.assertEqual(pp.encode_categorical(probe, fit=False).GarageQual.tolist(), [-1, -1])
        np.testing.assert_array_equal(original, pp.label_encoders["GarageQual"].classes_)

    def test_masonry_presence_is_distinct_and_identifier_is_not_a_feature(self):
        raw = pd.DataFrame({"Id": [1, 2, 3], "MasVnrType": ["None", np.nan, "BrkFace"]})
        prep = TreeFeatures()
        x, _ = prep.fit_transform(raw, pd.Series([150_000., 120_000., 200_000.]))
        self.assertNotIn("Id", x.columns)
        self.assertEqual(x["MasVnrType_None"].tolist(), [1., 0., 0.])
        self.assertEqual(x["MasVnrType___MISSING__"].tolist(), [0., 1., 0.])
        pd.testing.assert_frame_equal(x, prep.transform(raw.assign(Id=[77, 40, 106])))

    def test_common_metrics_and_split_identity(self):
        self.assertEqual(metrics(np.array([10., 20.]), np.array([10., 20.])),
                         {"rmsle": 0., "r2": 1., "mae": 0.})
        self.assertEqual(metrics([[0.], [20.]], [-7., 20.]), {"rmsle": 0., "r2": 1., "mae": 0.})
        with self.assertRaises(ValueError):
            metrics([0., np.nan], [0., 20.])
        self.assertEqual(split_fingerprint([1, 2]), split_fingerprint(np.array([1, 2])))
        self.assertNotEqual(split_fingerprint([1, 2]), split_fingerprint([2, 1]))


if __name__ == "__main__":
    unittest.main()
