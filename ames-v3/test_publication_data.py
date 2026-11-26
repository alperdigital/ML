"""Independently reconstruct the published comparison from the raw houses and receipts."""
import csv
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "ames-v3/results"
VERSIONS = ("v1", "v2", "v3")


def records(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def report(path):
    return json.loads(path.read_text())


def scores(actual, predicted):
    residual = actual - predicted
    return {
        "rmsle": np.sqrt(np.mean((np.log1p(actual) - np.log1p(predicted)) ** 2)),
        "mae": np.mean(np.abs(residual)),
        "r2": 1 - np.dot(residual, residual) / np.sum((actual - actual.mean()) ** 2),
    }


class PublicationDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = records(RESULTS / "three-version-holdout.csv")
        cls.actual = np.array([float(row["SalePrice"]) for row in cls.rows])
        cls.pred = {v: np.array([float(row[v]) for row in cls.rows]) for v in VERSIONS}
        cls.release = report(RESULTS / "three-version-holdout.json")

    def test_seed_42_houses_match_source_final_and_interactive_site(self):
        source = ROOT / "data/train.csv"
        self.assertEqual(self.release["dataset_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        raw = records(source)
        self.assertEqual(len(raw), 1460)
        self.assertEqual(len({row["Id"] for row in raw}), 1460)
        masonry = [row["MasVnrType"] for row in raw]
        self.assertEqual((masonry.count("None"), masonry.count("NA")), (864, 8))
        indices = np.random.RandomState(42).permutation(1460)
        for name, split in (("development", indices[292:]), ("historical", indices[:292])):
            self.assertEqual(hashlib.sha256(split.astype("<i8").tobytes()).hexdigest(),
                             self.release["ordered_split_sha256"][name])
        self.assertEqual((self.release["development_rows"], self.release["holdout_rows"]), (1168, 292))
        expected = [raw[i] for i in indices[:292]]
        final = records(RESULTS / "final.csv")
        for field in ("Id", "Neighborhood", "YrSold", "SalePrice"):
            for rows in (self.rows, final):
                self.assertEqual([r[field] for r in rows], [r[field] for r in expected])

        frozen = report(RESULTS / "frozen-final.json")
        self.assertEqual(frozen["input_schema"], self.release["input_schema"])
        self.assertEqual(self.release["v3_frozen_recipe"], frozen["recipe"])
        selected = "cand_" + frozen["best_new"] if frozen["recipe"] == "best_new" else frozen["recipe"]
        for key in ("v3", selected):
            np.testing.assert_allclose(self.pred["v3"], [float(r[key]) for r in final], rtol=0, atol=1e-7)
        web = report(ROOT / "docs/house-results.json")
        self.assertEqual(len(web), 292)
        for item, row in zip(web, sorted(self.rows, key=lambda r: int(r["Id"]))):
            self.assertEqual(item, [int(row["Id"]), row["Neighborhood"], int(row["YrSold"]),
                                    *[round(float(row[k]), 2) for k in ("SalePrice", *VERSIONS)]])

    def test_three_metrics_match_the_correct_upstream_experiments(self):
        comparison = report(ROOT / "ames-v2/v1-v2-comparison.json")
        published = report(ROOT / "ames-v2/results.json")
        nested = report(RESULTS / "summary.json")
        self.assertEqual(self.release["input_schema"], "ames-v2-literal-none")
        for receipt in (comparison, published, nested):
            self.assertEqual(receipt["dataset_sha256"], self.release["dataset_sha256"])
            self.assertEqual(receipt["input_schema"], self.release["input_schema"])
        v1 = min((r for r in comparison["candidates"] if r["version"] == "v1"), key=lambda r: r["cv_rmsle"])
        v2 = min(published["candidates"], key=lambda r: r["cv_rmsle"])
        self.assertEqual(v1["model"], self.release["models"]["v1"])
        self.assertEqual(v2["model"], self.release["models"]["v2"])
        self.assertEqual(v2["params"], self.release["v2_published_params"])
        for v, prices in self.pred.items():
            self.assertTrue(np.all(np.isfinite(prices) & (prices >= 0)))
            for metric, value in scores(self.actual, prices).items():
                reference = {"v1": v1["holdout"][metric], "v2": published["selected"]["holdout_" + metric],
                             "v3": nested["holdout"]["v3"][metric]}
                with self.subTest(version=v, metric=metric):
                    self.assertAlmostEqual(value, reference[v], places=7)
                    self.assertAlmostEqual(value, self.release["metrics"][v][metric], places=7)

    def test_pairwise_differences_and_house_bootstrap(self):
        y = self.actual
        for old, new in (("v1", "v2"), ("v2", "v3"), ("v1", "v3")):
            with self.subTest(previous=old, current=new):
                a, b = self.pred[old], self.pred[new]
                left, right = scores(y, a), scores(y, b)
                delta = self.release["differences"][f"{new}_minus_{old}"]
                for metric, key in (("rmsle", "rmsle"), ("mae", "mae_usd"), ("r2", "r2_points")):
                    self.assertAlmostEqual(delta[key], right[metric] - left[metric], places=7)
                for metric in ("rmsle", "mae"):
                    self.assertAlmostEqual(delta[metric + "_pct"], 100 * (right[metric]/left[metric] - 1))
                ci = delta["paired_interval"]
                self.assertEqual((ci["seed"], ci["draws"], ci["unit"]), (42, 20_000, "house"))
                self.assertIs(ci["conditional_on_fixed_predictions"], True)
                rng = np.random.default_rng(42)
                log_a = (np.log1p(y) - np.log1p(a)) ** 2
                log_b = (np.log1p(y) - np.log1p(b)) ** 2
                savings = np.abs(y - a) - np.abs(y - b)
                gains, dollars = [], []
                for _ in range(200):
                    draw = rng.integers(len(y), size=(100, len(y)))
                    ratio = log_b[draw].sum(axis=1) / log_a[draw].sum(axis=1)
                    gains.extend(100 * (1 - np.sqrt(ratio)))
                    dollars.extend(savings[draw].mean(axis=1))
                for values, key in ((gains, "rmsle_gain_pct_ci95"), (dollars, "mae_reduction_usd_ci95")):
                    np.testing.assert_allclose(np.quantile(values, [.025, .975]), ci[key], rtol=0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
