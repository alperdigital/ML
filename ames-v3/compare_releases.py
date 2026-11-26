"""Recreate the three published releases' paired predictions on the historical holdout."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from design import DATA_SHA, load_data, partitions

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ames-v2"))
from compare_v1_v2 import configure_v1_imports, metrics, split_fingerprint, v1_predict, v2_predict


def read_json(path):
    return json.loads((ROOT / path).read_text())


def run():
    df = load_data(ROOT / "data/train.csv")
    dev, test, _, _ = partitions(df)["final"]
    comparison = read_json("ames-v2/v1-v2-comparison.json")
    published = read_json("ames-v2/results.json")
    frozen = read_json("ames-v3/results/frozen-final.json")
    summary = read_json("ames-v3/results/summary.json")
    predictions_dir = ROOT / "ames-v3/results"
    final = pd.read_csv(predictions_dir / "final.csv")

    ids = {"development": split_fingerprint(dev), "historical": split_fingerprint(test)}
    if (ids != summary["splits"] or ids["development"] != comparison["development_index_sha256"]
            or ids["historical"] != comparison["holdout_index_sha256"]):
        raise ValueError("Published split fingerprints differ")
    if any(obj["dataset_sha256"] != DATA_SHA for obj in (comparison, published, summary)):
        raise ValueError("Published source datasets differ")
    houses = df.iloc[test][["Id", "Neighborhood", "YrSold", "SalePrice"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(houses, final[houses.columns], check_exact=True)
    if len(dev) != 1168 or len(houses) != 292 or not houses.Id.is_unique:
        raise ValueError("Unexpected development or holdout cohort")

    training = df.iloc[dev].copy()
    y = training.pop("SalePrice")
    X = df.iloc[test].drop(columns="SalePrice")
    configure_v1_imports(ROOT)
    v1 = comparison["selected_by_cv"]["v1"]
    v1_reference = next(r for r in comparison["candidates"] if r["version"] == "v1" and r["model"] == v1)
    houses["v1"], v1_train_rows, v1_params = v1_predict(v1, training, y, X)
    if v1_params != v1_reference["params"] or v1_train_rows != v1_reference["development_train_rows"]:
        raise ValueError("V1 configuration differs from its published comparison")

    # The original five-fold V2 release and the nested V3 experiment tuned V2
    # separately. Use the former's winning configuration to compare releases.
    v2 = min(published["candidates"], key=lambda row: row["cv_rmsle"])
    if (v2["model"], v2["engineered"]) != (published["selected"]["model"], published["selected"]["engineered"]):
        raise ValueError("V2 configuration differs from its published selection")
    houses["v2"], v2_train_rows, _ = v2_predict(v2, training, y, X)
    if frozen["recipe"] != "best_new" or frozen["best_new"] != "tabpfn_v2":
        raise ValueError("Unexpected frozen V3 recipe")
    np.testing.assert_array_equal(final.v3, final.cand_tabpfn_v2)
    houses["v3"] = final.v3.to_numpy()

    targets = {
        "v1": v1_reference["holdout"],
        "v2": {m: published["selected"]["holdout_" + m] for m in ("rmsle", "mae", "r2")},
        "v3": summary["holdout"]["v3"],
    }
    scores = {}
    for version, reference in targets.items():
        scores[version] = metrics(houses.SalePrice, houses[version])
        for metric, observed in scores[version].items():
            np.testing.assert_allclose(observed, reference[metric], rtol=1e-8, atol=1e-9)

    differences = {}
    for old, new in [("v1", "v2"), ("v2", "v3"), ("v1", "v3")]:
        a, b = scores[old], scores[new]
        differences[f"{new}_minus_{old}"] = {
            "rmsle": b["rmsle"] - a["rmsle"],
            "rmsle_pct": 100 * (b["rmsle"] / a["rmsle"] - 1),
            "mae_usd": b["mae"] - a["mae"],
            "mae_pct": 100 * (b["mae"] / a["mae"] - 1),
            "r2_points": b["r2"] - a["r2"],
        }
    audit = {
        "dataset_sha256": DATA_SHA, "ordered_split_sha256": ids, "seed": 42,
        "development_rows": len(dev), "holdout_rows": len(test),
        "effective_training_rows": {"v1": v1_train_rows, "v2": v2_train_rows, "v3": len(dev)},
        "models": {"v1": v1, "v2": v2["model"], "v3": frozen["best_new"]},
        "v2_published_params": v2["params"], "metrics": scores, "differences": differences,
        "source_commit": {"v1": comparison["v1_source_commit"],
                          "v2": "7473f585685b15cf886daa32041f8f293e887d1e",
                          "v3": "2fbce4aed48813bf58f0e00b63729f86e3d209ef"},
    }
    houses.to_csv(predictions_dir / "three-version-holdout.csv", index=False)
    (predictions_dir / "three-version-holdout.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    # The portfolio uses this compact, price-rounded view of the same records.
    website_rows = [
        [house_id, neighborhood, year, *(round(float(price), 2) for price in prices)]
        for house_id, neighborhood, year, *prices in
        houses.sort_values("Id").itertuples(index=False, name=None)
    ]
    (ROOT / "docs/house-results.json").write_text(
        json.dumps(website_rows, separators=(",", ":"), allow_nan=False) + "\n"
    )
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    run()
