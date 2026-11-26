"""Compare the three corrected procedures, paired by sale on the historical holdout."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from design import DATA_SHA, load_data, partitions
from src.ames_data import INPUT_SCHEMA
from src.ames_metrics import price_vectors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ames-v2"))
from compare_v1_v2 import configure_v1_imports, metrics, split_fingerprint, v1_predict, v2_predict


def read_json(path):
    return json.loads((ROOT / path).read_text())


def paired_intervals(actual, previous, current, draws=20_000, seed=42):
    """Resample the same houses for both models; fitting uncertainty is excluded."""
    y, previous = price_vectors(actual, previous)
    _, current = price_vectors(y, current)
    estimates = np.column_stack([previous, current])
    squared_log = (np.log1p(estimates) - np.log1p(y[:, None])) ** 2
    absolute = np.abs(estimates - y[:, None])
    rng = np.random.default_rng(seed)
    relative, dollars = [], []
    for start in range(0, draws, 256):
        indices = rng.integers(len(y), size=(min(256, draws - start), len(y)))
        rmsle = np.sqrt(squared_log[indices].mean(axis=1))
        mae = absolute[indices].mean(axis=1)
        if np.any(rmsle[:, 0] == 0):
            raise ValueError("Relative RMSLE gain is undefined for a zero-error reference draw")
        relative.extend(100 * (1 - rmsle[:, 1] / rmsle[:, 0]))
        dollars.extend(mae[:, 0] - mae[:, 1])
    return {
        "rmsle_gain_pct_ci95": np.percentile(relative, [2.5, 97.5]).tolist(),
        "mae_reduction_usd_ci95": np.percentile(dollars, [2.5, 97.5]).tolist(),
        "draws": draws, "seed": seed, "unit": "house", "conditional_on_fixed_predictions": True,
    }


def published_v2_config(published):
    """Read the revised five-fold V2 winner without using holdout scores."""
    from src.ames_data import INPUT_SCHEMA
    if published.get("input_schema") != INPUT_SCHEMA:
        raise ValueError("V2 result uses an older input schema; rerun the V2 benchmark")
    if published["folds"] != 5 or published["trials"] != 20 or published["dataset_sha256"] != DATA_SHA:
        raise ValueError("Expected the published five-fold, 20-trial V2 experiment")
    best = min(published["candidates"], key=lambda row: row["cv_rmsle"])
    if any(best[key] != published["selected"][key] for key in ("model", "engineered")):
        raise ValueError("V2 configuration differs from its published selection")
    return {key: best[key] for key in ("model", "params", "engineered")}


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
    if any(obj.get("input_schema") != INPUT_SCHEMA for obj in (comparison, published, summary, frozen)):
        raise ValueError("All results must use the same Ames CSV interpretation")
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

    # Compare the five-fold V2 rerun with the other releases. V3's nested
    # three-fold V2 comparator is a separately tuned procedure.
    v2 = published_v2_config(published)
    houses["v2"], v2_train_rows, _ = v2_predict(v2, training, y, X)
    recipe = frozen["recipe"]
    recipe_column = "cand_" + frozen["best_new"] if recipe == "best_new" else recipe
    np.testing.assert_array_equal(final.v3, final[recipe_column])
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
            "paired_interval": paired_intervals(houses.SalePrice, houses[old], houses[new]),
        }
    audit = {
        "dataset_sha256": DATA_SHA, "input_schema": INPUT_SCHEMA, "ordered_split_sha256": ids, "seed": 42,
        "development_rows": len(dev), "holdout_rows": len(test),
        "effective_training_rows": {"v1": v1_train_rows, "v2": v2_train_rows, "v3": len(dev)},
        "models": {"v1": v1, "v2": v2["model"],
                   "v3": frozen["best_new"] if recipe == "best_new" else recipe},
        "v2_published_params": v2["params"], "metrics": scores, "differences": differences,
        "v1_historical_source_commit": comparison["v1_source_commit"],
        "v1_revision": comparison["v1_revision"],
        "v2_historical_source_commit": "7473f585685b15cf886daa32041f8f293e887d1e",
        "v2_rerun_selection": "ames-v2/results.json (5 folds, 20 trials per family)",
        "v3_protocol": "protocol.md", "v3_frozen_recipe": recipe,
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
