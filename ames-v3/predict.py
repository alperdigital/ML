"""Refit the frozen recipe on development houses and price one Ames input row."""
import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype, pandas_dtype

from baselines import fit_v1, fit_v2
from design import load_data, partitions
from models_v3 import blend_predict, fit_model


def selected_recipe(audit, version):
    if version == "v2":
        return "v2"
    if version == "v3":
        return audit["recipe"]
    raise ValueError("Version must resolve to v2 or v3")


def fit_recipe(training, audit, version):
    """Fit only the members used by the selected recipe; hyperparameters are frozen."""
    y = training["SalePrice"]
    X = training.drop(columns=["SalePrice"])
    recipe = selected_recipe(audit, version)
    needed = audit["member_order"] if recipe in ("equal_blend", "weighted_blend") else [recipe]
    models = {}
    for key in needed:
        if key in ("v2", "v2_cat"):
            models[key] = fit_v2(audit["v2_configs"][audit[key]], X, y)
        elif key == "v1_xgb":
            models[key] = fit_v1(X, y)
        elif key == "best_new":
            models[key] = fit_model(audit["specs"][audit["best_new"]], X, y)
        else:
            raise ValueError(f"Unsupported frozen member {key}")
    return {"models": models, "audit": audit, "columns": X.columns.tolist(),
            "dtypes": X.dtypes.astype(str).to_dict(), "fit_version": version}


def prepare_row(frame, columns, dtypes, house_id=None):
    """Match training column order and reject malformed values before prediction."""
    if not frame.columns.is_unique:
        raise ValueError("Duplicate input column")
    missing = set(columns).difference(frame.columns)
    extra = set(frame.columns).difference(columns)
    if missing or extra:
        raise ValueError(f"Input schema mismatch. Missing {sorted(missing)}, unexpected {sorted(extra)}")
    if set(dtypes) != set(columns):
        raise ValueError("Training schema must provide one dtype per input column")
    if house_id is not None:
        ids = pd.to_numeric(frame["Id"], errors="raise")
        frame = frame.loc[ids == house_id]
    if len(frame) != 1:
        raise ValueError("Select exactly one input house by Id or use a one-row input file")
    frame = frame.loc[:, columns].copy()
    for col in columns:
        dtype = pandas_dtype(dtypes[col])
        if is_numeric_dtype(dtype):
            try:
                value = pd.to_numeric(frame[col], errors="raise")
            except ValueError as exc:
                raise ValueError(f"Expected numeric input for {col}") from exc
            observed = value.dropna().to_numpy(dtype=float)
            if not np.isfinite(observed).all():
                raise ValueError(f"Infinite numeric input in {col}")
            if is_integer_dtype(dtype):
                if np.any(observed != np.floor(observed)):
                    raise ValueError(f"Expected whole-number input for {col}")
                if value.isna().any():
                    frame[col] = value.astype(float)
                    continue
            frame[col] = value.astype(dtype)
        else:
            # Preserve missing categorical values as NaN for scikit-learn imputers.
            frame[col] = frame[col].astype("object").map(lambda x: np.nan if pd.isna(x) else str(x))
    if frame.Id.isna().any():
        raise ValueError("The input house needs a numeric Id")
    return frame


def price(bundle, row, version):
    recipe = selected_recipe(bundle["audit"], version)
    models = bundle["models"]
    if recipe in ("equal_blend", "weighted_blend"):
        order = bundle["audit"]["member_order"]
        members = np.column_stack([models[name].predict(row) for name in order])
        weights = bundle["audit"]["weights"] if recipe == "weighted_blend" else np.full(len(order), 1/len(order))
        pred = blend_predict(members, weights)
    else:
        pred = np.asarray(models[recipe].predict(row), dtype=float)
    if pred.shape != (1,) or not np.isfinite(pred[0]) or pred[0] < 0:
        raise ValueError("Expected one finite nonnegative predicted price")
    return {"Id": int(row.Id.iloc[0]), "version": version, "recipe": recipe,
            "predicted_sale_price_usd": round(float(pred[0]), 2)}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Ames-format CSV (all 80 non-target columns)")
    parser.add_argument("--id", type=int, help="Id of exactly one input row")
    parser.add_argument("--version", choices=["recommended", "v2", "v3"], default="recommended")
    parser.add_argument("--results", type=Path, default=root / "ames-v3/results")
    parser.add_argument("--data", type=Path, default=root / "data/train.csv")
    parser.add_argument("--bundle", type=Path, help="Load a trusted local joblib model bundle")
    parser.add_argument("--save-bundle", type=Path, help="Store a locally refitted recipe for later use")
    args = parser.parse_args()
    version = args.version
    if version == "recommended":
        summary = json.loads((args.results / "summary.json").read_text())
        version = "v3" if summary["promote_v3"] else "v2"
    if args.bundle:
        # joblib files contain executable Python objects. Use your own local bundle.
        bundle = joblib.load(args.bundle)
        if bundle.get("fit_version", version) != version:
            raise ValueError("This lightweight bundle was fitted for a different version")
    else:
        audit = json.loads((args.results / "frozen-final.json").read_text())
        df = load_data(args.data)
        dev = partitions(df)["final"][0]
        with redirect_stdout(sys.stderr):
            bundle = fit_recipe(df.iloc[dev], audit, version)
        if args.save_bundle:
            from run import save_bundle
            args.save_bundle.parent.mkdir(parents=True, exist_ok=True)
            save_bundle(args.save_bundle, bundle)
    if "dtypes" not in bundle:
        raise ValueError("Bundle has no input dtypes. Refit from the frozen result files")
    # Parse as text so a numeric-looking category keeps its literal spelling.
    row = prepare_row(pd.read_csv(args.input, dtype=str), bundle["columns"], bundle["dtypes"], args.id)
    with redirect_stdout(sys.stderr):
        prediction = price(bundle, row, version)
    print(json.dumps(prediction, allow_nan=False))


if __name__ == "__main__":
    main()
