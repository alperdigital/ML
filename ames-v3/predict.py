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

ROOT = Path(__file__).resolve().parents[1]


def selected_recipe(audit, version):
    if version == "v2":
        return "v2_release"
    if version == "v2-nested":
        return "v2"
    if version == "v3":
        return audit["recipe"]
    raise ValueError("Version must resolve to v2, v2-nested or v3")


def recommended_version(summary):
    # The promotion experiment evaluated its locally retuned V2 challenger.
    return "v3" if summary["promote_v3"] else "v2-nested"


def fit_recipe(training, audit, version):
    """Fit only the members used by the selected recipe; hyperparameters are frozen."""
    y = training["SalePrice"]
    X = training.drop(columns=["SalePrice"])
    recipe = selected_recipe(audit, version)
    needed = audit["member_order"] if recipe in ("equal_blend", "weighted_blend") else [recipe]
    models = {}
    for key in needed:
        if key == "v2_release":
            from compare_releases import published_v2_config
            published = json.loads((ROOT / "ames-v2/results.json").read_text())
            config = published_v2_config(published)
            models[key] = fit_v2(config, X, y)
            audit = {"published_v2": config}
        elif key in ("v2", "v2_cat"):
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
    models = bundle["models"]
    fit_version = bundle.get("fit_version")
    # Bundles created before the release distinction called the nested fit "v2".
    if fit_version == "v2" and "v2_release" not in models:
        fit_version = "v2-nested"
    if version == "v2" and "v2_release" not in models:
        raise ValueError("Bundle lacks the published V2. Refit with --version v2 without --bundle.")
    if fit_version and fit_version != version:
        raise ValueError(f"Bundle fitted for {fit_version}, requested {version}")
    recipe = selected_recipe(bundle["audit"], version)
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
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Ames-format CSV (all 80 non-target columns)")
    source.add_argument("--holdout-id", type=int, help="Price one of the 292 historical holdout sales by Id")
    parser.add_argument("--id", type=int, help="Select one Id in --input")
    parser.add_argument("--version", choices=["recommended", "v2", "v2-nested", "v3"], default="recommended",
                        help="v2: published portfolio model; v2-nested: V3 experiment's V2 challenger")
    parser.add_argument("--results", type=Path, default=ROOT / "ames-v3/results")
    parser.add_argument("--data", type=Path, default=ROOT / "data/train.csv")
    parser.add_argument("--bundle", type=Path, help="Load a trusted local joblib model bundle")
    parser.add_argument("--save-bundle", type=Path, help="Store a locally refitted recipe for later use")
    args = parser.parse_args()
    if args.id is not None and args.holdout_id is not None:
        parser.error("--id selects a row from --input; use --holdout-id on its own")
    if args.bundle and args.save_bundle:
        parser.error("Choose --bundle to reuse a fit or --save-bundle to save a new fit")
    version = args.version
    if version == "recommended":
        summary = json.loads((args.results / "summary.json").read_text())
        version = recommended_version(summary)
    if args.holdout_id is not None or not args.bundle:
        df = load_data(args.data)
        dev, holdout, _, _ = partitions(df)["final"]
    if args.holdout_id is not None:
        inputs = df.iloc[holdout].drop(columns="SalePrice")
        if not inputs.Id.eq(args.holdout_id).any():
            parser.error(f"House {args.holdout_id} is outside the 292 historical holdout sales")
        house_id = args.holdout_id
    else:
        # Keep literal spellings of numeric-looking categories from user files.
        inputs = pd.read_csv(args.input, dtype=str)
        house_id = args.id
    if args.bundle:
        # joblib files contain executable Python objects. Use your own local bundle.
        bundle = joblib.load(args.bundle)
    else:
        audit = {} if version == "v2" else json.loads((args.results / "frozen-final.json").read_text())
        with redirect_stdout(sys.stderr):
            bundle = fit_recipe(df.iloc[dev], audit, version)
        if args.save_bundle:
            from run import save_bundle
            args.save_bundle.parent.mkdir(parents=True, exist_ok=True)
            save_bundle(args.save_bundle, bundle)
    if "dtypes" not in bundle:
        raise ValueError("Bundle has no input dtypes. Refit from the frozen result files")
    row = prepare_row(inputs, bundle["columns"], bundle["dtypes"], house_id)
    with redirect_stdout(sys.stderr):
        prediction = price(bundle, row, version)
    print(json.dumps(prediction, allow_nan=False))


if __name__ == "__main__":
    main()
