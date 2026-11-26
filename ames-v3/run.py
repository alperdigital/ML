"""Run the corrected nested Ames comparison and freeze the development fit."""
import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from base_stage import build
from baselines import fit_v1, fit_v2, v1_oof
from design import DATA_SHA, inner_splits, load_data, partitions
from models_v3 import blend_predict, blend_weights, candidates, fit_model
from recipe_cache import cached_oof

MEMBERS = ["v1_xgb", "v2_cat", "best_new"]
REPORT_COLS = ["Id", "SalePrice", "YrSold", "Neighborhood"]
BLEND_SPLIT_SEED = 42


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def mean_fold_rmsle(y, predictions, splits):
    predictions = np.asarray(predictions)
    if predictions.shape != (len(y),) or not np.isfinite(predictions).all() or (predictions < 0).any():
        raise ValueError("Invalid price prediction in model selection")
    errors = np.log1p(y.to_numpy()) - np.log1p(predictions)
    return float(np.mean([np.sqrt(np.mean(errors[val]**2)) for _, val in splits]))


def choose_model(cost):
    single = min(["v1_xgb", "v2_cat", "best_new", "v2"], key=cost.get)
    blend = min(["equal_blend", "weighted_blend"], key=cost.get)
    return single if cost[single] <= cost[blend] + .0005 else blend


def crossfit_blend(y, matrix, splits):
    """Score weight learning on other houses from the same base-model validation fold."""
    targets = np.asarray(y)
    prices = np.full(len(targets), np.nan)
    audit = []
    for fold, (_, validation) in enumerate(splits):
        two_halves = KFold(2, shuffle=True, random_state=BLEND_SPLIT_SEED)
        for half, (fit_pos, score_pos) in enumerate(two_halves.split(validation)):
            calibration, scoring = validation[fit_pos], validation[score_pos]
            weights = blend_weights(targets[calibration], matrix[calibration])
            prices[scoring] = blend_predict(matrix[scoring], weights)
            audit.append(dict(inner_fold=fold, half=half, fit_rows=len(calibration),
                              scored_rows=len(scoring), weights=weights.tolist()))
    if not np.isfinite(prices).all():
        raise ValueError("Every inner validation row requires a held-out blend prediction")
    return prices, audit


def select_and_fit(train, search, work=None):
    if train.Id.tolist() != search["train_ids"]:
        raise ValueError("Cached V2 run uses different training house IDs")
    start = time.monotonic()
    X = train.drop(columns="SalePrice")
    y = train.SalePrice
    splits = inner_splits(X)
    specs = candidates()
    cache_start = time.monotonic()
    cached = {k: cached_oof(X, y, splits, k, spec, work) for k, spec in specs.items()}
    actual_cache_seconds = time.monotonic() - cache_start
    new_oof = {k: value[0] for k,value in cached.items()}
    candidate_cost = {k: mean_fold_rmsle(y, values, splits) for k, values in new_oof.items()}
    best = min(candidate_cost, key=candidate_cost.get)
    winner, cat = search["winner"], search["catboost"]
    oof = {"v1_xgb": v1_oof(X, y, splits), "v2_cat": search["oof"][cat],
           "best_new": new_oof[best], "v2": search["oof"][winner]}
    matrix = np.column_stack([oof[key] for key in MEMBERS])
    oof["equal_blend"] = blend_predict(matrix, [1/3]*3)
    oof["weighted_blend"], weight_folds = crossfit_blend(y, matrix, splits)
    costs = {key: mean_fold_rmsle(y, values, splits) for key, values in oof.items()}
    recipe = choose_model(costs)
    # Outer and deployment predictions may use all of this partition's training OOF.
    weights = blend_weights(y, matrix)

    baseline_start = time.monotonic()
    fitted_v2 = fit_v2(search["configs"][winner], X, y)
    baseline_refit_seconds = time.monotonic() - baseline_start
    models = {"v2": fitted_v2, "v2_cat": fitted_v2 if cat == winner else fit_v2(search["configs"][cat], X, y),
              "v1_xgb": fit_v1(X, y)}
    models.update({"cand_"+k: fit_model(spec, X, y) for k, spec in specs.items()})
    models["best_new"] = models["cand_"+best]
    audit = {"v2": winner, "v2_cat": cat, "best_new": best, "recipe": recipe,
             "specs": specs, "weights": weights.tolist(), "member_order": MEMBERS,
             "weight_selection": {"seed": BLEND_SPLIT_SEED, "folds": weight_folds},
             "inner_rmsle": costs, "new_inner_rmsle": candidate_cost, "candidate_oof_seconds": {k:v[1] for k,v in cached.items()}, "v2_configs": search["configs"],
             "v2_search": search["audit"], "v2_search_seconds": search["elapsed_seconds"],
             "v2_refit_seconds": baseline_refit_seconds,
             "additional_and_refit_seconds": time.monotonic()-start-actual_cache_seconds+sum(v[1] for v in cached.values())}
    return models, audit


def predict_all(models, audit, X):
    predictions = {key: model.predict(X) for key, model in models.items() if key != "best_new"}
    predictions["best_new"] = predictions["cand_"+audit["best_new"]]
    matrix = np.column_stack([predictions[key] for key in MEMBERS])
    predictions["equal_blend"] = blend_predict(matrix, [1/3]*3)
    predictions["weighted_blend"] = blend_predict(matrix, audit["weights"])
    predictions["v3"] = predictions[audit["recipe"]]
    for key, values in predictions.items():
        if np.asarray(values).shape != (len(X),) or not np.isfinite(values).all() or np.any(np.asarray(values)<0):
            raise ValueError(f"Invalid predictions from {key}")
    return predictions


def evaluate(data, name, work, out):
    df = load_data(data)
    train_idx, val_idx, repeat, fold = partitions(df)[name]
    search = joblib.load(Path(work)/f"{name}.joblib")
    models, audit = select_and_fit(df.iloc[train_idx], search, work)
    audit.update(task=name, train_n=len(train_idx), eval_n=len(val_idx))
    dest = Path(out)
    pieces = Path(work)/"evaluations"
    pieces.mkdir(parents=True, exist_ok=True)
    if name == "final":
        # Freeze the recipe to disk before opening the historical comparison.
        write_json(dest/"frozen-final.json", {k:audit[k] for k in ("v2","v2_cat","best_new","recipe","weights","member_order","inner_rmsle","specs","v2_configs","weight_selection")})
        inputs = df.drop(columns="SalePrice")
        bundle = {"models":models, "audit":audit, "columns":inputs.columns.tolist(),
                  "dtypes": inputs.dtypes.astype(str).to_dict()}
        save_bundle(Path(work)/"development-fit.joblib", bundle)
    prediction_start = time.monotonic()
    prices = predict_all(models, audit, df.iloc[val_idx].drop(columns="SalePrice"))
    audit["batch_prediction_seconds_including_cache_hits"] = time.monotonic() - prediction_start
    result = df.iloc[val_idx][REPORT_COLS].copy()
    result["repeat"], result["fold"] = repeat, fold
    for key, values in prices.items():
        result[key] = values
    prediction_dir = dest if name in ("final", "temporal") else pieces
    result.to_csv(prediction_dir/f"{name}.csv", index=False)
    write_json(pieces/f"{name}.json", audit)
    print("Evaluated", name, flush=True)


def save_bundle(path, bundle):
    temp = path.with_suffix(".partial")
    joblib.dump(bundle, temp, compress=1)
    temp.replace(path)


def run(data, work, out, jobs):
    from report import summarize
    from models_v3 import diagnostic_counts
    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    df = load_data(data)
    tasks = partitions(df)
    build(data, work, jobs)
    names = [name for name in tasks if name != "final"]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        fs = [pool.submit(evaluate, data, name, work, out) for name in names]
        for future in as_completed(fs):
            future.result()
    pieces = Path(work)/"evaluations"
    outer = pd.concat([pd.read_csv(pieces/f"{name}.csv") for name in names if name != "temporal"],ignore_index=True)
    temporal = pd.read_csv(dest/"temporal.csv")
    # Score the frozen training procedure before reading its historical predictions.
    evaluate(data, "final", work, out)
    holdout = pd.read_csv(dest/"final.csv")
    summary = summarize(outer, temporal, holdout)
    dev_idx, hist_idx, _, _ = tasks["final"]
    summary["dataset_sha256"] = DATA_SHA
    summary["splits"] = {name:hashlib.sha256(np.asarray(idx,dtype="<i8").tobytes()).hexdigest()
                         for name,idx in [("development",dev_idx),("historical",hist_idx)]}
    summary["development_missingness"] = diagnostic_counts(df.iloc[dev_idx].drop(columns="SalePrice"))
    write_json(dest/"summary.json", summary)
    # Compact the full audit without discarding any predictions or trial records.
    outer.to_csv(dest/"outer-predictions.csv.gz", index=False, compression={"method":"gzip","mtime":0})
    audits = {name:json.loads((pieces/f"{name}.json").read_text()) for name in tasks}
    import gzip
    (dest/"split-audits.json.gz").write_bytes(gzip.compress(json.dumps(audits,allow_nan=False).encode(),mtime=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/train.csv")
    parser.add_argument("--work", default=".ames-v3-work")
    parser.add_argument("--out", default="ames-v3/results")
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    run(args.data, args.work, args.out, args.jobs)
