"""Compare the corrected retrospective V1 tree baseline and the released V2 on identical sales."""
import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from types import ModuleType
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tree_features import TreeFeatures, fill_validation

from benchmark import SEED, estimator, rmsle, package_versions

V1_SOURCE = "a1cf99b304c604125019a10b587613a5d3b1ee5c"
V1_MODELS = ("xgboost", "lightgbm")


def configure_v1_imports(root):
    """Load original files without V1 package initializers importing optional KAN."""
    source = Path(root).resolve() / "src"
    for package, path in (("_ames_v1", source), ("_ames_v1.models", source / "models"),
                          ("_ames_v1.utils", source / "utils")):
        module = ModuleType(package)
        module.__path__ = [str(path)]
        sys.modules[package] = module


def v1_predict(name, Xtr, ytr, Xval):
    from _ames_v1.models.xgboost_model import XGBoostModel
    from _ames_v1.models.lightgbm_model import LightGBMModel

    features = TreeFeatures()
    tr, target = features.fit_transform(Xtr, ytr)
    val = features.transform(Xval)
    # V1 examples retain Id and train the tree regressors on raw dollars.
    wrapper = XGBoostModel if name == "xgboost" else LightGBMModel
    base = wrapper()
    params = dict(base.params, n_jobs=1)
    if name == "lightgbm":
        params["verbosity"] = -1
    model = wrapper(params=params).train(tr, target, verbose=False)
    pred = model.predict(val)
    return np.maximum(pred, 0), len(tr), params


def v2_predict(candidate, Xtr, ytr, Xval):
    model = estimator(candidate["model"], candidate["params"], candidate["engineered"])
    tr, val = Xtr.drop(columns=["Id"], errors="ignore"), Xval.drop(columns=["Id"], errors="ignore")
    return np.maximum(model.fit(tr, ytr).predict(val), 0), len(tr), candidate["params"]


def metrics(y, pred):
    assert len(y) == len(pred) and np.isfinite(pred).all()
    return {"rmsle": rmsle(y, pred), "r2": float(r2_score(y, pred)),
            "mae": float(mean_absolute_error(y, pred))}


def split_fingerprint(index):
    return hashlib.sha256(np.asarray(index, dtype="<i8").tobytes()).hexdigest()


def run(data, published):
    df = pd.read_csv(data)
    y = df.pop("SalePrice")
    Xd, Xh, yd, yh = train_test_split(df, y, test_size=0.2, random_state=SEED)
    cv = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(Xd))
    ref = json.loads(Path(published).read_text())
    dataset_hash = hashlib.sha256(Path(data).read_bytes()).hexdigest()
    assert dataset_hash == ref["dataset_sha256"] and ref["seed"] == SEED
    assert ref["folds"] == 5 and ref["trials"] == 20
    assert (len(df), len(Xd), len(Xh)) == (ref["rows"], ref["development_rows"], ref["holdout_rows"])

    rows = []
    selected = {}
    for name in V1_MODELS:
        scores, counts = [], []
        for train, val in cv:
            pred, n, params = v1_predict(name, Xd.iloc[train], yd.iloc[train], Xd.iloc[val])
            scores.append(metrics(yd.iloc[val], pred)["rmsle"])
            counts.append(n)
        rows.append(dict(version="v1", model=name, engineered=True, params=params,
                         cv_rmsle=float(np.mean(scores)), cv_std=float(np.std(scores)),
                         fold_rmsle=scores, fold_train_rows=counts))
    selected["v1"] = min(rows, key=lambda r: r["cv_rmsle"])["model"]

    # Re-evaluate the CV-best already-tuned configuration of every V2 model.
    for name in ("dummy", "ridge", "xgboost", "lightgbm", "catboost"):
        candidate = min((r for r in ref["candidates"] if r["model"] == name), key=lambda r:r["cv_rmsle"])
        scores, counts = [], []
        for train, val in cv:
            pred, n, params = v2_predict(candidate, Xd.iloc[train], yd.iloc[train], Xd.iloc[val])
            scores.append(metrics(yd.iloc[val], pred)["rmsle"])
            counts.append(n)
        np.testing.assert_allclose(scores, candidate["folds"], atol=1e-8, rtol=1e-7)
        rows.append(dict(version="v2", model=name, engineered=candidate["engineered"], params=params,
                         cv_rmsle=float(np.mean(scores)), cv_std=float(np.std(scores)),
                         fold_rmsle=scores, fold_train_rows=counts))
    selected["v2"] = min((r for r in rows if r["version"] == "v2"), key=lambda r:r["cv_rmsle"])["model"]
    assert selected["v2"] == ref["selected"]["model"]

    # Record all candidates' holdout metrics for descriptive comparison, not selection.
    for row in rows:
        if row["version"] == "v1":
            pred, n, _ = v1_predict(row["model"], Xd, yd, Xh)
        else:
            pred, n, _ = v2_predict(row, Xd, yd, Xh)
        row.update(holdout=metrics(yh, pred), development_train_rows=n)
        print(row["version"], row["model"], row["cv_rmsle"], row["holdout"])
    winner = next(r for r in rows if r["version"] == "v2" and r["model"] == selected["v2"])
    for metric, value in winner["holdout"].items():
        np.testing.assert_allclose(value, ref["selected"]["holdout_"+metric], atol=1e-7, rtol=1e-7)

    return dict(v1_source_commit=V1_SOURCE, v1_revision="v3-corrected-tree-features",
                versions=dict(package_versions(), yellowbrick=version("yellowbrick")), seed=SEED,
                dataset_sha256=dataset_hash, development_rows=len(Xd), holdout_rows=len(Xh),
                development_index_sha256=split_fingerprint(Xd.index), holdout_index_sha256=split_fingerprint(Xh.index),
                cv_folds=5, v2_trials_per_model=ref["trials"], selected_by_cv=selected,
                v1_scope="XGBoost and LightGBM from examples/compare_models.py; KAN is not in that executable comparison.",
                candidates=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/train.csv")
    parser.add_argument("--published", default="ames-v2/results.json")
    parser.add_argument("--v1-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", default="ames-v2/v1-v2-comparison.json")
    args = parser.parse_args()
    configure_v1_imports(args.v1_root)
    result = run(args.data, args.published)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
