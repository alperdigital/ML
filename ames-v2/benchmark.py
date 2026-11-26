"""Reproducible Ames Housing benchmark, with a sealed final holdout."""
import argparse
import hashlib
import json
import platform
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42

def rmsle(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    if (y < 0).any():
        raise ValueError("Negative target")
    return float(np.sqrt(np.mean((np.log1p(y) - np.log1p(np.maximum(pred, 0))) ** 2)))

class Features(BaseEstimator, TransformerMixin):
    def __init__(self, engineered=True):
        self.engineered = engineered

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.engineered:
            for name, cols in {
                "TotalSF": ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"],
                "TotalBath": ["FullBath", "BsmtFullBath"],
            }.items():
                if set(cols).issubset(X.columns):
                    X[name] = X[cols].sum(axis=1, min_count=len(cols))
            for name, year in [("HouseAge", "YearBuilt"), ("RemodelAge", "YearRemodAdd")]:
                if {"YrSold", year}.issubset(X.columns):
                    X[name] = X["YrSold"] - X[year]
        return X

def estimator(name, params=None, engineered=True):
    params = params or {}
    if name == "ridge":
        model = Ridge(**params)
    elif name == "dummy":
        model = DummyRegressor(strategy="mean")
    elif name == "xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor(random_state=SEED, n_jobs=1, **params)
    elif name == "lightgbm":
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(random_state=SEED, n_jobs=1, verbosity=-1, **params)
    elif name == "catboost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(random_seed=SEED, thread_count=1, verbose=False,
                                  allow_writing_files=False, **params)
    else:
        raise ValueError(name)
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True,
                                                  keep_empty_features=True)),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    prep = ColumnTransformer([("num", numeric, make_column_selector(dtype_include=np.number)),
                              ("cat", categorical, make_column_selector(dtype_exclude=np.number))])
    return TransformedTargetRegressor(
        regressor=Pipeline([("features", Features(engineered)), ("prep", prep), ("model", model)]),
        func=np.log1p, inverse_func=np.expm1)

def search_space(trial, name):
    if name == "ridge":
        return {"alpha": trial.suggest_float("alpha", 0.01, 100, log=True)}
    rounds = "iterations" if name == "catboost" else "n_estimators"
    depth = "depth" if name == "catboost" else "max_depth"
    return {rounds: trial.suggest_int(rounds, 200, 1000),
            depth: trial.suggest_int(depth, 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True)}

def cv_scores(model, X, y, cv):
    return -cross_val_score(model, X, y, cv=cv, scoring=make_scorer(rmsle, greater_is_better=False),
                            error_score="raise")

def package_versions():
    result = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost", "optuna"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            pass
    return result

def run(path, models, trials=20, folds=5):
    df = pd.read_csv(path)
    y = df.pop("SalePrice")
    if y.isna().any() or (y < 0).any() or not np.isfinite(y).all():
        raise ValueError("SalePrice must contain finite nonnegative labels")
    X = df.drop(columns=["Id"], errors="ignore")
    Xd, Xh, yd, yh = train_test_split(X, y, test_size=0.2, random_state=SEED)
    cv = KFold(n_splits=folds, shuffle=True, random_state=SEED)
    rows = []
    for name in ["dummy", *models]:
        params = {}
        if trials and name != "dummy":
            import optuna
            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
            study.optimize(lambda t: cv_scores(estimator(name, search_space(t, name)), Xd, yd, cv).mean(),
                           n_trials=trials)
            params = study.best_params
        for engineered in ([True] if name == "dummy" else [True, False]):
            scores = cv_scores(estimator(name, params, engineered), Xd, yd, cv)
            rows.append(dict(model=name, engineered=engineered, params=params,
                             cv_rmsle=float(scores.mean()), cv_std=float(scores.std()), folds=scores.tolist()))
    # Select once on development CV; the holdout never enters selection or tuning.
    best = min(rows, key=lambda row: row["cv_rmsle"])
    fitted = estimator(best["model"], best["params"], best["engineered"]).fit(Xd, yd)
    pred = np.maximum(fitted.predict(Xh), 0)
    return dict(seed=SEED, versions=package_versions(), dataset_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                rows=len(X), development_rows=len(Xd), holdout_rows=len(Xh),
                trials=trials, folds=folds, candidates=rows,
                selected=dict(model=best["model"], engineered=best["engineered"],
                              holdout_rmsle=rmsle(yh, pred), holdout_r2=float(r2_score(yh, pred)),
                              holdout_mae=float(mean_absolute_error(yh, pred))))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/train.csv")
    p.add_argument("--models", nargs="+", choices=["ridge", "xgboost", "lightgbm", "catboost"],
                   default=["ridge", "xgboost", "lightgbm", "catboost"])
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--output", default="ames-v2/results.json")
    args = p.parse_args()
    result = run(args.data, args.models, args.trials, args.folds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["selected"], indent=2))
