"""Fit the original Ames V2 search and V1 XGBoost on caller-supplied partitions."""
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import optuna

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ames-v2"))
import benchmark
import compare_v1_v2 as legacy


def _without_id(frame):
    return frame.drop(columns=["Id"], errors="ignore")


class V2Predictor:
    def __init__(self, estimator):
        self.estimator = estimator

    def predict(self, frame):
        return np.maximum(self.estimator.predict(_without_id(frame)), 0)


def fit_v2(config, X, y):
    estimator = benchmark.estimator(config["model"], config["params"], config["engineered"])
    return V2Predictor(estimator.fit(_without_id(X), y))


def _prepare_folds(X, y, splits, engineered):
    """Preprocessing is invariant to model parameters; fit it once per training fold."""
    folds = []
    for train, valid in splits:
        preprocess = benchmark.estimator("ridge", engineered=engineered).regressor[:-1]
        train_features = preprocess.fit_transform(X.iloc[train], y.iloc[train])
        valid_features = preprocess.transform(X.iloc[valid])
        folds.append((train_features, valid_features, np.log1p(y.iloc[train].to_numpy()),
                      valid, y.iloc[valid].to_numpy()))
    return folds


def _evaluate(name, params, prepared, n_rows):
    oof = np.full(n_rows, np.nan)
    scores = []
    for Xtr, Xval, log_y, positions, yval in prepared:
        model = benchmark.estimator(name, params).regressor.named_steps["model"]
        model.fit(Xtr, log_y)
        prices = np.maximum(np.expm1(model.predict(Xval)), 0)
        oof[positions] = prices
        scores.append(benchmark.rmsle(yval, prices))
    if not np.isfinite(oof).all():
        raise ValueError("Inner splits must give every training row a finite out-of-fold prediction")
    return oof, scores


def select_v2(X, y, splits, trials=20):
    """Original V2 objective, TPE search, engineered-feature ablation and tie order.

    All indices are positional relative to X. Every inner-validation row is seen
    exactly once. Raw-price OOF vectors are returned for downstream blending.
    """
    splits = list(splits)
    assignments = np.concatenate([valid for _, valid in splits])
    if not np.array_equal(np.sort(assignments), np.arange(len(X))):
        raise ValueError("Inner validation partitions must cover the training set exactly once")
    if any(len(np.intersect1d(train, valid)) for train, valid in splits):
        raise ValueError("Inner training and validation rows overlap")
    start = perf_counter()
    X = _without_id(X)
    folds = {flag: _prepare_folds(X, y, splits, flag) for flag in (True, False)}
    prep_seconds = perf_counter() - start
    oof, configs, audits, model_times, scores = {}, {}, {}, {}, {}
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for name in ("dummy", "ridge", "xgboost", "lightgbm", "catboost"):
        model_start = perf_counter()
        params = {}
        # Cache trial OOF vectors rather than re-fitting the best trial.
        by_trial = {}
        history = []
        if trials and name != "dummy":
            def objective(trial):
                trial_params = benchmark.search_space(trial, name)
                predicted, per_fold = _evaluate(name, trial_params, folds[True], len(X))
                by_trial[trial.number] = (predicted, per_fold)
                return float(np.mean(per_fold))

            study = optuna.create_study(direction="minimize",
                                        sampler=optuna.samplers.TPESampler(seed=benchmark.SEED))
            study.optimize(objective, n_trials=trials)
            params = study.best_params
            history = [{"trial": t.number, "inner_rmsle": t.value, "params": t.params} for t in study.trials]
        for engineered in ([True] if name == "dummy" else [True, False]):
            key = "{}_{}".format(name, "engineered" if engineered else "raw")
            if engineered and by_trial:
                predicted, per_fold = by_trial[study.best_trial.number]
            else:
                predicted, per_fold = _evaluate(name, params, folds[engineered], len(X))
            oof[key] = predicted
            configs[key] = {"model": name, "params": params, "engineered": engineered}
            scores[key] = {"rmsle": float(np.mean(per_fold)), "folds": per_fold}
        audits[name] = history
        model_times[name] = perf_counter() - model_start

    winner = min(configs, key=lambda key: scores[key]["rmsle"])
    catboost = min((k for k in configs if configs[k]["model"] == "catboost"),
                   key=lambda k: scores[k]["rmsle"])
    return {"winner": winner, "catboost": catboost, "configs": configs, "oof": oof,
            "audit": {"trials_per_model": trials, "inner_folds": len(splits),
                      "scores": scores, "trials": audits, "model_seconds": model_times,
                      "preprocessing_seconds": prep_seconds, "seconds": perf_counter() - start}}


def init_v1(root=None):
    source_root = Path(root or os.environ.get("AMES_V1_ROOT", ROOT))
    if not (source_root / "src" / "data_preprocessing.py").exists():
        raise FileNotFoundError("Set AMES_V1_ROOT to a checkout containing the V1 src directory")
    legacy.configure_v1_imports(source_root)


class V1Predictor(legacy.TreeFeatures):
    """XGBoost with the shared retrospective V1 preprocessor."""

    def predict(self, frame):
        if getattr(self, "feature_schema_version", None) != 3:
            raise ValueError("Stored V1 member predates the corrected features. Refit without --bundle.")
        return np.maximum(self.model.predict(self.transform(frame)), 0)


def fit_v1(X, y):
    if "_ames_v1" not in sys.modules:
        init_v1()
    from _ames_v1.models.xgboost_model import XGBoostModel
    predictor = V1Predictor()
    features, target = predictor.fit_transform(X, y)
    predictor.feature_schema_version = 3
    model = XGBoostModel(params=dict(XGBoostModel().params, n_jobs=1))
    model.train(features, target, verbose=False)
    predictor.model = model.model
    return predictor


def v1_oof(X, y, splits):
    predictions = np.full(len(X), np.nan)
    for train, valid in splits:
        predictions[valid] = fit_v1(X.iloc[train], y.iloc[train]).predict(X.iloc[valid])
    if not np.isfinite(predictions).all():
        raise ValueError("Incomplete V1 OOF predictions")
    return predictions
