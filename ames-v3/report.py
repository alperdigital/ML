"""Compare locked prediction procedures on the same homes and splits."""
import itertools

import numpy as np
import pandas as pd


BOOTSTRAP_SEED = 2603
BOOTSTRAP_DRAWS = 5000
MEMBERS = ("v1_xgb", "v2_cat", "best_new", "equal_blend", "weighted_blend")
REQUIRED = {"Id", "SalePrice", "YrSold", "Neighborhood", "repeat", "fold", "v2", "v3"}


def _errors(frame, name):
    actual = frame["SalePrice"].to_numpy(dtype=float)
    pred = frame[name].to_numpy(dtype=float)
    if not np.isfinite(actual).all() or not np.isfinite(pred).all():
        raise ValueError(f"Nonfinite observed price or {name} prediction")
    if np.any(actual < 0) or np.any(pred < 0):
        raise ValueError(f"Negative observed price or {name} prediction")
    return pred - actual, np.log1p(pred) - np.log1p(actual)


def _metrics(frame, name):
    dollar_error, log_error = _errors(frame, name)
    actual = frame["SalePrice"].to_numpy(dtype=float)
    total_ss = np.sum((actual - actual.mean()) ** 2)
    return {
        "rmsle": float(np.sqrt(np.mean(log_error**2))),
        "mae": float(np.mean(np.abs(dollar_error))),
        "r2": float(1 - np.sum(dollar_error**2) / total_ss) if total_ss > 0 else None,
    }


def _delta(v2, v3):
    if v2["rmsle"] == 0 or v2["mae"] == 0:
        raise ValueError("Relative performance requires nonzero reference error.")
    return {
        "rmsle_gain_pct": 100 * (1 - v3["rmsle"] / v2["rmsle"]),
        "mae_change_pct": 100 * (v3["mae"] / v2["mae"] - 1),
        "mae_change_usd": v3["mae"] - v2["mae"],
    }


def _comparison(frame):
    v2, v3 = _metrics(frame, "v2"), _metrics(frame, "v3")
    return {
        "properties": int(frame["Id"].nunique()), "predictions": len(frame),
        "v2": v2, "v3": v3, **_delta(v2, v3),
    }


def _check(frame, name):
    missing = REQUIRED.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} lacks required columns: {sorted(missing)}")
    if frame.empty or frame["Id"].isna().any() or frame.duplicated(["Id", "repeat"]).any():
        raise ValueError(f"{name} requires one prediction per property in each repeat")
    identity = ["SalePrice", "YrSold", "Neighborhood"]
    if frame.groupby("Id")[identity].nunique(dropna=False).gt(1).any().any():
        raise ValueError(f"{name} contains inconsistent sale details for an Id")


def _bootstrap(outer):
    # The resampling unit is one house with both repeats. The interval describes
    # these fixed out-of-fold predictions and excludes training-set variability.
    if set(outer["repeat"]) != {0, 1} or not outer.groupby("Id").size().eq(2).all():
        raise ValueError("Every outer property must have exactly two paired repeats")
    components = {}
    for model in ("v2", "v3"):
        dollars, log = _errors(outer, model)
        components[model + "_sq"] = log**2
        components[model + "_abs"] = np.abs(dollars)
    by_home = pd.DataFrame(components).groupby(outer["Id"].to_numpy(), sort=True).mean().to_numpy()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rmsle_gains, mae_changes = [], []
    for start in range(0, BOOTSTRAP_DRAWS, 128):
        batch = min(128, BOOTSTRAP_DRAWS - start)
        idx = rng.integers(len(by_home), size=(batch, len(by_home)))
        v2_sq, v2_abs, v3_sq, v3_abs = by_home[idx].mean(axis=1).T
        if np.any(v2_sq == 0) or np.any(v2_abs == 0):
            raise ValueError("Zero reference error in a bootstrap replicate")
        rmsle_gains.extend(100 * (1 - np.sqrt(v3_sq / v2_sq)))
        mae_changes.extend(100 * (v3_abs / v2_abs - 1))
    return {
        "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
        "unit": "Id with both paired repeats", "conditional_on_fixed_predictions": True,
        "rmsle_gain_pct_ci95": np.percentile(rmsle_gains, [2.5, 97.5]).tolist(),
        "mae_change_pct_ci95": np.percentile(mae_changes, [2.5, 97.5]).tolist(),
    }


def _by(frame, labels, min_n=1):
    groups = {}
    for value, group in frame.groupby(labels, observed=True, sort=True, dropna=False):
        if group["Id"].nunique() >= min_n:
            groups[str(value)] = _comparison(group)
    return groups


def _complementarity(frame):
    keys = [k for k in ("v1_xgb", "v2_cat", "best_new") if k in frame]
    errors = {key: _errors(frame, key)[1] for key in keys}
    pairs = {}
    for a, b in itertools.combinations(keys, 2):
        ea, eb = errors[a], errors[b]
        pairs[f"{a} vs {b}"] = {
            "log_residual_pearson_r": float(np.corrcoef(ea, eb)[0, 1])
            if ea.std() > 0 and eb.std() > 0 else None,
            "opposite_error_sign_fraction": float(np.mean(ea * eb < 0)),
            "first_closer_abs_log_error_fraction": float(np.mean(np.abs(ea) < np.abs(eb))),
        }
    return pairs


def _partition(frame, price_cutoffs):
    for name in ("v2", "v3"):
        _errors(frame, name)
    extra_models = [c for c in frame if c.startswith("cand_") or c in MEMBERS]
    # An unavailable optional model may have an entirely absent/empty column.
    observed = [name for name in extra_models if frame[name].notna().any()]
    price_groups = np.searchsorted(price_cutoffs, frame["SalePrice"], side="left")
    return {
        **_comparison(frame),
        "model_metrics": {name: _metrics(frame, name) for name in ["v2", "v3", *observed]},
        "by_repeat": _by(frame, "repeat"),
        "by_year": _by(frame, "YrSold"),
        "by_neighborhood_min20": _by(frame, "Neighborhood", min_n=20),
        "by_price_quartile": _by(frame, np.array(["Q1", "Q2", "Q3", "Q4"])[price_groups]),
        "residual_complementarity": _complementarity(frame),
    }


def summarize(outer, temporal, holdout):
    """Primary estimates pool all outer predictions, equally weighting each house."""
    for name, frame in (("outer", outer), ("temporal", temporal), ("holdout", holdout)):
        _check(frame, name)
    if set(outer["Id"]) & set(holdout["Id"]):
        raise ValueError("Development and historical holdout property IDs overlap")
    sale_columns = ["SalePrice", "YrSold", "Neighborhood"]
    sales = outer.drop_duplicates("Id").set_index("Id")[sale_columns]
    future = temporal.set_index("Id")[sale_columns]
    expected = sales.reindex(future.index)
    same = expected.eq(future) | (expected.isna() & future.isna())
    if not future.index.isin(sales.index).all() or not same.all().all():
        raise ValueError("Temporal houses and sale details must match the development sales")
    cutoff = outer.drop_duplicates("Id")["SalePrice"].quantile([0.25, 0.50, 0.75]).to_numpy()
    primary = _partition(outer, cutoff)
    primary["paired_bootstrap"] = _bootstrap(outer)
    primary["by_fold"] = _by(outer, ["repeat", "fold"])
    return {
        "outer": primary, "temporal": _partition(temporal, cutoff),
        "holdout": _partition(holdout, cutoff),
        "price_quartile_boundaries_usd": cutoff.tolist(),
        "subgroup_rule": "Neighborhood rows require >=20 distinct houses in each partition. "
                         "Year and price cells are descriptive and include counts.",
        "promote_v3": bool(primary["rmsle_gain_pct"] >= 2 and primary["mae_change_pct"] < 1),
        "promotion_rule": "Pooled outer RMSLE relative gain >=2% and MAE relative increase <1%",
    }
