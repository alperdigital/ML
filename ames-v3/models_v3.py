"""Predeclared fold-local Ames recipes; all public predictions are in dollars."""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MISSING = "__UNKNOWN__"
ABSENT = "__ABSENT__"
GARAGE = ("GarageType", "GarageFinish", "GarageQual", "GarageCond")
BASEMENT = ("BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2")


def candidates():
    """The four structural/native variants form a 2x2 frontage ablation."""
    base = dict(model="catboost", native=True, structural=False,
                neighborhood=False, indicator=False)
    structural = dict(base, structural=True)
    full = dict(structural, neighborhood=True, indicator=True)
    return {
        "onehot_plain": dict(base, native=False),
        "native_plain": base,
        "native_structural": structural,
        "native_neighborhood": dict(structural, neighborhood=True),
        "native_indicator": dict(structural, indicator=True),
        "native_full": full,
        "ridge_full": dict(full, model="ridge", native=False),
        "tabpfn_v2": dict(model="tabpfn", checkpoint="tabpfn-v2-regressor.ckpt",
                          version="10.2.0", ensemble="auto", seed=42),
    }


def absence_masks(X):
    # A missing counter conveys no affirmative evidence of absence.
    return {
        "fireplace": X["Fireplaces"].eq(0),
        "garage": X["GarageArea"].eq(0) & X["GarageCars"].eq(0),
        "basement": X["TotalBsmtSF"].eq(0),
        "pool": X["PoolArea"].eq(0),
    }


def diagnostic_counts(X):
    masks = absence_masks(X)
    return {
        "rows": len(X),
        "fireplace_quality_missing": int(X["FireplaceQu"].isna().sum()),
        "fireplace_missing_and_zero_count": int((X["FireplaceQu"].isna() & masks["fireplace"]).sum()),
        "quality_observed_despite_zero_fireplaces": int((X["FireplaceQu"].notna() & masks["fireplace"]).sum()),
        "garage_zero_area_nonzero_cars_or_reverse": int(((X.GarageArea.eq(0) != X.GarageCars.eq(0)) & X.GarageArea.notna() & X.GarageCars.notna()).sum()),
        "garage_type_missing_with_positive_area": int((X.GarageType.isna() & X.GarageArea.gt(0)).sum()),
        "garage_descriptor_observed_despite_zero_area_and_cars": int((masks["garage"] & X[list(GARAGE)].notna().any(axis=1)).sum()),
        "lot_frontage_missing": int(X.LotFrontage.isna().sum()),
    }


class AmesFeatures:
    """Learn numeric medians, neighborhood frontage and schema from training."""
    def __init__(self, spec):
        self.spec = dict(spec)

    def fit(self, X):
        self.columns = [col for col in X if col not in ("SalePrice", "Id")]
        self.frontage = X.LotFrontage.median()
        self.frontage = 0.0 if pd.isna(self.frontage) else self.frontage
        self.by_neighborhood = X.groupby("Neighborhood")["LotFrontage"].median()
        frame = self._prepare(X)
        self.cat = list(frame.select_dtypes(exclude=np.number))
        self.num = list(frame.select_dtypes(include=np.number))
        self.medians = frame[self.num].median().fillna(0.0)
        return self

    def _prepare(self, X):
        frame = X.loc[:, self.columns].copy()
        if self.spec["structural"]:
            groups = (("fireplace", ("FireplaceQu",)), ("garage", GARAGE),
                      ("basement", BASEMENT), ("pool", ("PoolQC",)))
            masks = absence_masks(frame)
            for group, cols in groups:
                for col in cols:
                    # Preserve an observed descriptor even when the counter conflicts.
                    absent = masks[group] & frame[col].isna()
                    frame.loc[absent, col] = ABSENT
            frame.loc[masks["garage"] & frame.GarageYrBlt.isna(), "GarageYrBlt"] = 0
        if self.spec["indicator"]:
            frame["LotFrontageMissing"] = frame.LotFrontage.isna().astype(float)
        if self.spec["neighborhood"]:
            frame["LotFrontage"] = frame.LotFrontage.fillna(frame.Neighborhood.map(self.by_neighborhood))
        frame["LotFrontage"] = frame.LotFrontage.fillna(self.frontage)

        # Same feature definitions for every recipe to isolate categorical handling.
        frame["TotalSF"] = frame[["1stFlrSF", "2ndFlrSF", "TotalBsmtSF"]].sum(axis=1, min_count=3)
        frame["TotalBath"] = frame.FullBath + frame.BsmtFullBath + 0.5 * (frame.HalfBath + frame.BsmtHalfBath)
        frame["HouseAge"] = frame.YrSold - frame.YearBuilt
        frame["RemodelAge"] = frame.YrSold - frame.YearRemodAdd
        frame["MSSubClass"] = frame.MSSubClass.astype("string")
        return frame

    def transform(self, X):
        frame = self._prepare(X)
        for col in self.cat:
            frame[col] = frame[col].astype("string").fillna(MISSING).astype(str)
        frame[self.num] = frame[self.num].fillna(self.medians)
        return frame


class FittedModel:
    def __init__(self, features, encoder, learner):
        self.features = features
        self.encoder = encoder
        self.learner = learner

    def predict(self, X):
        data = self.features.transform(X)
        if self.encoder is not None:
            data = self.encoder.transform(data)
        return np.maximum(np.expm1(self.learner.predict(data)), 0)


def fit_model(spec, X, y):
    y = np.asarray(y, dtype=float)
    if len(y) != len(X) or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Training requires one finite nonnegative price per home")
    if spec["model"] == "tabpfn":
        from tabpfn_bridge import FoundationRegressor
        return FoundationRegressor(X, y)
    features = AmesFeatures(spec).fit(X)
    data = features.transform(X)
    encoder = None
    if not spec["native"]:
        numeric = StandardScaler() if spec["model"] == "ridge" else "passthrough"
        encoder = ColumnTransformer([
            ("numeric", numeric, features.num),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), features.cat),
        ])
        data = encoder.fit_transform(data)
    if spec["model"] == "catboost":
        learner = CatBoostRegressor(iterations=500, depth=4, learning_rate=0.05,
                                    random_seed=42, thread_count=1, verbose=False,
                                    allow_writing_files=False,
                                    cat_features=features.cat if spec["native"] else None)
    elif spec["model"] == "ridge":
        learner = Ridge(alpha=30.0)
    else:
        raise ValueError(f"Unknown model family: {spec['model']}")
    learner.fit(data, np.log1p(y))
    return FittedModel(features, encoder, learner)


def crossfit(spec, X, y, folds):
    pred = np.full(len(X), np.nan)
    seen = np.zeros(len(X), dtype=bool)
    y = np.asarray(y)
    for train, val in folds:
        if len(np.intersect1d(train, val)) or seen[val].any():
            raise ValueError("Overlapping fold assignments")
        pred[val] = fit_model(spec, X.iloc[train], y[train]).predict(X.iloc[val])
        seen[val] = True
    if not seen.all() or not np.isfinite(pred).all():
        raise ValueError("Missing or nonfinite cross-fitted predictions")
    return pred


def _log_matrix(pred_matrix):
    p = np.asarray(pred_matrix, dtype=float)
    if p.ndim != 2 or p.shape[1] == 0 or (p < 0).any() or not np.isfinite(p).all():
        raise ValueError("Expected finite nonnegative [homes, models] predictions")
    return np.log1p(p)


def blend_weights(y, pred_matrix):
    """Fit nonnegative unit-sum weights to inner OOF predictions."""
    design = _log_matrix(pred_matrix)
    target = np.log1p(np.asarray(y, dtype=float))
    if target.shape != (len(design),) or not np.isfinite(target).all():
        raise ValueError("Invalid blend target")
    n = design.shape[1]
    def objective(w):
        residual = design @ w - target
        return np.mean(residual ** 2), 2 * design.T @ residual / len(target)
    fit = minimize(objective, np.ones(n) / n, jac=True, method="SLSQP",
                   bounds=[(0, 1)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"ftol": 1e-12, "maxiter": 300})
    if not fit.success:
        raise RuntimeError("Blend optimizer failed: " + fit.message)
    weights = np.maximum(fit.x, 0)
    return weights / weights.sum()


def blend_predict(pred_matrix, weights):
    design = _log_matrix(pred_matrix)
    weights = np.asarray(weights, dtype=float)
    if len(weights) != design.shape[1] or (weights < 0).any() or not np.isclose(weights.sum(), 1.):
        raise ValueError("Expected nonnegative weights summing to one")
    return np.maximum(0, np.expm1(design @ weights))
