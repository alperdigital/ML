"""Training-local feature preparation shared by all corrected V1 tree runs."""
import numpy as np
import pandas as pd

from .data_preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer


def fill_validation(preprocessor, train, validation, filled_train):
    """Apply the V1 missing-value rules without learning from validation."""
    validation = validation.copy()
    for col, rule in preprocessor.strategies.items():
        if col not in validation:
            continue
        observed = train[col].dropna()
        if rule == "Drop":
            validation = validation.drop(columns=col)
        elif rule == "NeighborhoodMedian":
            overall = observed.median() if len(observed) else 0
            replacement = overall
            if "Neighborhood" in train:
                groups = train.groupby("Neighborhood")[col].median()
                replacement = validation["Neighborhood"].map(groups).fillna(overall)
            validation[col] = validation[col].fillna(replacement)
        elif rule in ("None", "Zero", "Typ"):
            if validation[col].isna().any():
                validation[col] = validation[col].fillna({"None": "None", "Zero": 0, "Typ": "Typ"}[rule])
        elif rule == "Mode":
            mode = observed.mode()
            validation[col] = validation[col].fillna(mode.iloc[0] if len(mode) else 0)
        elif rule in ("Median", "Mean"):
            value = getattr(observed, rule.lower())() if len(observed) else 0
            validation[col] = validation[col].fillna(value)
    for col in filled_train.select_dtypes(include=np.number):
        if validation[col].isna().any():
            median = filled_train[col].median()
            validation[col] = validation[col].fillna(0 if pd.isna(median) else median)
    return validation


class TreeFeatures:
    """Fit V1's imputation, category encoders and output schema on training rows."""

    @staticmethod
    def _matrix(preprocessor, data, fit):
        encoded = preprocessor.encode_categorical(data, fit=fit)
        engineered = FeatureEngineer().create_new_features(encoded)
        return preprocessor.create_dummy_variables(engineered).astype(float)

    def fit_transform(self, raw, target):
        if not raw.index.equals(target.index) or not raw.index.is_unique:
            raise ValueError("Training prices must align with unique feature row indices")
        if "SalePrice" in raw:
            raise ValueError("Pass SalePrice separately from the feature matrix")
        pp = DataPreprocessor()
        self.raw_train = raw.copy()
        self.filled_train = pp.fill_missing_values(self.raw_train)
        retained = pp.remove_outliers(self.filled_train.assign(SalePrice=target))
        prices = retained.pop("SalePrice")
        matrix = self._matrix(pp, retained, fit=True)
        self.encoders = pp.label_encoders
        self.columns = matrix.columns
        return matrix, prices

    def transform(self, raw):
        if set(raw.columns) != set(self.raw_train.columns):
            raise ValueError("Input columns must match the training schema")
        pp = DataPreprocessor()
        pp.label_encoders = self.encoders
        data = fill_validation(pp, self.raw_train, raw, self.filled_train)
        matrix = self._matrix(pp, data, fit=False)
        return matrix.reindex(columns=self.columns, fill_value=0.0)
