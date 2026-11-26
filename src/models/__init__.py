"""
Model classes for Ames Housing Price Prediction
"""

from .xgboost_model import XGBoostModel
from .lightgbm_model import LightGBMModel

__all__ = ['XGBoostModel', 'LightGBMModel', 'KANModel']


def __getattr__(name):
    if name == "KANModel":
        from .kan_model import KANModel
        return KANModel
    raise AttributeError(name)
