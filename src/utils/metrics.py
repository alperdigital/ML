"""Evaluation metrics for Ames dollar-price regression models."""
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.ames_metrics import price_vectors, rmsle


def calculate_rmsle(y_true, y_pred):
    """Compare matched actual and predicted dollar prices on the log1p scale."""
    return rmsle(y_true, y_pred)


def calculate_metrics(y_true, y_pred):
    """Return finite-input, pairwise price metrics on their documented scales."""
    actual, predicted = price_vectors(y_true, y_pred)
    mse = mean_squared_error(actual, predicted)
    return {
        "r2": r2_score(actual, predicted),
        "rmsle": rmsle(actual, predicted),
        "rmse": np.sqrt(mse),
        "mae": mean_absolute_error(actual, predicted),
        "mse": mse,
    }


def print_metrics(y_true, y_pred, model_name="Model"):
    """
    Print formatted metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
    """
    metrics = calculate_metrics(y_true, y_pred)
    
    print("\n" + "="*60)
    print(f"{model_name} Performans Metrikleri")
    print("="*60)
    print(f"R² Score:     {metrics['r2']:.5f}")
    print(f"RMSLE:        {metrics['rmsle']:.5f}")
    print(f"RMSE:         {metrics['rmse']:.2f}")
    print(f"MAE:          {metrics['mae']:.2f}")
    print(f"MSE:          {metrics['mse']:.2f}")
    print("="*60 + "\n")
    
    return metrics

