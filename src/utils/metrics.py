"""
Evaluation metrics for regression models
"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error, mean_absolute_error


def calculate_rmsle(y_true, y_pred):
    """
    Calculate Root Mean Squared Log Error
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        RMSLE score
    """
    # Validate inputs
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must have same length. Got {len(y_true)} and {len(y_pred)}")
    
    if len(y_true) == 0:
        raise ValueError("Inputs cannot be empty")
    
    # Ensure no negative values
    y_pred = np.maximum(0, y_pred)
    y_true = np.maximum(0, y_true)
    
    # Check for inf or nan values before calculation
    if np.any(np.isinf(y_pred)) or np.any(np.isinf(y_true)) or np.any(np.isnan(y_pred)) or np.any(np.isnan(y_true)):
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=np.finfo(np.float64).max, neginf=0.0)
        y_true = np.nan_to_num(y_true, nan=0.0, posinf=np.finfo(np.float64).max, neginf=0.0)
    
    # Ensure values are still non-negative after nan_to_num (in case nan was replaced with negative)
    y_pred = np.maximum(0, y_pred)
    y_true = np.maximum(0, y_true)
    
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def calculate_metrics(y_true, y_pred):
    """
    Calculate multiple regression metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary with metrics
    """
    # Validate inputs
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must have same length. Got {len(y_true)} and {len(y_pred)}")
    
    if len(y_true) == 0:
        raise ValueError("Inputs cannot be empty")
    
    y_pred = np.maximum(0, y_pred)
    
    # Check for inf or nan values before calculation (for all metrics)
    if np.any(np.isinf(y_pred)) or np.any(np.isinf(y_true)):
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=np.finfo(np.float64).max, neginf=0.0)
        y_true = np.nan_to_num(y_true, nan=0.0, posinf=np.finfo(np.float64).max, neginf=0.0)
    
    # Check for NaN values
    if np.any(np.isnan(y_pred)) or np.any(np.isnan(y_true)):
        y_pred = np.nan_to_num(y_pred, nan=0.0)
        y_true = np.nan_to_num(y_true, nan=0.0)
    
    metrics = {
        'r2': r2_score(y_true, y_pred),
        'rmsle': calculate_rmsle(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred)
    }
    
    return metrics


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

