"""
Visualization utilities for Ames Housing Price Prediction
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats


def setup_plot_style():
    """Setup consistent plotting style"""
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        try:
            plt.style.use('seaborn')
        except OSError:
            plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10


def plot_model_comparison(models_results, save_path=None):
    """
    Create a comparison dashboard for multiple models
    
    Args:
        models_results: Dictionary with model names as keys and metrics as values
        save_path: Path to save the figure
    """
    if not models_results or len(models_results) == 0:
        raise ValueError("models_results cannot be empty")
    
    setup_plot_style()
    
    models = list(models_results.keys())
    if len(models) == 0:
        raise ValueError("models_results must contain at least one model")
    
    r2_scores = [models_results[m].get('r2', 0) if models_results[m].get('r2') is not None else 0 for m in models]
    rmsle_scores = [models_results[m].get('rmsle', 0) if models_results[m].get('rmsle') is not None else 0 for m in models]
    
    # Validate scores are not all None/zero (at least one valid score)
    if all(score == 0 for score in r2_scores) and all(score == 0 for score in rmsle_scores):
        raise ValueError("All model scores are None or invalid. Cannot create comparison plot.")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # R² Comparison
    axes[0].bar(models, r2_scores, color='skyblue', edgecolor='black')
    axes[0].set_title('R² Score Karşılaştırması', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('R² Score', fontsize=12)
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(r2_scores):
        axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
    
    # RMSLE Comparison
    axes[1].bar(models, rmsle_scores, color='coral', edgecolor='black')
    axes[1].set_title('RMSLE Karşılaştırması (Düşük = İyi)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('RMSLE', fontsize=12)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(rmsle_scores):
        axes[1].text(i, v + 0.005, f'{v:.4f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(save_path)
        os.makedirs(dir_path if dir_path else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grafik kaydedildi: {save_path}")
    
    return fig


def plot_residuals(y_true, y_pred, model_name="Model", save_path=None):
    """
    Create residual analysis plots
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
        save_path: Path to save the figure
    """
    # Validate inputs
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must have same length. Got {len(y_true)} and {len(y_pred)}")
    if len(y_true) == 0:
        raise ValueError("Inputs cannot be empty")
    
    setup_plot_style()
    
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Residual vs Predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_title(f'{model_name} - Residuals vs Predicted', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Predicted Values', fontsize=10)
    axes[0, 0].set_ylabel('Residuals', fontsize=10)
    axes[0, 0].grid(alpha=0.3)
    
    # Residual Histogram
    axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Residual', fontsize=10)
    axes[0, 1].set_ylabel('Frequency', fontsize=10)
    axes[0, 1].grid(alpha=0.3)
    
    # Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # Actual vs Predicted
    axes[1, 1].scatter(y_true, y_pred, alpha=0.5, s=20)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[1, 1].set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Actual Values', fontsize=10)
    axes[1, 1].set_ylabel('Predicted Values', fontsize=10)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(save_path)
        os.makedirs(dir_path if dir_path else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grafik kaydedildi: {save_path}")
    
    return fig


def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """
    Plot feature importance
    
    Args:
        model: Trained model with feature_importances_ attribute (can be wrapper or actual model)
        feature_names: List of feature names
        top_n: Number of top features to display
        save_path: Path to save the figure
    """
    # Validate inputs
    if model is None:
        raise ValueError("model cannot be None")
    if feature_names is None or len(feature_names) == 0:
        raise ValueError("feature_names cannot be None or empty")
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    
    setup_plot_style()
    
    # Handle both wrapper models and direct models
    if hasattr(model, 'feature_importances_'):
        # Direct model (e.g., xgb.XGBRegressor)
        importances = model.feature_importances_
    elif hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
        # Wrapper model (e.g., XGBoostModel)
        importances = model.model.feature_importances_
    elif hasattr(model, 'get_feature_importances'):
        # Wrapper with method
        importances = model.get_feature_importances()
    else:
        raise ValueError("Model does not have feature_importances_ attribute or get_feature_importances method")
    
    # Validate importances
    if importances is None or len(importances) == 0:
        raise ValueError("Model feature importances are empty")
    
    # Validate feature_names length matches importances
    if len(feature_names) != len(importances):
        raise ValueError(f"Feature names length ({len(feature_names)}) doesn't match importances length ({len(importances)})")
    
    # Create DataFrame
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(top_n)
    
    # Plot
    fig = plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
    plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(save_path)
        os.makedirs(dir_path if dir_path else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grafik kaydedildi: {save_path}")
    
    return fig


def plot_training_history(history, save_path=None):
    """
    Plot training history for KAN model
    
    Args:
        history: Dictionary with training history
        save_path: Path to save the figure
    """
    if not history or not isinstance(history, dict):
        raise ValueError("history must be a non-empty dictionary")
    
    setup_plot_style()
    
    # Get available data
    train_loss = history.get('train_loss', [])
    test_loss = history.get('test_loss', [])
    train_r2 = history.get('train_r2', [])
    test_r2 = history.get('test_r2', [])
    
    if not train_loss and not train_r2:
        raise ValueError("history must contain at least 'train_loss' or 'train_r2'")
    
    # Determine epochs based on available data
    max_len = max(len(train_loss), len(test_loss), len(train_r2), len(test_r2), 1)
    epochs = range(1, max_len + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    if train_loss and test_loss:
        # Ensure same length
        min_len = min(len(train_loss), len(test_loss), len(epochs))
        axes[0].plot(epochs[:min_len], train_loss[:min_len], label='Train Loss', marker='o')
        axes[0].plot(epochs[:min_len], test_loss[:min_len], label='Test Loss', marker='s')
        axes[0].set_title('Model Loss During Training', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=10)
        axes[0].set_ylabel('Loss', fontsize=10)
        axes[0].legend()
        axes[0].grid(alpha=0.3)
    elif train_loss:
        min_len = min(len(train_loss), len(epochs))
        axes[0].plot(epochs[:min_len], train_loss[:min_len], label='Train Loss', marker='o')
        axes[0].set_title('Model Loss During Training', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=10)
        axes[0].set_ylabel('Loss', fontsize=10)
        axes[0].legend()
        axes[0].grid(alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'No loss data available', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Model Loss During Training', fontsize=12, fontweight='bold')
    
    # R² plot
    if train_r2 and test_r2:
        # Ensure same length
        min_len = min(len(train_r2), len(test_r2), len(epochs))
        axes[1].plot(epochs[:min_len], train_r2[:min_len], label='Train R²', marker='o')
        axes[1].plot(epochs[:min_len], test_r2[:min_len], label='Test R²', marker='s')
        axes[1].set_title('R² Score During Training', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=10)
        axes[1].set_ylabel('R² Score', fontsize=10)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    elif train_r2:
        min_len = min(len(train_r2), len(epochs))
        axes[1].plot(epochs[:min_len], train_r2[:min_len], label='Train R²', marker='o')
        axes[1].set_title('R² Score During Training', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=10)
        axes[1].set_ylabel('R² Score', fontsize=10)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No R² data available', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('R² Score During Training', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(save_path)
        os.makedirs(dir_path if dir_path else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grafik kaydedildi: {save_path}")
    
    return fig


def plot_correlation_heatmap(data, top_n=30, save_path=None):
    """
    Plot correlation heatmap for top features
    
    Args:
        data: DataFrame with features
        top_n: Number of top correlated features
        save_path: Path to save the figure
    """
    setup_plot_style()
    
    if 'SalePrice' not in data.columns:
        raise ValueError("DataFrame must contain 'SalePrice' column")
    
    numeric_data = data.select_dtypes(include=[np.number])
    
    if 'SalePrice' not in numeric_data.columns:
        raise ValueError("'SalePrice' must be a numeric column")
    
    if len(numeric_data.columns) < 2:
        raise ValueError("DataFrame must have at least 2 numeric columns")
    
    corr = numeric_data.corr().abs()
    
    # Get top correlated features with SalePrice (excluding SalePrice itself)
    top_corr = corr['SalePrice'].sort_values(ascending=False).head(top_n + 1)
    top_features = top_corr.index.tolist()
    
    # Remove SalePrice from features (it's the target, not a feature)
    top_features = [f for f in top_features if f != 'SalePrice' and f in numeric_data.columns]
    
    # If we removed SalePrice, we might have fewer features, so add SalePrice back for correlation matrix
    # (SalePrice will show correlation with other features but not with itself in the heatmap)
    if 'SalePrice' in numeric_data.columns and 'SalePrice' not in top_features:
        # Include SalePrice in the matrix to show its correlations with other features
        top_features_for_matrix = top_features + ['SalePrice']
    else:
        top_features_for_matrix = top_features
    
    if len(top_features) == 0:
        raise ValueError("No valid features found for correlation heatmap")
    
    # Ensure we have at least one feature (excluding SalePrice) for meaningful heatmap
    if len(top_features_for_matrix) < 2:
        raise ValueError("Need at least 2 features (including SalePrice) for correlation heatmap")
    
    # Create correlation matrix for top features (including SalePrice for reference)
    corr_matrix = numeric_data[top_features_for_matrix].corr()
    
    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(f'Correlation Heatmap - Top {top_n} Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(save_path)
        os.makedirs(dir_path if dir_path else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grafik kaydedildi: {save_path}")
    
    return fig

