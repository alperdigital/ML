#!/usr/bin/env python3
"""
Example script: Compare the V1 tree models (XGBoost and LightGBM)

This script demonstrates how to:
1. Train multiple models
2. Compare their performance
3. Generate comparison visualizations
"""

import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tree_features import TreeFeatures
from src.models.xgboost_model import XGBoostModel
from src.models.lightgbm_model import LightGBMModel
from src.utils.metrics import calculate_metrics
from src.utils.visualization import plot_model_comparison, plot_residuals


def main():
    """Main comparison function"""
    
    print("="*60)
    print("Model Comparison - Ames Housing Price Prediction")
    print("="*60)
    
    # Load and preprocess data
    print("\n[1/4] Loading and preprocessing data...")
    try:
        train = pd.read_csv('data/train.csv')
    except FileNotFoundError as e:
        print(f"❌ Hata: Veri dosyası bulunamadı: {e}")
        print("Lütfen data/ klasöründe train.csv dosyasının olduğundan emin olun.")
        return
    
    if train.empty:
        print("❌ Hata: Veri dosyası boş!")
        return
    
    prices = train.pop("SalePrice")
    X_fit, X_val_raw, y_fit, y_val = train_test_split(
        train, prices, test_size=0.2, random_state=42
    )
    preparation = TreeFeatures()
    X_train_split, y_train_split = preparation.fit_transform(X_fit, y_fit)
    X_val = preparation.transform(X_val_raw)

    print("✓ Data prepared")
    
    # Train models
    print("\n[2/4] Training models...")
    models = {
        'XGBoost': XGBoostModel(),
        'LightGBM': LightGBMModel()
    }
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.train(X_train_split, y_train_split, verbose=True)
    
    print("✓ All models trained")
    
    # Evaluate models
    print("\n[3/4] Evaluating models...")
    results = {}
    predictions = {}  # Store predictions to avoid recomputing
    
    for name, model in models.items():
        y_pred = model.predict(X_val)
        predictions[name] = y_pred  # Store for later use
        metrics = calculate_metrics(y_val, y_pred)
        results[name] = metrics
        print(f"\n{name} Results:")
        print(f"  R² Score: {metrics['r2']:.5f}")
        print(f"  RMSLE: {metrics['rmsle']:.5f}")
        print(f"  RMSE: {metrics['rmse']:.2f}")
    
    # Visualize comparison
    print("\n[4/4] Generating visualizations...")
    plot_model_comparison(
        results,
        save_path='results/visualizations/model_comparison.png'
    )
    
    # Residual plots for each model (reuse stored predictions)
    for name in models.keys():
        plot_residuals(
            y_val, predictions[name], name,
            save_path=f'results/visualizations/{name.lower()}_residuals.png'
        )
    
    print("\n" + "="*60)
    print("Model comparison completed!")
    print("="*60)
    print("\nResults saved to results/visualizations/")


if __name__ == "__main__":
    main()

