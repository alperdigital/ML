#!/usr/bin/env python3
"""
Example script: Train XGBoost model for Ames Housing Price Prediction

This script demonstrates how to:
1. Load and preprocess data
2. Train an XGBoost model
3. Evaluate the model
4. Save predictions
"""

import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ames_data import read_ames_csv
from src.tree_features import TreeFeatures
from src.models.xgboost_model import XGBoostModel
from src.utils.metrics import print_metrics
from src.utils.visualization import plot_residuals, plot_feature_importance


def main():
    """Main training function"""
    
    print("="*60)
    print("XGBoost Model Training - Ames Housing Price Prediction")
    print("="*60)
    
    # Load data
    print("\n[1/5] Loading data...")
    try:
        train = read_ames_csv('data/train.csv')
        test = read_ames_csv('data/test.csv')
    except FileNotFoundError as e:
        print(f"❌ Hata: Veri dosyası bulunamadı: {e}")
        print("Lütfen data/ klasöründe train.csv ve test.csv dosyalarının olduğundan emin olun.")
        return
    
    if train.empty or test.empty:
        print("❌ Hata: Veri dosyaları boş!")
        return
    
    print(f"✓ Training data: {train.shape}")
    print(f"✓ Test data: {test.shape}")
    
    # Hold out raw sales before fitting imputation, category encoders or outlier rules.
    prices = train.pop("SalePrice")
    X_fit, raw_val, y_fit, y_val = train_test_split(
        train, prices, test_size=0.2, random_state=42
    )
    preparation = TreeFeatures()
    X_train_split, y_train_split = preparation.fit_transform(X_fit, y_fit)
    X_val = preparation.transform(raw_val)
    print(f"✓ Training set: {X_train_split.shape}, validation set: {X_val.shape}")

    # Train model
    print("\nTraining XGBoost model...")
    model = XGBoostModel()
    model.train(X_train_split, y_train_split, verbose=True)
    
    # Evaluate
    print("\n[5/5] Evaluating model...")
    y_pred = model.predict(X_val)
    metrics = print_metrics(y_val, y_pred, "XGBoost")
    
    # Visualizations
    print("\nGenerating visualizations...")
    plot_residuals(y_val, y_pred, "XGBoost", 
                   save_path='results/visualizations/xgboost_residuals.png')
    
    # Get feature names from model or use X_train_split columns
    if model.feature_names is not None:
        feature_names = model.feature_names
    elif hasattr(X_train_split, 'columns'):
        feature_names = X_train_split.columns.tolist()
    else:
        feature_names = [f'feature_{i}' for i in range(X_train_split.shape[1])]
    
    # Plot feature importance - wrapper model handles it automatically
    plot_feature_importance(model, feature_names, top_n=20,
                          save_path='results/visualizations/xgboost_feature_importance.png')
    
    # Use all labeled sales for the submission once holdout scoring is complete.
    submission_features = TreeFeatures()
    all_features, all_prices = submission_features.fit_transform(train, prices)
    model.train(all_features, all_prices, verbose=False)
    model.save_model('results/models/xgboost_model.json')
    test_predictions = model.predict(submission_features.transform(test))
    submission = pd.DataFrame({"Id": test["Id"], "SalePrice": test_predictions})

    # Create directory if it doesn't exist
    os.makedirs('results/submissions', exist_ok=True)
    submission.to_csv('results/submissions/xgboost_submission.csv', index=False)
    print("✓ Submission file saved: results/submissions/xgboost_submission.csv")
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

