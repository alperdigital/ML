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
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
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
        train = pd.read_csv('data/train.csv')
        test = pd.read_csv('data/test.csv')
    except FileNotFoundError as e:
        print(f"❌ Hata: Veri dosyası bulunamadı: {e}")
        print("Lütfen data/ klasöründe train.csv ve test.csv dosyalarının olduğundan emin olun.")
        return
    
    if train.empty or test.empty:
        print("❌ Hata: Veri dosyaları boş!")
        return
    
    print(f"✓ Training data: {train.shape}")
    print(f"✓ Test data: {test.shape}")
    
    # Preprocessing
    print("\n[2/5] Preprocessing data...")
    preprocessor = DataPreprocessor()
    
    # Separate target
    if 'SalePrice' not in train.columns:
        raise ValueError("'SalePrice' column not found in training data")
    y_train = train['SalePrice']
    X_train = train.drop('SalePrice', axis=1)
    
    # Save test Id column before preprocessing (might be dropped)
    test_ids_original = None
    if 'Id' in test.columns:
        test_ids_original = test['Id'].copy()
    
    # Fill missing values
    X_train = preprocessor.fill_missing_values(X_train)
    X_test = preprocessor.fill_missing_values(test)
    
    # Remove outliers
    # Ensure indices align before concatenation
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    train_with_target = pd.concat([X_train, y_train], axis=1)
    train_with_target = preprocessor.remove_outliers(train_with_target, target_col='SalePrice')
    
    if 'SalePrice' not in train_with_target.columns:
        raise ValueError("'SalePrice' column missing after outlier removal")
    
    y_train = train_with_target['SalePrice']
    X_train = train_with_target.drop('SalePrice', axis=1)
    
    # Encode categorical
    X_train = preprocessor.encode_categorical(X_train, fit=True)
    X_test = preprocessor.encode_categorical(X_test, fit=False)
    
    print("✓ Preprocessing completed")
    
    # Feature engineering
    print("\n[3/5] Feature engineering...")
    fe = FeatureEngineer()
    X_train = fe.create_new_features(X_train)
    X_test = fe.create_new_features(X_test)
    
    # Align columns
    common_cols = X_train.columns.intersection(X_test.columns)
    if len(common_cols) == 0:
        raise ValueError("No common columns between X_train and X_test after feature engineering!")
    
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    print(f"✓ Feature engineering completed. Common columns: {len(common_cols)}")
    
    # Split data for evaluation
    print("\n[4/5] Splitting data for evaluation...")
    from sklearn.model_selection import train_test_split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    print(f"✓ Training set: {X_train_split.shape}, Validation set: {X_val.shape}")
    
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
    
    # Save model
    model.save_model('results/models/xgboost_model.json')
    
    # Make predictions on test set
    print("\nMaking predictions on test set...")
    test_predictions = model.predict(X_test)
    
    # Create submission file
    # Handle Id column - might be in original test or need to be created
    # Use len(test_predictions) to ensure IDs match predictions length
    num_predictions = len(test_predictions)
    if test_ids_original is not None and len(test_ids_original) == num_predictions:
        test_ids = test_ids_original
    else:
        # If Id was dropped or length mismatch, create sequential IDs starting from 1461
        test_ids = pd.Series(range(1461, 1461 + num_predictions), name='Id')
    
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': test_predictions
    })
    
    # Create directory if it doesn't exist
    os.makedirs('results/submissions', exist_ok=True)
    submission.to_csv('results/submissions/xgboost_submission.csv', index=False)
    print("✓ Submission file saved: results/submissions/xgboost_submission.csv")
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

