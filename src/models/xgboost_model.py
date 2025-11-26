"""
XGBoost Model for Ames Housing Price Prediction
"""

import os
import json
import xgboost as xgb
import numpy as np
from ..utils.metrics import calculate_metrics, print_metrics


class XGBoostModel:
    """
    XGBoost Regressor Model wrapper
    """
    
    def __init__(self, params=None):
        """
        Initialize XGBoost model
        
        Args:
            params: Dictionary of XGBoost parameters. If None, uses default optimal parameters.
        """
        self.default_params = {
            'colsample_bytree': 0.89407,
            'gamma': 0.0012,
            'learning_rate': 0.063732,
            'max_depth': 4,
            'n_estimators': 222,
            'subsample': 0.5213,
            'reg_alpha': 0.4640,
            'reg_lambda': 0.8571,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0  # Replaced deprecated 'silent' parameter
        }
        
        self.params = params or self.default_params
        self.model = None
        self.feature_names = None
        
    def train(self, X_train, y_train, verbose=True):
        """
        Train the XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training target
            verbose: Whether to print training progress
        """
        # Validate inputs
        if X_train is None or (hasattr(X_train, '__len__') and len(X_train) == 0):
            raise ValueError("X_train cannot be empty")
        if y_train is None or (hasattr(y_train, '__len__') and len(y_train) == 0):
            raise ValueError("y_train cannot be empty")
        if hasattr(X_train, '__len__') and hasattr(y_train, '__len__'):
            if len(X_train) != len(y_train):
                raise ValueError(f"X_train and y_train must have same length. Got {len(X_train)} and {len(y_train)}")
        
        self.model = xgb.XGBRegressor(**self.params)
        
        if verbose:
            print("XGBoost modeli eğitiliyor...")
        
        self.model.fit(X_train, y_train)
        
        # Store feature names if available
        if hasattr(X_train, 'columns'):
            self.feature_names = X_train.columns.tolist()
        elif hasattr(X_train, 'shape'):
            self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        
        if verbose:
            print("✓ Model eğitimi tamamlandı!")
        
        return self
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Features to predict on
        
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmemiş. Önce train() metodunu çağırın.")
        
        # Validate input
        if X is None or (hasattr(X, '__len__') and len(X) == 0):
            raise ValueError("X cannot be empty")
        
        # Handle 1D array (single sample) - reshape to 2D
        # XGBoost handles pandas DataFrames natively, so only reshape numpy arrays
        if hasattr(X, 'shape') and not hasattr(X, 'columns') and len(X.shape) == 1:
            # NumPy array - reshape to 2D
            X = X.reshape(1, -1)
        
        # Check feature count matches
        if hasattr(X, 'shape') and len(X.shape) >= 2 and hasattr(self.model, 'n_features_in_'):
            if X.shape[1] != self.model.n_features_in_:
                raise ValueError(f"Feature count mismatch. Expected {self.model.n_features_in_}, got {X.shape[1]}")
        
        return self.model.predict(X)
    
    def evaluate(self, X, y, verbose=True):
        """
        Evaluate model performance
        
        Args:
            X: Features
            y: True target values
            verbose: Whether to print metrics
        
        Returns:
            Dictionary with metrics
        """
        y_pred = self.predict(X)
        metrics = calculate_metrics(y, y_pred)
        
        if verbose:
            print_metrics(y, y_pred, "XGBoost")
        
        return metrics
    
    def get_feature_importances(self):
        """
        Get feature importances
        
        Returns:
            Array of feature importances
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmemiş.")
        
        return self.model.feature_importances_
    
    def save_model(self, filepath):
        """
        Save model to file
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmemiş.")
        
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path if dir_path else '.', exist_ok=True)
        
        # Save model
        try:
            self.model.save_model(filepath)
        except (IOError, OSError) as e:
            raise IOError(f"Could not save XGBoost model to {filepath}: {e}")
        
        # Save feature_names if available
        if self.feature_names is not None:
            metadata_file = filepath + '.metadata.json'
            try:
                with open(metadata_file, 'w') as f:
                    json.dump({'feature_names': self.feature_names}, f)
            except (IOError, OSError) as e:
                print(f"Warning: Could not save metadata file {metadata_file}: {e}")
        
        print(f"Model kaydedildi: {filepath}")
    
    def load_model(self, filepath):
        """
        Load model from file
        
        Args:
            filepath: Path to load the model from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {filepath}")
        
        self.model = xgb.XGBRegressor()
        try:
            self.model.load_model(filepath)
        except Exception as e:
            # Catch any XGBoost or file-related errors
            if isinstance(e, (IOError, OSError)):
                raise IOError(f"Could not load XGBoost model from {filepath}: {e}")
            else:
                raise ValueError(f"Error loading XGBoost model from {filepath}: {e}")
        
        # Load feature_names if available
        metadata_file = filepath + '.metadata.json'
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.feature_names = metadata.get('feature_names', None)
            except (IOError, OSError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load metadata file {metadata_file}: {e}")
        
        print(f"Model yüklendi: {filepath}")

