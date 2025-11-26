"""
LightGBM Model for Ames Housing Price Prediction
"""

import os
import json
import pickle
import lightgbm as lgb
import numpy as np
from ..utils.metrics import calculate_metrics, print_metrics


class LightGBMModel:
    """
    LightGBM Regressor Model wrapper
    """
    
    def __init__(self, params=None):
        """
        Initialize LightGBM model
        
        Args:
            params: Dictionary of LightGBM parameters. If None, uses default optimal parameters.
        """
        self.default_params = {
            'objective': 'regression',
            'num_leaves': 5,
            'learning_rate': 0.05,
            'n_estimators': 720,
            'max_bin': 55,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'feature_fraction': 0.2319,
            'feature_fraction_seed': 9,
            'bagging_seed': 9,
            'min_data_in_leaf': 6,
            'min_sum_hessian_in_leaf': 11
        }
        
        self.params = params or self.default_params
        self.model = None
        self.feature_names = None
        
    def train(self, X_train, y_train, verbose=True):
        """
        Train the LightGBM model
        
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
        
        self.model = lgb.LGBMRegressor(**self.params)
        
        if verbose:
            print("LightGBM modeli eğitiliyor...")
        
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
        # LightGBM handles pandas DataFrames natively, so only reshape numpy arrays
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
            print_metrics(y, y_pred, "LightGBM")
        
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
        
        # LightGBM models can be saved directly
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
        except (IOError, OSError) as e:
            raise IOError(f"Could not save model to {filepath}: {e}")
        
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
        
        try:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
        except (IOError, OSError, pickle.UnpicklingError) as e:
            raise IOError(f"Could not load model from {filepath}: {e}")
        
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

