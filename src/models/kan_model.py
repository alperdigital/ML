"""
KAN (Kolmogorov-Arnold Network) Model for Ames Housing Price Prediction
"""

import os
import copy
import torch
import numpy as np
import pandas as pd
from kan import KAN
from sklearn.preprocessing import StandardScaler
from ..utils.metrics import calculate_metrics, print_metrics


class KANModel:
    """
    KAN Model wrapper for regression
    """
    
    def __init__(self, input_dim=None, params=None):
        """
        Initialize KAN model
        
        Args:
            input_dim: Input dimension (number of features)
            params: Dictionary of KAN parameters. If None, uses default optimal parameters.
        """
        self.default_params = {
            'width': [None, 16, 8, 1],  # Will be set based on input_dim
            'grid': 5,
            'k': 4,
            'seed': 42,
            'opt': 'LBFGS',
            'steps': 222,
            'lr': 0.063,
            'lamb': 0.0012,
            'lamb_entropy': 0.8,
            'update_grid': False
        }
        
        # Use deepcopy to avoid mutating default_params (width is a list)
        self.params = params or copy.deepcopy(self.default_params)
        self.input_dim = input_dim
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
        # Set width based on input_dim if provided
        if input_dim is not None:
            self.params['width'][0] = input_dim
    
    def _prepare_data(self, X, y=None, fit_scaler=False):
        """
        Prepare data for KAN model (scaling and tensor conversion)
        
        Args:
            X: Features
            y: Target (optional)
            fit_scaler: Whether to fit the scaler
        
        Returns:
            Dictionary with tensors
        """
        # Validate input
        if X is None or (hasattr(X, '__len__') and len(X) == 0):
            raise ValueError("X cannot be empty")
        
        # Convert to numpy array (handles both pandas DataFrame and numpy array)
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.asarray(X)
        
        # Handle 1D array (single sample) - reshape to 2D
        if len(X_array.shape) == 1:
            X_array = X_array.reshape(1, -1)
        
        # Scale features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X_array)
        else:
            if not hasattr(self.scaler, 'mean_'):
                raise ValueError("Scaler must be fitted before transform. Call train() first or set fit_scaler=True.")
            X_scaled = self.scaler.transform(X_array)
        
        # Convert to tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        result = {'input': X_tensor}
        
        if y is not None:
            # Ensure y is positive for log transform
            y_clean = np.array(y)
            # Check for invalid values before transformation
            if np.any(np.isnan(y_clean)):
                raise ValueError("Target values contain NaN. Cannot perform log transform.")
            if np.any(y_clean <= -1):
                raise ValueError("Target values must be > -1 for log1p transform. Found values <= -1.")
            y_clean = np.maximum(y_clean, 0)  # Ensure non-negative
            
            # Log transform target for RMSLE optimization
            y_log = np.log1p(y_clean)
            # Convert to numpy array (handles both pandas Series and numpy array)
            if isinstance(y_log, pd.Series):
                y_array = y_log.values
            else:
                y_array = np.asarray(y_log)
            y_tensor = torch.tensor(y_array.reshape(-1, 1), dtype=torch.float32)
            result['label'] = y_tensor
        
        return result
    
    def train(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """
        Train the KAN model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            verbose: Whether to print training progress
        """
        # Validate inputs
        if X_train is None or len(X_train) == 0:
            raise ValueError("X_train cannot be empty")
        if y_train is None or len(y_train) == 0:
            raise ValueError("y_train cannot be empty")
        if len(X_train) != len(y_train):
            raise ValueError(f"X_train and y_train must have same length. Got {len(X_train)} and {len(y_train)}")
        
        if self.input_dim is None:
            if hasattr(X_train, 'shape'):
                self.input_dim = X_train.shape[1]
            elif hasattr(X_train, 'columns'):
                self.input_dim = len(X_train.columns)
            else:
                raise ValueError("Cannot determine input dimension from X_train")
            self.params['width'][0] = self.input_dim
        
        # Store feature names if available
        if hasattr(X_train, 'columns'):
            self.feature_names = X_train.columns.tolist()
        elif hasattr(X_train, 'shape'):
            self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        
        # Prepare training data
        train_data = self._prepare_data(X_train, y_train, fit_scaler=True)
        
        # Prepare validation data if provided
        if X_val is not None and y_val is not None:
            # Validate validation data
            if len(X_val) == 0 or len(y_val) == 0:
                raise ValueError("X_val and y_val cannot be empty")
            if len(X_val) != len(y_val):
                raise ValueError(f"X_val and y_val must have same length. Got {len(X_val)} and {len(y_val)}")
            val_data = self._prepare_data(X_val, y_val, fit_scaler=False)
            dataset = {
                "train_input": train_data['input'],
                "train_label": train_data['label'],
                "test_input": val_data['input'],
                "test_label": val_data['label']
            }
        else:
            # Use training data for both train and test
            dataset = {
                "train_input": train_data['input'],
                "train_label": train_data['label'],
                "test_input": train_data['input'],
                "test_label": train_data['label']
            }
        
        # Initialize model
        self.model = KAN(
            width=self.params['width'],
            grid=self.params['grid'],
            k=self.params['k'],
            seed=self.params['seed']
        )
        
        if verbose:
            print("KAN modeli eğitiliyor...")
            print(f"Model yapısı: {self.params['width']}")
            print(f"Eğitim adımları: {self.params['steps']}")
        
        # Training parameters
        fit_params = {
            "opt": self.params['opt'],
            "steps": self.params['steps'],
            "lr": self.params['lr'],
            "lamb": self.params['lamb'],
            "lamb_entropy": self.params['lamb_entropy'],
            "update_grid": self.params['update_grid']
        }
        
        # Train model
        self.model.fit(dataset, **fit_params)
        
        if verbose:
            print("✓ Model eğitimi tamamlandı!")
        
        return self
    
    def predict(self, X, return_log=False):
        """
        Make predictions
        
        Args:
            X: Features to predict on
            return_log: Whether to return log-transformed predictions
        
        Returns:
            Predictions array (exp-transformed by default)
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmemiş. Önce train() metodunu çağırın.")
        
        # Validate input
        if X is None or (hasattr(X, '__len__') and len(X) == 0):
            raise ValueError("X cannot be empty")
        
        # Prepare data
        data = self._prepare_data(X, fit_scaler=False)
        X_tensor = data['input']
        
        # Make predictions
        with torch.no_grad():
            y_pred_log_tensor = self.model(X_tensor)
            # Detach and convert to numpy (handles both CPU and GPU tensors)
            y_pred_log = y_pred_log_tensor.detach().cpu().numpy().flatten()
        
        if return_log:
            return y_pred_log
        
        # Transform back from log space
        y_pred = np.expm1(y_pred_log)
        
        # Check for inf or nan values (overflow/underflow protection)
        if np.any(np.isinf(y_pred)) or np.any(np.isnan(y_pred)):
            # Clip extreme values to prevent inf
            y_pred = np.clip(y_pred, 0, np.finfo(np.float32).max)
            # Replace any remaining inf/nan with 0
            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=np.finfo(np.float32).max, neginf=0.0)
        
        return y_pred
    
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
            print_metrics(y, y_pred, "KAN")
        
        return metrics
    
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
        
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'params': self.params,
                'input_dim': self.input_dim,
                'feature_names': self.feature_names
            }, filepath)
        except (IOError, OSError) as e:
            raise IOError(f"Could not save model to {filepath}: {e}")
        
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
            checkpoint = torch.load(filepath, map_location='cpu')
        except (IOError, OSError, RuntimeError) as e:
            raise IOError(f"Could not load model from {filepath}: {e}")
        
        # Validate checkpoint structure
        required_keys = ['params', 'input_dim', 'scaler', 'model_state_dict']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise ValueError(f"Invalid checkpoint file. Missing keys: {missing_keys}")
        
        self.params = checkpoint['params']
        self.input_dim = checkpoint['input_dim']
        self.scaler = checkpoint['scaler']
        self.feature_names = checkpoint.get('feature_names', None)
        
        # Validate params structure before reinitializing model
        if 'width' not in self.params or self.params['width'] is None:
            raise ValueError("Invalid params: 'width' is missing or None")
        if not isinstance(self.params['width'], list) or len(self.params['width']) < 2:
            raise ValueError(f"Invalid params: 'width' must be a list with at least 2 elements. Got: {self.params['width']}")
        
        # Reinitialize model
        try:
            self.model = KAN(
                width=self.params['width'],
                grid=self.params.get('grid', 5),
                k=self.params.get('k', 4),
                seed=self.params.get('seed', 42)
            )
        except Exception as e:
            raise ValueError(f"Failed to reinitialize KAN model: {e}")
        
        # Load model state
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            raise ValueError(f"Failed to load model state dict: {e}")
        
        print(f"Model yüklendi: {filepath}")

