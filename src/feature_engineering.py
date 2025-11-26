"""
Feature engineering utilities for Ames Housing dataset
"""

import pandas as pd
import numpy as np
from yellowbrick.features import Rank1D
from sklearn.preprocessing import MinMaxScaler


class FeatureEngineer:
    """
    Feature engineering class for Ames Housing dataset
    """
    
    def __init__(self):
        """Initialize feature engineer"""
        self.scaler = MinMaxScaler()
        self.selected_features = None
        self.rank1d = None
        
    def create_new_features(self, data):
        """
        Create new features from existing ones
        
        Args:
            data: DataFrame
        
        Returns:
            DataFrame with new features
        """
        data = data.copy()
        features_created = 0
        
        # Total square footage
        required_cols = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
        if all(col in data.columns for col in required_cols):
            data['TotalSF'] = data['TotalBsmtSF'].fillna(0) + data['1stFlrSF'].fillna(0) + data['2ndFlrSF'].fillna(0)
            features_created += 1
        
        # Total bathrooms
        bath_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
        if all(col in data.columns for col in bath_cols):
            data['TotalBath'] = (data['FullBath'].fillna(0) + 0.5 * data['HalfBath'].fillna(0) + 
                                data['BsmtFullBath'].fillna(0) + 0.5 * data['BsmtHalfBath'].fillna(0))
            features_created += 1
        
        # Calculate YrSold median once if needed for multiple features
        yr_sold_median = None
        if 'YrSold' in data.columns:
            yr_sold_median_val = data['YrSold'].median()
            yr_sold_median = yr_sold_median_val if not data['YrSold'].isnull().all() and not pd.isna(yr_sold_median_val) else 2010
        
        # House age
        if 'YrSold' in data.columns and 'YearBuilt' in data.columns:
            year_built_median_val = data['YearBuilt'].median()
            year_built_median = year_built_median_val if not data['YearBuilt'].isnull().all() and not pd.isna(year_built_median_val) else 1970
            data['HouseAge'] = data['YrSold'].fillna(yr_sold_median) - data['YearBuilt'].fillna(year_built_median)
            features_created += 1
        
        # Remodel age
        if 'YrSold' in data.columns and 'YearRemodAdd' in data.columns:
            year_remod_median_val = data['YearRemodAdd'].median()
            year_remod_median = year_remod_median_val if not data['YearRemodAdd'].isnull().all() and not pd.isna(year_remod_median_val) else 1985
            data['RemodelAge'] = data['YrSold'].fillna(yr_sold_median) - data['YearRemodAdd'].fillna(year_remod_median)
            features_created += 1
        
        # Is remodeled
        if 'YearBuilt' in data.columns and 'YearRemodAdd' in data.columns:
            # Handle NaN values: if either is NaN, set to 0 (not remodeled)
            is_remodeled = (data['YearBuilt'] != data['YearRemodAdd'])
            data['IsRemodeled'] = is_remodeled.fillna(False).astype(int)
            features_created += 1
        
        # Total porch square footage
        porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
        available_porch_cols = [col for col in porch_cols if col in data.columns]
        if len(available_porch_cols) > 0:
            data['TotalPorchSF'] = data[available_porch_cols].fillna(0).sum(axis=1)
            features_created += 1
        
        # Overall score
        if 'OverallQual' in data.columns and 'OverallCond' in data.columns:
            data['OverallScore'] = data['OverallQual'].fillna(0) * data['OverallCond'].fillna(0)
            features_created += 1
        
        # Garage score
        if 'GarageCars' in data.columns and 'GarageArea' in data.columns:
            data['GarageScore'] = data['GarageCars'].fillna(0) * data['GarageArea'].fillna(0)
            features_created += 1
        
        # Total rooms
        if 'TotRmsAbvGrd' in data.columns and 'BedroomAbvGr' in data.columns:
            data['TotalRooms'] = data['TotRmsAbvGrd'].fillna(0) + data['BedroomAbvGr'].fillna(0)
            features_created += 1
        
        print(f"Yeni özellikler oluşturuldu: {features_created} özellik. Toplam özellik sayısı: {data.shape[1]}")
        
        return data
    
    def select_top_features(self, X, y, n_features=50, method='shapiro'):
        """
        Select top features using Rank1D
        
        Args:
            X: Features DataFrame
            y: Target Series
            n_features: Number of top features to select
            method: Ranking method ('shapiro' or 'pearson')
        
        Returns:
            DataFrame with selected features
        """
        # Validate inputs
        if X is None or len(X) == 0:
            raise ValueError("X cannot be empty")
        if y is None or len(y) == 0:
            raise ValueError("y cannot be empty")
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length. Got {len(X)} and {len(y)}")
        
        if n_features > len(X.columns):
            print(f"Warning: n_features ({n_features}) > available features ({len(X.columns)}). Using all features.")
            n_features = len(X.columns)
        
        feature_names = X.columns.values
        
        # Create Rank1D visualizer
        try:
            self.rank1d = Rank1D(features=feature_names, algorithm=method)
            self.rank1d.fit(X, y)
            self.rank1d.transform(X)
            
            # Get feature rankings
            if not hasattr(self.rank1d, 'ranks_') or self.rank1d.ranks_ is None:
                raise ValueError("Rank1D ranks_ not available. Ensure fit() was called successfully.")
            
            if len(self.rank1d.ranks_) != len(feature_names):
                raise ValueError(f"ranks_ length ({len(self.rank1d.ranks_)}) doesn't match feature_names length ({len(feature_names)})")
            
            df_ranks = pd.DataFrame({
                'feature_name': feature_names,
                'ranks': self.rank1d.ranks_
            }).sort_values(by=['ranks'], ascending=False)
            
            # Select top n features
            self.selected_features = df_ranks.head(n_features)['feature_name'].values
            
            # Validate all selected features exist in X
            available_features = [f for f in self.selected_features if f in X.columns]
            if len(available_features) < len(self.selected_features):
                missing = set(self.selected_features) - set(available_features)
                print(f"Warning: {len(missing)} selected features not in X.columns: {missing}")
                self.selected_features = available_features
            
            if len(available_features) == 0:
                print("Warning: No selected features available in X. Returning all features.")
                return X
            
            print(f"Top {len(available_features)} özellik seçildi.")
            
            return X[available_features]
        except Exception as e:
            print(f"Warning: Feature selection failed: {e}. Returning all features.")
            self.selected_features = feature_names
            return X
    
    def scale_features(self, X_train, X_test=None, fit=True):
        """
        Scale features using MinMaxScaler
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            fit: Whether to fit the scaler
        
        Returns:
            Scaled features
        """
        # Validate input
        if X_train is None or (hasattr(X_train, '__len__') and len(X_train) == 0):
            raise ValueError("X_train cannot be empty")
        
        if fit:
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            if not hasattr(self.scaler, 'scale_'):
                raise ValueError("Scaler must be fitted before transform. Set fit=True or call fit first.")
            X_train_scaled = self.scaler.transform(X_train)
        
        if X_test is not None:
            # Validate X_test
            # Check feature count match (handle both pandas DataFrames and numpy arrays)
            if hasattr(X_train, 'columns') and hasattr(X_test, 'columns'):
                # Pandas DataFrames
                if len(X_train.columns) != len(X_test.columns):
                    raise ValueError(f"X_test must have same number of features as X_train. Got {len(X_test.columns)} and {len(X_train.columns)}")
            elif hasattr(X_train, 'shape') and hasattr(X_test, 'shape') and len(X_train.shape) >= 2 and len(X_test.shape) >= 2:
                # NumPy arrays
                if X_train.shape[1] != X_test.shape[1]:
                    raise ValueError(f"X_test must have same number of features as X_train. Got {X_test.shape[1]} and {X_train.shape[1]}")
            
            if not hasattr(self.scaler, 'scale_'):
                raise ValueError("Scaler must be fitted before transforming test data.")
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def get_selected_features(self):
        """Get list of selected features"""
        if self.selected_features is None:
            return None
        # Handle both numpy array and list
        if hasattr(self.selected_features, 'tolist'):
            return self.selected_features.tolist()
        elif isinstance(self.selected_features, list):
            return self.selected_features
        else:
            # Convert to list if it's any other iterable
            return list(self.selected_features)
