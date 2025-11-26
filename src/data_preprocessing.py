"""
Data preprocessing utilities for Ames Housing dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from scipy.stats import skew
from scipy.special import boxcox1p


class DataPreprocessor:
    """
    Data preprocessing class for Ames Housing dataset
    """
    
    def __init__(self):
        """Initialize preprocessor with default strategies"""
        self.strategies = self._default_strategies()
        self.label_encoders = {}
        self.scaler = None
        
    def _default_strategies(self):
        """Default missing value filling strategies"""
        return {
            'PoolQC': 'None',
            'MiscFeature': 'None',
            'Alley': 'None',
            'Fence': 'None',
            'FireplaceQu': 'None',
            'LotFrontage': 'NeighborhoodMedian',
            'GarageType': 'None',
            'GarageFinish': 'None',
            'GarageQual': 'None',
            'GarageCond': 'None',
            'GarageYrBlt': 'Zero',
            'GarageArea': 'Zero',
            'GarageCars': 'Zero',
            'BsmtFinSF1': 'Zero',
            'BsmtFinSF2': 'Zero',
            'BsmtUnfSF': 'Zero',
            'TotalBsmtSF': 'Zero',
            'BsmtFullBath': 'Zero',
            'BsmtHalfBath': 'Zero',
            'BsmtQual': 'None',
            'BsmtCond': 'None',
            'BsmtExposure': 'None',
            'BsmtFinType1': 'None',
            'BsmtFinType2': 'None',
            'MasVnrType': 'None',
            'MasVnrArea': 'Zero',
            'MSZoning': 'Mode',
            'Utilities': 'Drop',
            'Functional': 'Typ',
            'Electrical': 'Mode',
            'KitchenQual': 'Mode',
            'Exterior1st': 'Mode',
            'Exterior2nd': 'Mode',
            'SaleType': 'Mode',
            'MSSubClass': 'None'
        }
    
    def fill_missing_values(self, data, strategies=None):
        """
        Fill missing values based on strategies
        
        Args:
            data: DataFrame with missing values
            strategies: Dictionary of strategies (optional)
        
        Returns:
            DataFrame with filled missing values
        """
        if strategies is None:
            strategies = self.strategies
        
        if data is None or data.empty:
            raise ValueError("Input data cannot be None or empty")
        
        data = data.copy()
        
        for column, strategy in strategies.items():
            if column not in data.columns:
                continue
                
            if strategy == 'None':
                data[column] = data[column].fillna('None')
            elif strategy == 'Zero':
                data[column] = data[column].fillna(0)
            elif strategy == 'Mode':
                mode_result = data[column].mode()
                if not mode_result.empty:
                    mode_value = mode_result[0]
                else:
                    # If no mode, use first non-null value or default
                    non_null_values = data[column].dropna()
                    mode_value = non_null_values.iloc[0] if len(non_null_values) > 0 else 0
                data[column] = data[column].fillna(mode_value)
            elif strategy == 'NeighborhoodMedian':
                if 'Neighborhood' in data.columns:
                    # Calculate overall median as fallback
                    overall_median = data[column].median()
                    if pd.isna(overall_median):
                        overall_median = 0
                    
                    # Fill missing values with neighborhood median, fallback to overall median
                    def fill_neighborhood_median(x):
                        group_median = x.median()
                        if pd.isna(group_median):
                            # If group median is NaN (all values in group are NaN), use overall median
                            return x.fillna(overall_median)
                        else:
                            return x.fillna(group_median)
                    
                    data[column] = data.groupby("Neighborhood")[column].transform(fill_neighborhood_median)
                else:
                    median_val = data[column].median()
                    if pd.isna(median_val):
                        data[column] = data[column].fillna(0)
                    else:
                        data[column] = data[column].fillna(median_val)
            elif strategy == 'Typ':
                data[column] = data[column].fillna('Typ')
            elif strategy == 'Drop':
                data = data.drop(columns=[column])
            elif strategy == 'Median':
                median_val = data[column].median()
                if pd.isna(median_val):
                    data[column] = data[column].fillna(0)
                else:
                    data[column] = data[column].fillna(median_val)
            elif strategy == 'Mean':
                mean_val = data[column].mean()
                if pd.isna(mean_val):
                    data[column] = data[column].fillna(0)
                else:
                    data[column] = data[column].fillna(mean_val)
        
        # Fill remaining numeric columns with median
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].isnull().any():
                median_val = data[col].median()
                # Handle case where all values are NaN
                if pd.isna(median_val):
                    data[col] = data[col].fillna(0)
                else:
                    data[col] = data[col].fillna(median_val)
        
        return data
    
    def remove_outliers(self, data, target_col='SalePrice'):
        """
        Remove outliers based on domain knowledge
        
        Args:
            data: DataFrame
            target_col: Target column name
        
        Returns:
            DataFrame with outliers removed
        """
        data = data.copy()
        original_shape = data.shape[0]
        
        # Check if target column exists
        if target_col not in data.columns:
            print(f"Warning: Target column '{target_col}' not found. Skipping outlier removal.")
            return data
        
        # Large house but cheap
        if 'GrLivArea' in data.columns:
            mask = (data['GrLivArea'] > 4000) & (data[target_col] < 300000)
            if mask.any():
                data = data.drop(data[mask].index)
        
        # Large basement but cheap
        if 'TotalBsmtSF' in data.columns:
            mask = (data['TotalBsmtSF'] > 3000) & (data[target_col] < 200000)
            if mask.any():
                data = data.drop(data[mask].index)
        
        # Old house but expensive
        if 'YearBuilt' in data.columns:
            mask = (data['YearBuilt'] < 1920) & (data[target_col] > 500000)
            if mask.any():
                data = data.drop(data[mask].index)
        
        # Large garage but cheap
        if 'GarageArea' in data.columns:
            mask = (data['GarageArea'] > 1000) & (data[target_col] < 150000)
            if mask.any():
                data = data.drop(data[mask].index)
        
        removed = original_shape - data.shape[0]
        if removed > 0:
            print(f"{removed} aykırı değer çıkarıldı. Yeni shape: {data.shape}")
        
        # Validate that DataFrame is not empty after outlier removal
        if data.shape[0] == 0:
            raise ValueError("All data points were removed as outliers. Cannot proceed with empty dataset.")
        
        return data
    
    def encode_categorical(self, data, label_encoder_cols=None, fit=True):
        """
        Encode categorical variables
        
        Args:
            data: DataFrame
            label_encoder_cols: List of columns to label encode
            fit: Whether to fit encoders (True for training, False for test)
        
        Returns:
            DataFrame with encoded categorical variables
        """
        data = data.copy()
        
        if label_encoder_cols is None:
            label_encoder_cols = [
                'FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
                'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual',
                'BsmtFinType1', 'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure',
                'GarageFinish', 'LandSlope', 'LotShape', 'PavedDrive', 'Street',
                'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 'YrSold', 'MoSold'
            ]
        
        for col_name in label_encoder_cols:
            if col_name not in data.columns:
                continue
            
            if fit:
                le = LabelEncoder()
                data[col_name] = le.fit_transform(data[col_name].astype(str))
                self.label_encoders[col_name] = le
            else:
                if col_name in self.label_encoders:
                    le = self.label_encoders[col_name]
                    # Handle unseen categories
                    # Fill NaN before converting to string to avoid 'nan' string values
                    if data[col_name].isnull().any():
                        # Use the first known class as default for NaN
                        if len(le.classes_) > 0:
                            nan_replacement = le.classes_[0]
                        else:
                            nan_replacement = '0'
                        data[col_name] = data[col_name].fillna(nan_replacement)
                    data[col_name] = data[col_name].astype(str)
                    unique_values = set(data[col_name].unique())
                    known_values = set(le.classes_)
                    unknown_values = unique_values - known_values
                    
                    if unknown_values:
                        # Replace unknown with most common (first class)
                        if len(le.classes_) > 0:
                            default_value = le.classes_[0]
                            data[col_name] = data[col_name].replace(list(unknown_values), default_value)
                        else:
                            # If no classes, use '0' as default
                            data[col_name] = data[col_name].replace(list(unknown_values), '0')
                    
                    # Only transform if all values are known
                    try:
                        data[col_name] = le.transform(data[col_name])
                    except ValueError as e:
                        # If transformation fails, use fit_transform with new data
                        # This updates the encoder to include new categories
                        print(f"Warning: Label encoding failed for {col_name}. Using fit_transform with new data.")
                        data[col_name] = le.fit_transform(data[col_name])
                        # Update stored encoder to reflect new fit
                        self.label_encoders[col_name] = le
                else:
                    # If encoder not found and fit=False, skip encoding
                    print(f"Warning: Encoder for '{col_name}' not found. Skipping encoding.")
        
        return data
    
    def apply_boxcox(self, data, lambda_param=0.15):
        """
        Apply Box-Cox transformation to skewed features
        
        Args:
            data: DataFrame
            lambda_param: Lambda parameter for Box-Cox transformation
        
        Returns:
            DataFrame with transformed features
        """
        data = data.copy()
        numeric_feats = data.dtypes[data.dtypes != "object"].index
        
        # Calculate skewness
        def safe_skew(x):
            x_clean = x.dropna()
            if len(x_clean) == 0:
                return 0  # Return 0 for empty series (no skew)
            return skew(x_clean)
        
        skewed_feats = data[numeric_feats].apply(safe_skew).sort_values(ascending=False)
        skewed_feats = skewed_feats[skewed_feats.abs() > 0.50]
        
        # Apply Box-Cox transformation
        features_transformed = 0
        for feat in skewed_feats.index:
            if feat in data.columns:
                # Ensure values are > -1 for boxcox1p (boxcox1p works for x > -1)
                feat_data = data[feat].copy()
                # Clip values to be > -1 (add small epsilon to avoid exactly -1)
                feat_data = np.maximum(feat_data, -0.999)
                data[feat] = boxcox1p(feat_data, lambda_param)
                features_transformed += 1
        
        print(f"Box-Cox transform uygulandı: {features_transformed} özellik")
        
        return data
    
    def create_dummy_variables(self, data):
        """
        Create dummy variables for categorical features
        
        Args:
            data: DataFrame
        
        Returns:
            DataFrame with dummy variables
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be None or empty")
        
        return pd.get_dummies(data)
    
    def scale_features(self, X_train, X_test=None, method='standard', fit=True):
        """
        Scale features
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            method: Scaling method ('standard' or 'minmax')
            fit: Whether to fit the scaler (True) or use existing scaler (False)
        
        Returns:
            Scaled features
        """
        # Validate inputs
        if X_train is None or (hasattr(X_train, '__len__') and len(X_train) == 0):
            raise ValueError("X_train cannot be None or empty")
        
        if method not in ['standard', 'minmax']:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        # Create or reuse scaler
        if fit or self.scaler is None:
            if method == 'standard':
                self.scaler = StandardScaler()
            else:
                self.scaler = MinMaxScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            # Use existing scaler
            if not hasattr(self.scaler, 'scale_'):
                raise ValueError("Scaler must be fitted before transform. Set fit=True or call fit first.")
            X_train_scaled = self.scaler.transform(X_train)
        
        if X_test is not None:
            # Validate X_test
            if hasattr(X_train, 'shape') and hasattr(X_test, 'shape'):
                if X_train.shape[1] != X_test.shape[1]:
                    raise ValueError(f"X_test must have same number of features as X_train. Got {X_test.shape[1]} and {X_train.shape[1]}")
            
            if not hasattr(self.scaler, 'scale_'):
                raise ValueError("Scaler must be fitted before transforming test data.")
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled

