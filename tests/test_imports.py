"""
Test script to verify all imports work correctly
Run this to check if there are any import errors
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_imports():
    """Test all module imports"""
    errors = []
    
    print("Testing imports...")
    print("="*60)
    
    # Test data preprocessing
    try:
        from src.data_preprocessing import DataPreprocessor
        print("✓ DataPreprocessor imported successfully")
    except Exception as e:
        errors.append(f"DataPreprocessor: {e}")
        print(f"✗ DataPreprocessor import failed: {e}")
    
    # Test feature engineering
    try:
        from src.feature_engineering import FeatureEngineer
        print("✓ FeatureEngineer imported successfully")
    except Exception as e:
        errors.append(f"FeatureEngineer: {e}")
        print(f"✗ FeatureEngineer import failed: {e}")
    
    # Test models
    try:
        from src.models.xgboost_model import XGBoostModel
        print("✓ XGBoostModel imported successfully")
    except Exception as e:
        errors.append(f"XGBoostModel: {e}")
        print(f"✗ XGBoostModel import failed: {e}")
    
    try:
        from src.models.lightgbm_model import LightGBMModel
        print("✓ LightGBMModel imported successfully")
    except Exception as e:
        errors.append(f"LightGBMModel: {e}")
        print(f"✗ LightGBMModel import failed: {e}")
    
    try:
        from src.models.kan_model import KANModel
        print("✓ KANModel imported successfully")
    except Exception as e:
        errors.append(f"KANModel: {e}")
        print(f"✗ KANModel import failed: {e}")
    
    # Test utils
    try:
        from src.utils.metrics import calculate_metrics, print_metrics
        print("✓ Metrics utilities imported successfully")
    except Exception as e:
        errors.append(f"Metrics: {e}")
        print(f"✗ Metrics import failed: {e}")
    
    try:
        from src.utils.visualization import (
            plot_model_comparison,
            plot_residuals,
            setup_plot_style
        )
        print("✓ Visualization utilities imported successfully")
    except Exception as e:
        errors.append(f"Visualization: {e}")
        print(f"✗ Visualization import failed: {e}")
    
    print("="*60)
    
    if errors:
        print(f"\n❌ {len(errors)} import error(s) found:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\n✅ All imports successful!")
        return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)

