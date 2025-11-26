"""
Basic functionality tests
Tests core functionality without requiring full dataset
"""

import sys
import os
import numpy as np
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_data_preprocessor():
    """Test DataPreprocessor basic functionality"""
    print("Testing DataPreprocessor...")
    try:
        from src.data_preprocessing import DataPreprocessor
        
        # Create sample data
        data = pd.DataFrame({
            'numeric1': [1, 2, np.nan, 4, 5],
            'numeric2': [10, 20, 30, np.nan, 50],
            'categorical': ['A', 'B', np.nan, 'A', 'B'],
            'PoolQC': [np.nan, np.nan, 'Gd', np.nan, 'TA']
        })
        
        preprocessor = DataPreprocessor()
        result = preprocessor.fill_missing_values(data)
        
        # Check that no NaN values remain (except in categorical if strategy is None)
        assert result['numeric1'].isnull().sum() == 0, "Numeric columns should have no NaN"
        assert result['numeric2'].isnull().sum() == 0, "Numeric columns should have no NaN"
        
        print("  ✓ fill_missing_values works")
        
        # Test encoding
        data_cat = pd.DataFrame({
            'quality': ['Ex', 'Gd', 'TA', 'Fa'],
            'condition': ['Norm', 'Norm', 'Abnorml', 'Norm']
        })
        encoded = preprocessor.encode_categorical(data_cat, fit=True)
        assert encoded['quality'].dtype in [np.int64, np.int32, int], "Encoding should convert to numeric"
        print("  ✓ encode_categorical works")
        
        return True
    except Exception as e:
        print(f"  ✗ DataPreprocessor test failed: {e}")
        return False

def test_metrics():
    """Test metrics calculation"""
    print("Testing metrics...")
    try:
        from src.utils.metrics import calculate_metrics
        
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert 'r2' in metrics
        assert 'rmsle' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        
        print("  ✓ calculate_metrics works")
        return True
    except Exception as e:
        print(f"  ✗ Metrics test failed: {e}")
        return False

def test_xgboost_model_init():
    """Test XGBoost model initialization"""
    print("Testing XGBoostModel initialization...")
    try:
        from src.models.xgboost_model import XGBoostModel
        
        model = XGBoostModel()
        assert model.params is not None
        assert 'n_estimators' in model.params
        assert 'learning_rate' in model.params
        
        print("  ✓ XGBoostModel initialization works")
        return True
    except Exception as e:
        print(f"  ✗ XGBoostModel test failed: {e}")
        return False

def test_visualization_setup():
    """Test visualization setup"""
    print("Testing visualization setup...")
    try:
        from src.utils.visualization import setup_plot_style
        
        setup_plot_style()
        print("  ✓ setup_plot_style works")
        return True
    except Exception as e:
        print(f"  ✗ Visualization test failed: {e}")
        return False

def run_all_tests():
    """Run all basic tests"""
    print("="*60)
    print("Running Basic Functionality Tests")
    print("="*60)
    
    tests = [
        test_data_preprocessor,
        test_metrics,
        test_xgboost_model_init,
        test_visualization_setup
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Test {test.__name__} crashed: {e}")
            results.append(False)
        print()
    
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

