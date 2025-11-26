# 🧪 Testing Guide

This guide explains how to test the project to ensure everything works correctly.

## Quick Test

Run the import test to verify all modules can be imported:

```bash
python tests/test_imports.py
```

Run basic functionality tests:

```bash
python tests/test_basic_functionality.py
```

## Test Scripts

### 1. `tests/test_imports.py`

Tests that all modules can be imported without errors.

**What it tests:**
- DataPreprocessor import
- FeatureEngineer import
- All model imports (XGBoost, LightGBM, KAN)
- Utility imports (metrics, visualization)

**Expected output:**
```
Testing imports...
============================================================
✓ DataPreprocessor imported successfully
✓ FeatureEngineer imported successfully
✓ XGBoostModel imported successfully
✓ LightGBMModel imported successfully
✓ KANModel imported successfully
✓ Metrics utilities imported successfully
✓ Visualization utilities imported successfully
============================================================

✅ All imports successful!
```

### 2. `tests/test_basic_functionality.py`

Tests core functionality without requiring the full dataset.

**What it tests:**
- DataPreprocessor: Missing value filling, encoding
- Metrics: Calculation functions
- XGBoostModel: Initialization
- Visualization: Style setup

**Expected output:**
```
============================================================
Running Basic Functionality Tests
============================================================
Testing DataPreprocessor...
  ✓ fill_missing_values works
  ✓ encode_categorical works

Testing metrics...
  ✓ calculate_metrics works

Testing XGBoostModel initialization...
  ✓ XGBoostModel initialization works

Testing visualization setup...
  ✓ setup_plot_style works

============================================================
Tests passed: 4/4
✅ All tests passed!
```

## Manual Testing

### Test Data Preprocessing

```python
from src.data_preprocessing import DataPreprocessor
import pandas as pd
import numpy as np

# Create test data
data = pd.DataFrame({
    'numeric': [1, 2, np.nan, 4],
    'categorical': ['A', 'B', np.nan, 'A']
})

preprocessor = DataPreprocessor()
result = preprocessor.fill_missing_values(data)
print(result)
```

### Test Model Initialization

```python
from src.models.xgboost_model import XGBoostModel

model = XGBoostModel()
print(model.params)
```

### Test Metrics

```python
from src.utils.metrics import calculate_metrics
import numpy as np

y_true = np.array([100, 200, 300])
y_pred = np.array([110, 190, 310])

metrics = calculate_metrics(y_true, y_pred)
print(metrics)
```

## Full Pipeline Test

To test the full pipeline with actual data:

1. Ensure data files are in `data/` directory:
   ```bash
   ls data/train.csv data/test.csv
   ```

2. Run the example script:
   ```bash
   python examples/train_xgboost.py
   ```

3. Check for errors and verify outputs:
   - Model should train without errors
   - Metrics should be printed
   - Visualizations should be saved
   - Submission file should be created

## Common Issues and Solutions

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'src'`

**Solution**: 
- Make sure you're in the project root directory
- Add project root to Python path:
  ```python
  import sys
  import os
  sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  ```

### Missing Dependencies

**Problem**: `ImportError: No module named 'xgboost'`

**Solution**:
```bash
pip install -r requirements.txt
```

### Data Not Found

**Problem**: `FileNotFoundError: data/train.csv`

**Solution**:
- Ensure data files are in the `data/` directory
- Check file paths in scripts

### KAN Import Warning

**Problem**: `ImportError: No module named 'kan'`

**Solution**:
- KAN is optional for basic functionality
- Install with: `pip install pykan` or `pip install kan`
- XGBoost and LightGBM work without KAN

## Continuous Testing

For development, you can set up automated testing:

```bash
# Watch mode (requires pytest-watch)
pytest-watch tests/

# Coverage report
pytest --cov=src tests/
```

## Test Coverage Goals

- [x] Import tests
- [x] Basic functionality tests
- [ ] Unit tests for each class
- [ ] Integration tests
- [ ] End-to-end pipeline tests

## Reporting Issues

If you find bugs during testing:

1. Check [BUG_FIXES.md](BUG_FIXES.md) for known issues
2. Create a detailed bug report:
   - What you were testing
   - Expected behavior
   - Actual behavior
   - Error messages
   - Steps to reproduce

---

**Last Updated**: 2025

