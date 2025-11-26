# 🚀 Quick Start Guide

Get up and running with the Ames Housing Price Prediction project in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/alperdigital/ML.git
cd ML/proje-main

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Quick Start Options

### Option 1: Run Example Script (Fastest)

```bash
# Train XGBoost model
python examples/train_xgboost.py

# Compare all models
python examples/compare_models.py
```

### Option 2: Use Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open model comparison notebook
notebooks/07_model_comparison.ipynb
```

### Option 3: Use Python API

```python
from src.models.xgboost_model import XGBoostModel
from src.data_preprocessing import DataPreprocessor
import pandas as pd

# Load data
train = pd.read_csv('data/train.csv')

# Preprocess
preprocessor = DataPreprocessor()
X_train = preprocessor.fill_missing_values(train.drop('SalePrice', axis=1))
y_train = train['SalePrice']

# Train model
model = XGBoostModel()
model.train(X_train, y_train)

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(f"R² Score: {metrics['r2']:.4f}")
```

## Expected Results

After running the training script, you should see:

```
============================================================
XGBoost Model Training - Ames Housing Price Prediction
============================================================

[1/5] Loading data...
✓ Training data: (1460, 80)
✓ Test data: (1459, 79)

[2/5] Preprocessing data...
✓ Preprocessing completed

[3/5] Feature engineering...
✓ Feature engineering completed

[4/5] Training XGBoost model...
XGBoost modeli eğitiliyor...
✓ Model eğitimi tamamlandı!

[5/5] Evaluating model...
============================================================
XGBoost Performans Metrikleri
============================================================
R² Score:     0.93780
RMSLE:        0.12190
RMSE:         23670.02
MAE:          18000.00
MSE:          560000000.00
============================================================
```

## Next Steps

1. **Explore Notebooks**: Check out the detailed notebooks in `notebooks/`
2. **Try Different Models**: Experiment with LightGBM and KAN models
3. **Optimize Hyperparameters**: Run the Optuna optimization notebook
4. **Read Documentation**: See [README.md](README.md) for full documentation

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the project root directory
cd /path/to/proje-main

# Verify installation
python -c "import pandas, xgboost; print('OK')"
```

### Data Not Found
```bash
# Ensure data files are in the data/ directory
ls data/train.csv data/test.csv
```

### Memory Issues
- Reduce dataset size for testing
- Use smaller models for initial testing
- Close other applications

## Need Help?

- Check [README.md](README.md) for detailed documentation
- Review [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- Open an issue on GitHub for bugs or questions

---

**Happy Coding! 🎉**

