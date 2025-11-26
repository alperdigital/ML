# 🏠 Ames Housing Price Prediction - Advanced Machine Learning Project

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()

**A comprehensive machine learning project comparing traditional gradient boosting algorithms with modern deep learning approaches for house price prediction**

[Features](#-key-features) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Documentation](#-documentation)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Model Performance](#-model-performance)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Testing](#-testing)
- [Project Structure](#-project-structure)
- [Usage Examples](#-usage-examples)
- [Methodology](#-methodology)
- [Key Findings](#-key-findings)
- [Technologies](#-technologies)
- [Bug Fixes](#-bug-fixes)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

This project implements a comprehensive machine learning pipeline for predicting house prices using the **Ames Housing Dataset**. The project compares three different modeling approaches:

- **XGBoost**: Gradient boosting with optimized hyperparameters
- **LightGBM**: Fast gradient boosting framework
- **KAN (Kolmogorov-Arnold Network)**: Modern deep learning architecture

The project demonstrates advanced data preprocessing, feature engineering, hyperparameter optimization, and model evaluation techniques, achieving **R² = 0.9378** and **RMSLE = 0.1219** with the best model.

### Dataset Information

- **Training Set**: 1,460 samples with 80 features
- **Test Set**: 1,459 samples with 79 features
- **Target Variable**: SalePrice (house price in USD)
- **Source**: [Kaggle Ames Housing Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

---

## ✨ Key Features

### 🔧 Data Preprocessing
- **Missing Value Handling**: Strategy-based imputation (None, Zero, Mode, Neighborhood-based median)
- **Outlier Detection**: Domain knowledge-based outlier removal
- **Encoding**: Label Encoding and One-Hot Encoding for categorical variables
- **Normalization**: Box-Cox transformation and StandardScaler for numerical features

### 🎨 Feature Engineering
- **New Features**: TotalSF, TotalBath, HouseAge, RemodelAge, OverallScore, etc.
- **Feature Selection**: Rank1D algorithm for selecting top 50 features
- **Correlation Analysis**: Comprehensive correlation heatmaps and analysis

### 🤖 Machine Learning Models
- **XGBoost**: Optimized gradient boosting (Best Performance)
- **LightGBM**: Fast gradient boosting alternative
- **KAN**: Kolmogorov-Arnold Network for deep learning approach

### 🎯 Hyperparameter Optimization
- **Optuna**: Bayesian optimization with 250+ trials
- **Cross-Validation**: 5-fold CV for robust evaluation
- **Automated Tuning**: Systematic hyperparameter search

### 📊 Visualization & Analysis
- Model performance comparison dashboards
- Residual analysis plots
- Feature importance visualizations
- Training history plots
- Correlation heatmaps

---

## 🏆 Model Performance

| Model | R² Score | RMSLE | CV R² | CV RMSLE | Training Time | Status |
|-------|----------|-------|-------|----------|---------------|--------|
| **XGBoost** | **0.9378** | **0.1219** | **0.9205** | **0.1185** | ~2 min | ✅ **Best** |
| **LightGBM** | 0.9300 | 0.1200 | 0.9200 | 0.1200 | ~1.5 min | ✅ Good |
| **KAN** | 0.9139 | 0.1443 | - | - | ~8.5 min | ⚠️ Research |

### Performance Highlights

- ✅ **93.78% R² Score** - Excellent model fit
- ✅ **0.1219 RMSLE** - Low prediction error
- ✅ **Robust Cross-Validation** - Consistent performance across folds
- ✅ **Production Ready** - Fast inference and reliable predictions

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- 4GB+ RAM recommended

### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/alperdigital/ML.git
   cd ML/proje-main
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import pandas, xgboost, torch; print('✓ All packages installed successfully!')"
   ```

---

## ⚡ Quick Start

### Verify Installation

First, test that everything is set up correctly:

```bash
# Test imports
python tests/test_imports.py

# Test basic functionality
python tests/test_basic_functionality.py
```

### Option 1: Using Jupyter Notebooks (Recommended for Exploration)

```bash
# Start Jupyter Notebook
jupyter notebook

# Open the model comparison notebook
notebooks/07_model_comparison.ipynb
```

### Option 2: Using Python Scripts

```python
from src.data_preprocessing import DataPreprocessor
from src.models.xgboost_model import XGBoostModel
from src.utils.metrics import print_metrics

# Load and preprocess data
preprocessor = DataPreprocessor()
train_clean = preprocessor.fill_missing_values(train)
train_clean = preprocessor.remove_outliers(train_clean)

# Train model
model = XGBoostModel()
model.train(X_train, y_train)

# Evaluate
metrics = model.evaluate(X_test, y_test)
```

### Option 3: Run Example Scripts

```bash
# Train XGBoost model
python examples/train_xgboost.py

# Train KAN model
python examples/train_kan.py

# Compare all models
python examples/compare_models.py
```

---

## 📁 Project Structure

```
proje-main/
│
├── 📄 README.md                    # Project documentation (this file)
├── 📄 LICENSE                      # MIT License
├── 📄 requirements.txt             # Python dependencies
├── 📄 config.yaml                  # Hyperparameter configuration
├── 📄 .gitignore                   # Git ignore rules
│
├── 📂 notebooks/                   # Jupyter Notebooks
│   ├── 04_xgboost_model.ipynb     # XGBoost implementation
│   ├── 05_kan_model.ipynb         # KAN model implementation
│   ├── 06_hyperparameter_optimization.ipynb  # Optuna optimization
│   └── 07_model_comparison.ipynb  # Model comparison & results
│
├── 📂 src/                         # Source code (modular architecture)
│   ├── __init__.py
│   ├── data_preprocessing.py      # Data preprocessing class
│   ├── feature_engineering.py      # Feature engineering utilities
│   │
│   ├── 📂 models/                  # ML model implementations
│   │   ├── __init__.py
│   │   ├── xgboost_model.py       # XGBoost wrapper class
│   │   ├── lightgbm_model.py       # LightGBM wrapper class
│   │   └── kan_model.py           # KAN model wrapper
│   │
│   └── 📂 utils/                   # Utility functions
│       ├── __init__.py
│       ├── metrics.py              # Evaluation metrics
│       └── visualization.py        # Plotting functions
│
├── 📂 examples/                    # Example scripts
│   ├── train_xgboost.py           # XGBoost training example
│   ├── train_kan.py                # KAN training example
│   └── compare_models.py          # Model comparison script
│
├── 📂 data/                        # Dataset files
│   ├── train.csv                   # Training data
│   └── test.csv                    # Test data
│
└── 📂 results/                     # Output directory
    ├── models/                     # Trained model files
    ├── visualizations/             # Generated plots
    └── submissions/                # Kaggle submission files
```

---

## 💻 Usage Examples

### Example 1: Complete Pipeline

```python
import pandas as pd
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.models.xgboost_model import XGBoostModel
from src.utils.visualization import plot_residuals, plot_model_comparison

# Load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Preprocessing
preprocessor = DataPreprocessor()
train_clean = preprocessor.fill_missing_values(train)
train_clean = preprocessor.remove_outliers(train_clean)
train_clean = preprocessor.encode_categorical(train_clean, fit=True)

# Feature engineering
fe = FeatureEngineer()
train_clean = fe.create_new_features(train_clean)

# Prepare data
X_train = train_clean.drop('SalePrice', axis=1)
y_train = train_clean['SalePrice']

# Train model
model = XGBoostModel()
model.train(X_train, y_train)

# Evaluate
metrics = model.evaluate(X_test, y_test)
print_metrics(y_test, y_pred, "XGBoost")

# Visualize
plot_residuals(y_test, y_pred, "XGBoost")
```

### Example 2: Hyperparameter Optimization

```python
import optuna
from src.models.xgboost_model import XGBoostModel

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    }
    
    model = XGBoostModel(params=params)
    model.train(X_train, y_train)
    metrics = model.evaluate(X_val, y_val)
    
    return metrics['rmsle']

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
print(f"Best parameters: {study.best_params}")
```

### Example 3: Model Comparison

```python
from src.models.xgboost_model import XGBoostModel
from src.models.lightgbm_model import LightGBMModel
from src.utils.visualization import plot_model_comparison

# Train multiple models
models = {
    'XGBoost': XGBoostModel(),
    'LightGBM': LightGBMModel()
}

results = {}
for name, model in models.items():
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    results[name] = metrics

# Visualize comparison
plot_model_comparison(results, save_path='results/visualizations/comparison.png')
```

---

## 🔬 Methodology

### 1. Data Exploration & Analysis
- Comprehensive EDA with statistical analysis
- Missing value pattern identification
- Outlier detection using domain knowledge
- Correlation analysis between features

### 2. Data Preprocessing
- **Missing Values**: Strategy-based imputation (19 different strategies)
- **Outliers**: Removed 3 outliers based on domain rules
- **Encoding**: Label encoding for ordinal, One-Hot for nominal
- **Normalization**: Box-Cox transformation (λ=0.15) for skewed features

### 3. Feature Engineering
- Created 8+ new features (TotalSF, TotalBath, HouseAge, etc.)
- Feature selection using Rank1D (top 50 features)
- Correlation-based feature analysis

### 4. Model Development
- **XGBoost**: Optimized with Optuna (250+ trials)
- **LightGBM**: Fast alternative with similar performance
- **KAN**: Deep learning approach with PyTorch

### 5. Hyperparameter Optimization
- Bayesian optimization with Optuna
- 5-fold cross-validation for robust evaluation
- Automated hyperparameter search space

### 6. Model Evaluation
- Multiple metrics: R², RMSLE, RMSE, MAE
- Cross-validation for generalization assessment
- Residual analysis for error patterns

---

## 📊 Key Findings

### Most Important Features
1. **OverallQual** (0.79 correlation) - Overall material and finish quality
2. **GrLivArea** (0.71 correlation) - Above grade living area
3. **TotalBsmtSF** (0.61 correlation) - Total basement square feet
4. **GarageCars** (0.64 correlation) - Garage capacity

### Optimal Hyperparameters (XGBoost)
```yaml
n_estimators: 222
learning_rate: 0.063732
max_depth: 4
subsample: 0.5213
colsample_bytree: 0.89407
gamma: 0.0012
```

### Model Insights
- **XGBoost** provides the best balance of performance and speed
- **Feature engineering** significantly improved model performance
- **Hyperparameter optimization** reduced RMSLE by ~15%
- **KAN model** shows potential but needs regularization improvements

---

## 🛠️ Technologies

### Core Technologies
- **Python 3.8+** - Programming language
- **Pandas & NumPy** - Data manipulation and numerical computing
- **Scikit-learn** - Machine learning utilities
- **XGBoost 2.0** - Gradient boosting framework
- **LightGBM 4.0** - Fast gradient boosting
- **PyTorch 2.0** - Deep learning framework
- **KAN** - Kolmogorov-Arnold Network implementation

### Supporting Libraries
- **Optuna 3.3** - Hyperparameter optimization
- **Matplotlib & Seaborn** - Data visualization
- **Yellowbrick** - ML visualization tools
- **Jupyter Notebook** - Interactive development

---

## 🧪 Testing

The project includes comprehensive tests to ensure all components work correctly.

### Quick Tests

```bash
# Test all imports
python tests/test_imports.py

# Test basic functionality
python tests/test_basic_functionality.py
```

### Test Coverage

- ✅ Import tests for all modules
- ✅ Basic functionality tests
- ✅ Data preprocessing tests
- ✅ Metrics calculation tests
- ✅ Model initialization tests

See [TESTING.md](TESTING.md) for detailed testing guide.

## 🐛 Bug Fixes

All identified bugs have been fixed. See [BUG_FIXES.md](BUG_FIXES.md) for:
- List of fixed issues
- Solutions applied
- Known limitations
- Recommendations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Abdullah Alper Baş**

- GitHub: [@alperdigital](https://github.com/alperdigital)
- LinkedIn: [Connect with me](https://linkedin.com/in/yourprofile)
- Email: [Your Email]

---

## 🙏 Acknowledgments

- **Kaggle** for providing the Ames Housing Dataset
- **KAN Authors** for the Kolmogorov-Arnold Network paper and implementation
- **Optuna Team** for the excellent hyperparameter optimization framework
- **XGBoost & LightGBM** developers for the powerful ML libraries

---

## 📚 References

1. [Ames Housing Dataset - Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
2. [KAN: Kolmogorov-Arnold Networks - arXiv](https://arxiv.org/abs/2404.19756)
3. [XGBoost Documentation](https://xgboost.readthedocs.io/)
4. [Optuna Documentation](https://optuna.readthedocs.io/)

---

## 📈 Project Status

✅ **Production Ready** - All core features implemented and tested

- [x] Data preprocessing pipeline
- [x] Feature engineering utilities
- [x] Multiple ML models (XGBoost, LightGBM, KAN)
- [x] Hyperparameter optimization
- [x] Model evaluation and comparison
- [x] Visualization tools
- [x] Documentation

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star! ⭐**

Made with ❤️ by Abdullah Alper Baş

</div>
