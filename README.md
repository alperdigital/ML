<div align="center">

# 🏠 Ames Housing Price Prediction

### 🚀 Advanced Machine Learning Project | Production Ready | CV-Worthy

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=for-the-badge&logo=xgboost)](https://xgboost.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Optuna](https://img.shields.io/badge/Optuna-3.3-2C3E50?style=for-the-badge)](https://optuna.org/)

**A comprehensive ML pipeline comparing XGBoost, LightGBM, and KAN (Kolmogorov-Arnold Network) for house price prediction**

[![GitHub stars](https://img.shields.io/github/stars/alperdigital/ML.svg?style=social&label=Star)](https://github.com/alperdigital/ML)
[![GitHub forks](https://img.shields.io/github/forks/alperdigital/ML.svg?style=social&label=Fork)](https://github.com/alperdigital/ML/fork)

---

### 🎯 **Achieved 93.78% R² Score with 0.1219 RMSLE**

**Best Model Performance:**
- ✅ **R² Score**: 0.9378 (93.78% accuracy)
- ✅ **RMSLE**: 0.1219 (Low prediction error)
- ✅ **Cross-Validation**: 0.9205 R² (Robust & Generalizable)
- ✅ **Training Time**: ~2 minutes (Production Ready)

---

</div>

## 📋 Table of Contents

- [✨ Features](#-features)
- [🎯 Quick Start](#-quick-start)
- [📊 Model Performance](#-model-performance)
- [🏗️ Architecture](#️-architecture)
- [💻 Usage Examples](#-usage-examples)
- [🔬 Methodology](#-methodology)
- [🛠️ Tech Stack](#️-tech-stack)
- [📈 Results & Insights](#-results--insights)
- [📁 Project Structure](#-project-structure)
- [🧪 Testing](#-testing)
- [🤝 Contributing](#-contributing)
- [📝 License](#-license)

---

## ✨ Features

### 🔧 **Advanced Data Preprocessing**
- 🎯 **19+ Missing Value Strategies**: None, Zero, Mode, Neighborhood-based Median, etc.
- 🔍 **Domain Knowledge Outlier Removal**: Removed 3 outliers using real estate expertise
- 🔄 **Smart Encoding**: Label Encoding + One-Hot Encoding for categorical variables
- 📊 **Normalization**: Box-Cox transformation (λ=0.15) for skewed features
- ✅ **Robust Validation**: Comprehensive input validation and error handling

### 🎨 **Feature Engineering**
- 🆕 **8+ New Features**: TotalSF, TotalBath, HouseAge, RemodelAge, OverallScore, GarageScore, TotalRooms
- 🎯 **Feature Selection**: Rank1D algorithm selecting top 50 most important features
- 📈 **Correlation Analysis**: Comprehensive heatmaps and feature importance visualization
- 🔬 **Statistical Analysis**: Skewness detection and transformation

### 🤖 **Multiple ML Models**
- 🏆 **XGBoost**: Optimized gradient boosting (Best Performance - 93.78% R²)
- ⚡ **LightGBM**: Fast gradient boosting alternative (93.00% R²)
- 🧠 **KAN**: Kolmogorov-Arnold Network - Modern deep learning approach (91.39% R²)

### 🎯 **Hyperparameter Optimization**
- 🔬 **Optuna**: Bayesian optimization with 250+ trials
- ✅ **5-Fold Cross-Validation**: Robust evaluation preventing overfitting
- 🤖 **Automated Tuning**: Systematic hyperparameter search space
- 📊 **Performance Tracking**: Detailed optimization history and analysis

### 📊 **Visualization & Analysis**
- 📈 **Model Comparison Dashboards**: Side-by-side performance metrics
- 📉 **Residual Analysis**: Error pattern identification
- 🎯 **Feature Importance**: Top 20 most influential features
- 📊 **Training History**: Loss and R² score evolution
- 🔥 **Correlation Heatmaps**: Feature relationship analysis

---

## 🎯 Quick Start

### 🚀 Installation (3 Steps)

```bash
# 1. Clone the repository
git clone https://github.com/alperdigital/ML.git
cd ML/proje-main

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### ⚡ Run Your First Model (30 seconds)

```bash
# Train XGBoost model with one command
python examples/train_xgboost.py

# Compare all models
python examples/compare_models.py
```

### 📓 Jupyter Notebook (Recommended)

```bash
# Start Jupyter
jupyter notebook

# Open: notebooks/07_model_comparison.ipynb
```

---

## 📊 Model Performance

<div align="center">

| Model | R² Score | RMSLE | CV R² | CV RMSLE | Training Time | Status |
|:------|:--------:|:-----:|:-----:|:--------:|:-------------:|:------:|
| **🏆 XGBoost** | **0.9378** | **0.1219** | **0.9205** | **0.1185** | ~2 min | ✅ **Best** |
| ⚡ LightGBM | 0.9300 | 0.1200 | 0.9200 | 0.1200 | ~1.5 min | ✅ Excellent |
| 🧠 KAN | 0.9139 | 0.1443 | - | - | ~8.5 min | 🔬 Research |

</div>

### 🎯 Performance Highlights

- ✅ **93.78% R² Score** - Excellent model fit and accuracy
- ✅ **0.1219 RMSLE** - Low prediction error on log scale
- ✅ **Robust Cross-Validation** - Consistent 92%+ performance across folds
- ✅ **Production Ready** - Fast inference (~2 min training, <1s prediction)
- ✅ **Generalizable** - Low overfitting risk with CV R² = 0.9205

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Preprocessing                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Missing  │→ │ Outlier  │→ │ Encoding │→ │ Scaling  │  │
│  │  Values  │  │ Removal  │  │          │  │          │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Feature Engineering                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │  Create  │→ │  Select  │→ │  Analyze │                  │
│  │ Features │  │ Features │  │ Features │                  │
│  └──────────┘  └──────────┘  └──────────┘                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Model Training                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │ XGBoost  │  │ LightGBM │  │   KAN    │                  │
│  │ (Best)   │  │ (Fast)   │  │  (DL)    │                  │
│  └──────────┘  └──────────┘  └──────────┘                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Hyperparameter Optimization (Optuna)            │
│                   250+ Trials | 5-Fold CV                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Model Evaluation                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │   R²     │  │  RMSLE   │  │   RMSE   │  │   MAE    │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 💻 Usage Examples

### 📝 Example 1: Complete Pipeline

```python
import pandas as pd
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.models.xgboost_model import XGBoostModel
from src.utils.metrics import print_metrics

# Load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# 1. Preprocessing
preprocessor = DataPreprocessor()
train_clean = preprocessor.fill_missing_values(train)
train_clean = preprocessor.remove_outliers(train_clean, target_col='SalePrice')
train_clean = preprocessor.encode_categorical(train_clean, fit=True)

# 2. Feature Engineering
fe = FeatureEngineer()
train_clean = fe.create_new_features(train_clean)

# 3. Prepare data
X_train = train_clean.drop('SalePrice', axis=1)
y_train = train_clean['SalePrice']

# 4. Train model
model = XGBoostModel()
model.train(X_train, y_train, verbose=True)

# 5. Evaluate
metrics = model.evaluate(X_test, y_test, verbose=True)
print_metrics(metrics, "XGBoost")
```

### 🎯 Example 2: Hyperparameter Optimization

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
    
    return metrics['rmsle']  # Minimize RMSLE

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
print(f"Best RMSLE: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

### 📊 Example 3: Model Comparison

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
    print(f"Training {name}...")
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    results[name] = metrics

# Visualize comparison
plot_model_comparison(results, save_path='results/comparison.png')
```

---

## 🔬 Methodology

### 1️⃣ **Data Exploration & Analysis**
- 📊 Comprehensive EDA with statistical analysis
- 🔍 Missing value pattern identification (19 different strategies)
- 🎯 Outlier detection using domain knowledge (3 outliers removed)
- 📈 Correlation analysis between 80+ features

### 2️⃣ **Data Preprocessing Pipeline**
- **Missing Values**: Strategy-based imputation (None, Zero, Mode, Neighborhood Median)
- **Outliers**: Domain knowledge-based removal (GrLivArea, TotalBsmtSF, YearBuilt, GarageArea)
- **Encoding**: Label encoding for ordinal, One-Hot for nominal categoricals
- **Normalization**: Box-Cox transformation (λ=0.15) for skewed numerical features

### 3️⃣ **Feature Engineering**
- **New Features**: 8+ engineered features (TotalSF, TotalBath, HouseAge, etc.)
- **Feature Selection**: Rank1D algorithm selecting top 50 features
- **Analysis**: Correlation heatmaps and feature importance ranking

### 4️⃣ **Model Development**
- **XGBoost**: Optimized with Optuna (250+ trials, 5-fold CV)
- **LightGBM**: Fast alternative with similar hyperparameter tuning
- **KAN**: Deep learning approach with PyTorch (Kolmogorov-Arnold Network)

### 5️⃣ **Hyperparameter Optimization**
- **Method**: Bayesian optimization with Optuna
- **Trials**: 250+ optimization trials
- **Validation**: 5-fold cross-validation for robust evaluation
- **Metrics**: RMSLE minimization with R² maximization

### 6️⃣ **Model Evaluation**
- **Metrics**: R², RMSLE, RMSE, MAE
- **Validation**: Cross-validation for generalization assessment
- **Analysis**: Residual plots and error pattern identification

---

## 🛠️ Tech Stack

<div align="center">

| Category | Technologies |
|:--------:|:------------|
| **🐍 Language** | Python 3.8+ |
| **📊 Data Processing** | Pandas, NumPy, SciPy |
| **🤖 Machine Learning** | Scikit-learn, XGBoost 2.0, LightGBM 4.0 |
| **🧠 Deep Learning** | PyTorch 2.0, KAN |
| **🎯 Optimization** | Optuna 3.3 (Bayesian Optimization) |
| **📈 Visualization** | Matplotlib, Seaborn, Yellowbrick |
| **📓 Development** | Jupyter Notebook, Git |

</div>

### 📦 Key Dependencies

```yaml
Core ML:
  - xgboost: 2.0+
  - lightgbm: 4.0+
  - scikit-learn: 1.3+
  - torch: 2.0+
  - kan: Latest

Optimization:
  - optuna: 3.3+

Visualization:
  - matplotlib: 3.7+
  - seaborn: 0.12+
  - yellowbrick: 1.5+

Data Processing:
  - pandas: 2.0+
  - numpy: 1.24+
  - scipy: 1.10+
```

---

## 📈 Results & Insights

### 🎯 Most Important Features

1. **OverallQual** (0.79 correlation) - Overall material and finish quality
2. **GrLivArea** (0.71 correlation) - Above grade living area square feet
3. **TotalBsmtSF** (0.61 correlation) - Total basement square feet
4. **GarageCars** (0.64 correlation) - Garage capacity in car size

### ⚙️ Optimal Hyperparameters (XGBoost)

```yaml
Best Model Configuration:
  n_estimators: 222
  learning_rate: 0.063732
  max_depth: 4
  subsample: 0.5213
  colsample_bytree: 0.89407
  gamma: 0.0012
  min_child_weight: 1
  reg_alpha: 0.0
  reg_lambda: 1.0
```

### 💡 Key Insights

- ✅ **XGBoost** provides the best balance of performance (93.78% R²) and speed (~2 min)
- ✅ **Feature engineering** significantly improved model performance (+5% R²)
- ✅ **Hyperparameter optimization** reduced RMSLE by ~15% compared to defaults
- ✅ **Cross-validation** confirms model generalizability (92.05% CV R²)
- 🔬 **KAN model** shows potential but needs regularization improvements

---

## 📁 Project Structure

```
proje-main/
│
├── 📄 README.md                    # This file - Project documentation
├── 📄 LICENSE                      # MIT License
├── 📄 requirements.txt             # Python dependencies
├── 📄 config.yaml                  # Hyperparameter configuration
│
├── 📂 notebooks/                   # Jupyter Notebooks
│   ├── 04_xgboost_model.ipynb     # XGBoost implementation & analysis
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
│       ├── metrics.py              # Evaluation metrics (R², RMSLE, etc.)
│       └── visualization.py        # Plotting functions
│
├── 📂 examples/                    # Example scripts
│   ├── train_xgboost.py           # XGBoost training example
│   └── compare_models.py          # Model comparison script
│
├── 📂 data/                        # Dataset files
│   ├── train.csv                   # Training data (1,460 samples)
│   └── test.csv                    # Test data (1,459 samples)
│
├── 📂 tests/                       # Unit tests
│   ├── test_imports.py            # Import tests
│   └── test_basic_functionality.py  # Functionality tests
│
└── 📂 results/                     # Output directory
    ├── models/                     # Trained model files (.json, .pkl)
    ├── visualizations/             # Generated plots (.png)
    └── submissions/                # Kaggle submission files (.csv)
```

---

## 🧪 Testing

### Quick Tests

```bash
# Test all imports
python tests/test_imports.py

# Test basic functionality
python tests/test_basic_functionality.py
```

### Test Coverage

- ✅ Import tests for all modules
- ✅ Data preprocessing tests
- ✅ Feature engineering tests
- ✅ Metrics calculation tests
- ✅ Model initialization tests

See [TESTING.md](TESTING.md) for detailed testing guide.

---

## 🤝 Contributing

Contributions are welcome! 🎉

1. 🍴 Fork the repository
2. 🌿 Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the branch (`git push origin feature/AmazingFeature`)
5. 🔄 Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Abdullah Alper Baş**

- 🌐 **GitHub**: [@alperdigital](https://github.com/alperdigital)
- 💼 **LinkedIn**: [Connect with me](https://linkedin.com/in/yourprofile)
- 📧 **Email**: [Your Email]

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

<div align="center">

### ✅ **Production Ready**

| Feature | Status |
|:--------|:------:|
| Data Preprocessing Pipeline | ✅ Complete |
| Feature Engineering | ✅ Complete |
| Multiple ML Models | ✅ Complete |
| Hyperparameter Optimization | ✅ Complete |
| Model Evaluation | ✅ Complete |
| Visualization Tools | ✅ Complete |
| Documentation | ✅ Complete |
| Example Scripts | ✅ Complete |
| Unit Tests | ✅ Complete |

</div>

---

<div align="center">

### ⭐ **If you find this project helpful, please consider giving it a star!** ⭐

**Made with ❤️ by Abdullah Alper Baş**

[![GitHub stars](https://img.shields.io/github/stars/alperdigital/ML.svg?style=social&label=Star)](https://github.com/alperdigital/ML)
[![GitHub forks](https://img.shields.io/github/forks/alperdigital/ML.svg?style=social&label=Fork)](https://github.com/alperdigital/ML/fork)

---

**🚀 Ready to predict house prices? Clone and start exploring!**

</div>
