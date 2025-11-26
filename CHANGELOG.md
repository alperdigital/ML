# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-XX

### Added
- Initial release of Ames Housing Price Prediction project
- Comprehensive data preprocessing pipeline
- Feature engineering utilities
- Three ML models: XGBoost, LightGBM, and KAN
- Hyperparameter optimization with Optuna
- Model comparison and visualization tools
- Complete documentation (README, CONTRIBUTING, ARCHITECTURE)
- Example scripts for training and comparison
- Professional project structure

### Features
- **Data Preprocessing**: Strategy-based missing value imputation, outlier removal, encoding
- **Feature Engineering**: New feature creation, feature selection, scaling
- **Models**: XGBoost (R²=0.9378), LightGBM (R²=0.93), KAN (R²=0.9139)
- **Optimization**: Optuna-based hyperparameter tuning
- **Visualization**: Model comparison, residual analysis, feature importance

### Performance
- Best model: XGBoost with R² = 0.9378 and RMSLE = 0.1219
- Cross-validation: Consistent performance across folds
- Production ready: Fast inference and reliable predictions

## [Unreleased]

### Planned
- Ensemble model implementation
- KAN model regularization improvements
- Additional visualization features
- Unit test suite
- CI/CD pipeline

