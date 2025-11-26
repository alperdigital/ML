# 🏗️ Project Architecture

This document describes the architecture and design decisions of the Ames Housing Price Prediction project.

## System Overview

The project follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Pipeline                              │
├─────────────────────────────────────────────────────────────┤
│  Data Loading → Preprocessing → Feature Engineering → Models │
└─────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Data Preprocessing Layer

**File**: `src/data_preprocessing.py`

**Responsibilities**:
- Missing value imputation
- Outlier detection and removal
- Categorical encoding
- Data normalization

**Key Classes**:
- `DataPreprocessor`: Main preprocessing class with strategy-based approach

**Design Pattern**: Strategy Pattern for missing value handling

### 2. Feature Engineering Layer

**File**: `src/feature_engineering.py`

**Responsibilities**:
- Creating new features
- Feature selection
- Feature scaling

**Key Classes**:
- `FeatureEngineer`: Feature engineering utilities

### 3. Model Layer

**Files**: `src/models/*.py`

**Responsibilities**:
- Model training
- Prediction
- Model persistence

**Key Classes**:
- `XGBoostModel`: XGBoost wrapper
- `LightGBMModel`: LightGBM wrapper
- `KANModel`: KAN model wrapper

**Design Pattern**: Wrapper Pattern for consistent model interface

### 4. Utilities Layer

**Files**: `src/utils/*.py`

**Responsibilities**:
- Metrics calculation
- Visualization
- Helper functions

## Data Flow

```
Raw Data (CSV)
    ↓
DataPreprocessor
    ├─→ Missing Value Imputation
    ├─→ Outlier Removal
    └─→ Encoding
    ↓
FeatureEngineer
    ├─→ New Feature Creation
    ├─→ Feature Selection
    └─→ Scaling
    ↓
Model Training
    ├─→ XGBoost
    ├─→ LightGBM
    └─→ KAN
    ↓
Evaluation & Visualization
```

## Design Principles

1. **Modularity**: Each component is independent and testable
2. **Reusability**: Common functionality is abstracted into utilities
3. **Extensibility**: Easy to add new models or preprocessing steps
4. **Maintainability**: Clear code structure and documentation

## Extension Points

### Adding a New Model

1. Create a new class in `src/models/`
2. Inherit from a base model interface (or follow existing pattern)
3. Implement `train()`, `predict()`, and `evaluate()` methods
4. Add to `src/models/__init__.py`

### Adding a New Preprocessing Step

1. Add method to `DataPreprocessor` class
2. Update `_default_strategies()` if needed
3. Document the new preprocessing step

## Performance Considerations

- **Memory**: Data is processed in chunks where possible
- **Speed**: Caching of encoders and scalers
- **Scalability**: Modular design allows for parallel processing

## Testing Strategy

- Unit tests for each component
- Integration tests for the full pipeline
- Performance benchmarks for model comparison

