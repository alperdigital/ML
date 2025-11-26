# 📁 Project Structure

Detailed explanation of the project directory structure.

## Root Directory

```
proje-main/
├── 📄 README.md                    # Main project documentation
├── 📄 LICENSE                      # MIT License
├── 📄 CONTRIBUTING.md              # Contribution guidelines
├── 📄 ARCHITECTURE.md              # System architecture documentation
├── 📄 QUICKSTART.md                # Quick start guide
├── 📄 CHANGELOG.md                 # Version history
├── 📄 PROJECT_STRUCTURE.md         # This file
├── 📄 PROJECT_SUMMARY.md          # Project improvement summary
├── 📄 requirements.txt             # Python dependencies
├── 📄 config.yaml                  # Configuration file
├── 📄 setup.py                     # Package setup script
└── 📄 .gitignore                   # Git ignore rules
```

## Source Code (`src/`)

```
src/
├── __init__.py                     # Package initialization
├── data_preprocessing.py          # Data preprocessing utilities
├── feature_engineering.py         # Feature engineering utilities
│
├── models/                         # Machine learning models
│   ├── __init__.py
│   ├── xgboost_model.py          # XGBoost implementation
│   ├── lightgbm_model.py         # LightGBM implementation
│   └── kan_model.py              # KAN model implementation
│
└── utils/                          # Utility functions
    ├── __init__.py
    ├── metrics.py                 # Evaluation metrics
    └── visualization.py          # Plotting functions
```

## Notebooks (`notebooks/`)

```
notebooks/
├── 04_xgboost_model.ipynb         # XGBoost model notebook
├── 05_kan_model.ipynb             # KAN model notebook
├── 06_hyperparameter_optimization.ipynb  # Optuna optimization
└── 07_model_comparison.ipynb      # Model comparison & results
```

## Examples (`examples/`)

```
examples/
├── train_xgboost.py               # XGBoost training example
├── train_kan.py                   # KAN training example (if exists)
└── compare_models.py              # Model comparison script
```

## Data (`data/`)

```
data/
├── train.csv                      # Training dataset
└── test.csv                       # Test dataset
```

## Results (`results/`)

```
results/
├── models/                        # Trained model files
│   └── .gitkeep
├── visualizations/                # Generated plots
│   └── .gitkeep
└── submissions/                   # Kaggle submission files
    └── .gitkeep
```

## Legacy Files

The following directories contain original project files (kept for reference):

```
proje-deriniz/                     # Original project files
├── Untitled1.ipynb                # Original main notebook
├── proje-KAN.ipynb                # Original KAN notebook
├── optimizasyon katsayı.ipynb     # Original optimization notebook
└── ...
```

## File Naming Conventions

- **Python files**: `snake_case.py`
- **Notebooks**: `##_descriptive_name.ipynb` (numbered for order)
- **Config files**: `lowercase.yaml` or `lowercase.json`
- **Documentation**: `UPPERCASE.md` (README, CONTRIBUTING, etc.)

## Directory Purposes

### `src/`
Contains all reusable Python code. Organized by functionality:
- **Root level**: Main processing classes
- **models/**: Model implementations
- **utils/**: Helper functions

### `notebooks/`
Jupyter notebooks for exploration and analysis. Numbered for sequential workflow.

### `examples/`
Standalone example scripts demonstrating usage of the codebase.

### `data/`
Raw and processed data files. Should not be committed if files are large.

### `results/`
Output directory for models, visualizations, and submissions. Git-kept but typically empty in repo.

## Best Practices

1. **Keep `src/` clean**: Only production-ready code
2. **Document notebooks**: Add markdown cells explaining steps
3. **Version control**: Use `.gitignore` for large files
4. **Organize results**: Use subdirectories for different output types
5. **Maintain structure**: Follow existing patterns when adding new files

## Adding New Components

### New Model
1. Create `src/models/new_model.py`
2. Follow existing model interface
3. Add to `src/models/__init__.py`
4. Create example in `examples/`

### New Utility
1. Add to appropriate `src/utils/` file or create new one
2. Document with docstrings
3. Add to `src/utils/__init__.py`

### New Notebook
1. Number sequentially (e.g., `08_new_analysis.ipynb`)
2. Add markdown cells for documentation
3. Reference in README if significant

---

For more details, see [ARCHITECTURE.md](ARCHITECTURE.md)

