# Ames Housing V2

An independent, reproducible benchmark. V1 remains untouched in the original repository tree and main branch.

## Run

From the repository root (Python 3.9+; 3.11 recommended):

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r ames-v2/requirements.txt
python -m unittest discover -s ames-v2 -v
python ames-v2/benchmark.py --data data/train.csv --trials 20
```

macOS may require the OpenMP runtime (`brew install libomp`) for XGBoost/LightGBM.

Use `--trials 0` for defaults or `--models ridge` for a lightweight run. Output defaults to `ames-v2/results.json`. The Kaggle test set has no labels and is not used for evaluation.

## Method

- Fixed 80/20 split (seed 42), 5 shuffled development folds. Only the CV-selected configuration is evaluated on the held-out 20%.
- Imputation, missingness indicators, numeric scaling and categorical encoding are fitted inside each fold. No target-dependent row removal, ID leakage, or train/test concatenation.
- Log1p target training, inverse transformation to dollars, nonnegative prediction clipping and RMSLE calculated on raw-dollar labels and predictions. R² is not accuracy.
- Dummy mean-log-price baseline, Ridge, XGBoost, LightGBM, CatBoost; one-hot categorical treatment for comparability.
- Seeded Optuna TPE search minimizes mean development-fold RMSLE. Engineered features are ablated at the same tuned parameters (not independently retuned).
- JSON reports individual fold scores, dispersion, selected config, dataset SHA-256, environment versions and holdout metrics. CV used for tuning is selection-biased; no claim of nested CV.

## Verification status

`results-ridge.json` records the initial true-data smoke benchmark (1,460 rows, 1,168 development, 292 holdout, 5 folds, no tuning). Ridge with no engineered features had CV RMSLE 0.15006 and holdout RMSLE 0.13085, R² 0.92654 and MAE $15,663.18. This is not a full four-model benchmark. The preliminary holdout was observed but was not used to tune or alter the candidate search.

The earlier /mnt/data prototype was unavailable. The V2 code was independently rewritten after reviewing the V1 README and repository assessment.

## Full-model benchmark (26 November 2026)

Actual V1 training data; 3 Optuna trials per model, 5 folds. This is a small search budget, not the recommended 20-trial run.

| Model | CV RMSLE, engineered | CV RMSLE, no engineered features |
|---|---:|---:|
| dummy | 0.39007 | — |
| ridge | 0.14212 | 0.14179 |
| xgboost | 0.12749 | 0.13070 |
| lightgbm | 0.12742 | 0.13064 |
| catboost | 0.12012 | 0.12253 |

Selected by development CV: **catboost** (engineered=True). Holdout RMSLE **0.13018**, R² **0.89192**, MAE **$15,344.18**. See [results.json](results.json) for fold-level scores, parameters and package versions. Three unit tests passed. No V1 score comparisons are claimed because its split and evaluation protocol differ.
