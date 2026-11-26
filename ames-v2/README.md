# Ames Housing V2

An independent, reproducible benchmark. The [V1 source](https://github.com/alperdigital/ML/tree/a1cf99b304c604125019a10b587613a5d3b1ee5c) preserves the original experiment. This branch also includes its corrected retrospective comparison.

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
- Seeded Optuna TPE search minimizes mean development-fold RMSLE. Engineered features are ablated at the same tuned parameters. The published V2 feature named `TotalBath` counts full bathrooms above ground and in the basement. Half baths enter the model through their separate source columns. V1 and the V3 feature-engineered candidates use half-bath equivalents in their derived total.
- JSON reports individual fold scores, dispersion, selected config, dataset SHA-256, environment versions and holdout metrics. CV used for tuning is selection-biased; no claim of nested CV.

## Verification status

`results-ridge.json` records the initial true-data smoke benchmark (1,460 rows, 1,168 development, 292 holdout, 5 folds, no tuning). Ridge with no engineered features had CV RMSLE 0.15006 and holdout RMSLE 0.13085, R² 0.92654 and MAE $15,663.18. This is not a full four-model benchmark. The preliminary holdout was observed but was not used to tune or alter the candidate search.

The earlier /mnt/data prototype was unavailable. The V2 code was independently rewritten after reviewing the V1 README and repository assessment.

## Full-model benchmark (26 November 2026)

Actual V1 training data (1,460 rows; 1,168 development and 292 holdout). The recommended 20 Optuna trials per model were completed for all four models (80 total), with five-fold development CV and the same seed and split as the preliminary run.

| Model | CV RMSLE, engineered | CV RMSLE, no engineered features |
|---|---:|---:|
| dummy | 0.39007 | — |
| ridge | 0.14154 | 0.14132 |
| xgboost | 0.12469 | 0.12577 |
| lightgbm | 0.12518 | 0.12665 |
| catboost | 0.11909 | 0.12141 |

Selected by development CV: **catboost** (engineered=True; CV RMSLE **0.11909**). Best parameters: `{"iterations": 545, "depth": 4, "learning_rate": 0.05243180891902853}`.

| Metric | 3 trials/model | 20 trials/model |
|---|---:|---:|
| Best development CV RMSLE | 0.12012 | 0.11909 |
| Holdout RMSLE | 0.13018 | 0.12346 |
| Holdout R² | 0.89192 | 0.89969 |
| Holdout MAE (USD) | 15,344.18 | 14,993.51 |

Holdout RMSLE decreased by **5.17%** relative to the 3-trial run. The same holdout was previously observed; it was not used for tuning or candidate selection, and this is not a new external validation set.

- [Latest results — 20 trials/model](results.json): individual fold scores, winning configurations, dataset fingerprint and package versions.
- [Optuna trial history](optuna-trials-20.json): all 80 completed trial scores and parameters.
- [Archived 3-trial benchmark](results-3-trials.json).
- Three unit tests passed again. Each model's 20 trial entries and best objective were verified against the benchmark output.

The historical V1 notebook scores used a different split. The corrected comparison below scores all candidates on the same sales.

## V1 vs V2 — same-split comparison (26 November 2026)

This retrospective run uses the corrected V1 baseline from this branch. Both versions use the same raw Ames dataset (SHA-256 recorded in the JSON), identical ordered development/holdout row indices (1,168/292, seed 42), the same five shuffled development folds, and the same raw-dollar metrics: RMSLE, R² and MAE. Lower RMSLE and MAE are better; higher R² is better. Each version's representative model is selected by **mean development CV RMSLE**, never by holdout scores.

### Results

| Version | Model | CV RMSLE | Holdout RMSLE | Holdout R² | Holdout MAE (USD) |
|---|---|---:|---:|---:|---:|
| V1 (corrected) | XGBoost | 0.12690 | 0.13518 | 0.92347 | **14,581.96** |
| V1 (corrected) | LightGBM (V1 CV winner) | 0.12243 | 0.13601 | **0.92592** | 15,047.65 |
| V2 | Dummy | 0.39007 | 0.43324 | -0.01583 | 59,931.24 |
| V2 | Ridge | 0.14132 | 0.13829 | 0.91310 | 16,730.05 |
| V2 | XGBoost | 0.12469 | 0.13369 | 0.89169 | 15,674.02 |
| V2 | LightGBM | 0.12518 | 0.13444 | 0.88251 | 16,118.42 |
| V2 | CatBoost (V2 CV winner) | **0.11909** | **0.12346** | 0.89969 | 14,993.51 |

### Version-level differences (CV-selected models)

- V2 CatBoost reduced mean CV RMSLE by **2.73%** and holdout RMSLE by **9.23%** versus corrected V1 LightGBM.
- V2's holdout MAE fell by **$54.15** and R² fell by **0.02623**. Among the candidates shown, corrected V1 LightGBM has the highest dollar R² and corrected V1 XGBoost has the lowest dollar MAE.
- Holding the model family fixed, V2 XGBoost improved holdout RMSLE by 1.10% and V2 LightGBM by 1.16%. These comparisons combine changes to preprocessing and fitting.

### Reproducibility and scope

```sh
python -m pip install -r ames-v2/requirements.txt
python -m pip install yellowbrick==1.5
python ames-v2/compare_v1_v2.py --data data/train.csv
python -m unittest discover -s ames-v2 -v
```

The corrected V1 baseline retains the raw-dollar target, ID input, outlier criteria, tree wrappers and original model parameters. The examples and comparison now share one training-only feature pipeline. It splits the raw sales first, learns missing-value statistics, applies price-based exclusions only to the training rows (two on the full development split), fits encoders and aligns nominal indicators to the training columns. Sale year, month and condition score retain their numeric meanings. Age is measured in calendar years and the total room count already includes bedrooms. All 292 holdout rows are scored. The [original source](https://github.com/alperdigital/ML/tree/a1cf99b304c604125019a10b587613a5d3b1ee5c) and [earlier comparison](https://github.com/alperdigital/ML/blob/7473f585685b15cf886daa32041f8f293e887d1e/ames-v2/v1-v2-comparison.json) remain available for provenance.

V1's experimental KAN model is not part of the executable `examples/compare_models.py` and was not benchmarked here. Its historical notebook score is not treated as a comparable result. The V1 comparison therefore covers the original two executable tree baselines, not every V1 notebook variant. Historical V1 hyperparameters may have seen these data during earlier experiments; V1's tuning budget was not repeated or matched to V2's 20 trials/model, and neither version has a newly blinded external test set. V2's selected configurations were fixed from the preceding 20-trial run and their same-fold scores and CatBoost holdout metrics were reproduced. The additional holdout model rows are descriptive and were not used to select a winner.

See [machine-readable results](v1-v2-comparison.json) for every model, fold-level RMSLE, training row counts, split fingerprints, parameters and dependency versions.
