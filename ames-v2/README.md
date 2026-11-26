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

No V1 score comparisons are claimed because its split and evaluation protocol differ.

## V1 vs V2 — same-split comparison (26 November 2026)

This is a new, retrospective comparison, not a juxtaposition of scores from the old README. Both versions use the same raw Ames dataset (SHA-256 recorded in the JSON), identical ordered development/holdout row indices (1,168/292, seed 42), the same five shuffled development folds, and the same raw-dollar metrics: RMSLE, R² and MAE. Lower RMSLE and MAE are better; higher R² is better. Each version's representative model is selected by **mean development CV RMSLE**, never by holdout scores.

### Results

| Version | Model | CV RMSLE | Holdout RMSLE | Holdout R² | Holdout MAE (USD) |
|---|---|---:|---:|---:|---:|
| V1 | XGBoost | 0.12655 | 0.13188 | **0.92344** | **14,478.22** |
| V1 | LightGBM (V1 CV winner) | 0.12236 | 0.13832 | 0.92181 | 15,515.46 |
| V2 | Dummy | 0.39007 | 0.43324 | -0.01583 | 59,931.24 |
| V2 | Ridge | 0.14132 | 0.13829 | 0.91310 | 16,730.05 |
| V2 | XGBoost | 0.12469 | 0.13369 | 0.89169 | 15,674.02 |
| V2 | LightGBM | 0.12518 | 0.13444 | 0.88251 | 16,118.42 |
| V2 | CatBoost (V2 CV winner) | **0.11909** | **0.12346** | 0.89969 | 14,993.51 |

### Version-level differences (CV-selected models)

- V2 CatBoost improves mean CV RMSLE by **2.68%** and holdout RMSLE by **10.74%** versus V1 LightGBM.
- V2 reduces holdout MAE by **$521.96**, but its R² is **0.02211 lower**. V1 XGBoost has the best holdout R² and MAE across the displayed models. No universal V2 improvement is claimed.
- Matched-model results are mixed: V2 XGBoost improves CV RMSLE by 1.47% but worsens holdout RMSLE by 1.37%; V2 LightGBM worsens CV RMSLE by 2.31% but improves holdout RMSLE by 2.81%.
- The difference reflects the entire pipeline, model selection and tuning, not one isolated preprocessing technique.

### Reproducibility and scope

```sh
python -m pip install -r ames-v2/requirements.txt
python -m pip install yellowbrick==1.5
python ames-v2/compare_v1_v2.py --data data/train.csv
python -m unittest discover -s ames-v2 -v
```

The [comparison harness](compare_v1_v2.py) uses the V1 preprocessing, engineered features and XGBoost/LightGBM wrappers from [the unchanged V1 source commit](https://github.com/alperdigital/ML/commit/a1cf99b304c604125019a10b587613a5d3b1ee5c). The V1 executable example originally preprocesses and removes outliers before splitting and leaves nominal strings unencoded; the harness splits first, learns imputation statistics only from training rows, encodes/aligns dummy columns using the V1 helpers and keeps the V1 outlier rules local to training (1–2 removals per fold, 2 on the full development split). All 292 holdout rows are kept. V1 retains its original raw-dollar target, ID feature and historical default hyperparameters (only thread/verbosity settings changed). It does not apply Box-Cox or Rank1D because the executable tree comparison example does not use those stages.

V1's experimental KAN model is not part of the executable `examples/compare_models.py` and was not benchmarked here. Its historical notebook score is not treated as a comparable result. The V1 comparison therefore covers the original two executable tree baselines, not every V1 notebook variant. Historical V1 hyperparameters may have seen these data during earlier experiments; V1's tuning budget was not repeated or matched to V2's 20 trials/model, and neither version has a newly blinded external test set. V2's selected configurations were fixed from the preceding 20-trial run and their same-fold scores and CatBoost holdout metrics were reproduced. The additional holdout model rows are descriptive and were not used to select a winner.

See [machine-readable results](v1-v2-comparison.json) for every model, fold-level RMSLE, training row counts, split fingerprints, parameters and dependency versions. Five unit tests passed.
