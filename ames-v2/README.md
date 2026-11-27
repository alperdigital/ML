# Ames Housing V2

V2 asks how well four model families predict the same sales when they receive identical training folds, feature definitions and search budgets. This page reports the corrected V2 procedure evaluated on the `v3` branch. The [archived V2 branch](https://github.com/alperdigital/ames-housing-price-prediction/tree/7473f585685b15cf886daa32041f8f293e887d1e) preserves the original result. The [checkpoint comparison](../README.md) and the current V2 command use the corrected five-fold winner. The live portfolio displays the preceding completed three-version release while V3 is paused.

## From the CSV to a prediction

1. Verify the source file and fix the original seed-42 split. The 1,460 labeled sales comprise 1,168 development sales and the same 292 holdout sales used by V1 and V3. The Kaggle test file contains inputs only. House IDs are excluded from model features.
2. Read the category literals before fitting any transformations. `MasVnrType=None` is an observed value meaning no masonry veneer (864 sales). The eight `NA` values have unknown veneer type. The source contains 677 confirmed absences and six unknowns in development, 187 and two respectively in the holdout. Previously the CSV parser combined both cases. These counts describe that column alone. [Data dictionary](../data_description.txt) · [CSV reader](../src/ames_data.py)
3. Fit median numeric imputation, missing-value indicators and numeric scaling using only the training part of each fold. Encode categorical values with one-hot indicators and an explicit missing category. The corrected V2 model can learn separate indicators for confirmed masonry absence and an unknown type. Known absence and a genuinely unobserved value can carry different pricing information.
4. Train a mean-log-price reference, Ridge, XGBoost, LightGBM and CatBoost. Keep the same categorical encoding across the four non-dummy families. Compare engineered versus original features at the same tuned hyperparameters. Engineered features include total area, full bathrooms above ground plus in the basement, and ages at sale. Half baths are also available as separate inputs. V1 and V3’s feature-engineered candidates include half-bath equivalents in their derived totals.
5. Use five shuffled development folds, 20 seeded Optuna TPE trials per non-dummy family and mean fold RMSLE. These folds tune and select models, so their winning score can be optimistic. Map the training target with `log1p` and invert predictions with `expm1` before comparing prices. RMSLE pairs one nonnegative target with one predicted price and applies `log1p` once per price. The scoring code accepts vectors and single-column tables, rejects incompatible shapes and nonfinite prices, and floors negative predictions at zero. MAE is measured in dollars. R² is the fraction of variation in dollar sale prices explained against a mean-price reference. [Scorer](../src/ames_metrics.py)
6. Choose the model, hyperparameters and feature setting by development CV. Refit on all development rows. Report the 292-sale holdout once for the selected comparison. Separately, the V1/V2 retrospective also reports other candidates descriptively. This historical holdout was already examined in previous releases.

## Search and feature ablation

| Model | Five-fold RMSLE, engineered | Five-fold RMSLE, original |
|---|---:|---:|
| Dummy | 0.39007 | — |
| Ridge | 0.14154 | 0.14132 |
| XGBoost | 0.12455 | 0.12858 |
| LightGBM | 0.12474 | 0.12634 |
| CatBoost | 0.11951 | 0.12203 |

The five-fold development winner was **CatBoost** with engineered features and mean RMSLE **0.11951**. It used 932 iterations, depth 3 and a learning rate of 0.04964482136657893. [All configurations and fold scores](results.json) · [80 completed Optuna trials](optuna-trials-20.json)

| Same 292-sale holdout | 3 trials per family | 20 trials per family |
|---|---:|---:|
| RMSLE | 0.12878 | 0.12451 |
| MAE in dollars | 15,304.13 | 15,042.20 |
| R² | 0.90456 | 0.90390 |

The three-trial run used the same corrected data interpretation and evaluation rules. The untuned Ridge smoke benchmark produced holdout RMSLE 0.13085, MAE $15,663.18 and R² 0.92654. [3-trial run](results-3-trials.json) · [Ridge smoke run](results-ridge.json)

## V1 and V2 on matched training and evaluation rows

The corrected V1 trees retain their original hyperparameters and raw-dollar target. V1 excludes the record ID, separates no veneer from an unknown type and assigns code −1 to unseen values in label-encoded columns. A garage with a quality grade absent from training now has a distinct code. The one-hot columns for nominal categories follow the training schema. Outlier rules use labels only in training and retain 1,166 of the 1,168 development sales for the final V1 fit. Calendar years and the recorded room count are used for age and room features. The two V1 trees and each V2 model's CV-selected settings were evaluated on identical folds, and both versions' representatives were chosen by development RMSLE. The other models’ holdout results below are descriptive.

| Version | Family | Development CV RMSLE | Holdout RMSLE | MAE (USD) | R² |
|---|---|---:|---:|---:|---:|
| V1 | XGBoost | 0.12906 | 0.13050 | 14,204.40 | 0.92648 |
| V1 | LightGBM **(CV winner)** | 0.12471 | 0.13679 | 15,225.01 | 0.92109 |
| V2 | Dummy | 0.39007 | 0.43324 | 59,931.24 | -0.01583 |
| V2 | Ridge | 0.14132 | 0.13829 | 16,730.14 | 0.91310 |
| V2 | XGBoost | 0.12455 | 0.13650 | 15,825.58 | 0.90108 |
| V2 | LightGBM | 0.12474 | 0.13476 | 16,114.17 | 0.88072 |
| V2 | CatBoost **(CV winner)** | 0.11951 | 0.12451 | 15,042.20 | 0.90390 |

Moving from V1 LightGBM to V2 CatBoost, holdout RMSLE changed by **-8.98%**, MAE by **-182.81 USD** and R² by **-0.01719**. The comparison changes multiple decisions together, including the training target, preprocessing, search budget and selected family. It measures the complete procedures on these historical sales. V1’s original hyperparameters may reflect earlier work with these prices. V1’s KAN notebook remained outside its executable two-tree comparison. [Original V1 source](https://github.com/alperdigital/ames-housing-price-prediction/tree/a1cf99b304c604125019a10b587613a5d3b1ee5c) · [Matched metric summary, parameters and split fingerprints](v1-v2-comparison.json)

## Reproduce the measurements

For the V2 benchmark alone, install [the V2 requirements](requirements.txt). The two-tree retrospective also needs Yellowbrick 1.5. For the complete comparison, use the dependency versions and supported environment in the [V3 setup instructions](../DEVAM.md). Run the following commands from the repository root.

```sh
python ames-v2/benchmark.py --models ridge --trials 0 --output ames-v2/results-ridge.json
python ames-v2/benchmark.py --trials 3 --output ames-v2/results-3-trials.json
python ames-v2/benchmark.py --trials 20 --output ames-v2/results.json --trial-history ames-v2/optuna-trials-20.json
python ames-v2/compare_v1_v2.py --data data/train.csv
python -m unittest discover -s ames-v2 -v
python ames-v3/predict.py --version v2 --holdout-id 899
```

The original V2 implementation was rewritten from the V1 project after the earlier scratch prototype was unavailable. The current `--version v2` uses this revised five-fold winner. The portfolio links its previous completed comparison to the matching source revision. `--version v2-nested` uses the independently retuned V2 challenger inside the V3 experiment. Historical bundles encoded veneer values differently. Refit from the current configuration to create a compatible bundle.
