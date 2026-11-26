# Ames Housing V3

V3 passes the original promotion threshold on the revised experiment. On 1,168 development homes, each predicted twice outside training, the corrected V3 selection procedure scored **0.11506 RMSLE** and **$12,804.61 MAE**. The nested V2 comparator scored **0.14102** and **$16,062.17**. The conditional 95% interval for relative RMSLE reduction was **8.89% to 29.66%** (point estimate 18.41%). The mean dollar error fell by **$3,257.56** (-20.28% change, conditional 95% interval -30.80% to -12.18%).

The [three-version release comparison](../README.md#v1-v2-ve-v3-aynı-292-evde) and [interactive portfolio](https://alperdigital.github.io/ML/) report the separate comparison on 292 historical sales. Its V2 is the original five-fold CatBoost release. The nested V2 challenger below is the three-inner-fold procedure used to assess V3.

## Paired results

| Evaluation and model | RMSLE | MAE (USD) | R² |
|---|---:|---:|---:|
| Repeated outer CV · V2 nested (1,168 houses) | 0.14102 | $16,062.17 | 0.2486 |
| Repeated outer CV · V3 (1,168 houses) | 0.11506 | $12,804.61 | 0.8832 |
| Forward year 2010 · V2 nested (136 houses) | 0.11494 | $15,410.84 | 0.8936 |
| Forward year 2010 · V3 (136 houses) | 0.09516 | $11,574.92 | 0.9555 |
| Historical holdout · V2 nested (292 houses) | 0.12425 | $14,833.24 | 0.9042 |
| Historical holdout · V3 (292 houses) | 0.11810 | $12,764.41 | 0.9402 |

V2's nested selection picked CatBoost in nine outer folds and Ridge in one. In the Ridge fold, house 1299 sold for $160,000 and was estimated at $2.97 million. Keeping the family fixed, the nested CatBoost scored **0.12692 RMSLE** across the same development homes. The relative gain of V3 against this fixed-family comparison was **9.35%**. All homes remain in the primary evaluation.

The median absolute V3 error was **$7,852**, and its 90th percentile was **$26,612**. Among neighborhoods with at least 20 distinct properties, Edwards had the highest RMSLE (0.2083, 87 homes). The lowest and highest price quartiles had V3 MAE of $9,988 and $20,230. The repeated estimates describe these historical homes.

## Which changes helped?

All figures below use the same outer validation houses. Each CatBoost feature comparison changes one element at a time. A positive gain means lower RMSLE.

| Change | Before RMSLE | After RMSLE | Relative gain |
|---|---:|---:|---:|
| One-hot → native categories | 0.12723 | 0.12888 | -1.30% |
| Generic → structural absence | 0.12888 | 0.12883 | +0.04% |
| Global → neighborhood frontage median | 0.12883 | 0.12959 | -0.59% |
| Add missingness indicator (global median) | 0.12883 | 0.12978 | -0.73% |
| Add missingness indicator (neighborhood median) | 0.12959 | 0.12860 | +0.76% |

Under matched CatBoost settings, distinguishing confirmed structural absence gave a **0.04%** relative RMSLE gain. The fixed TabPFNv2 candidate scored **0.11508** using raw input columns. The comparison measures downstream price prediction. Measuring imputation accuracy would require known attribute values set aside specifically for that purpose.

| Member or blend | Outer RMSLE | MAE (USD) | R² |
|---|---:|---:|---:|
| v1_xgb | 0.12975 | $15,077.09 | 0.8381 |
| v2_cat | 0.12692 | $14,768.21 | 0.8495 |
| cand_ridge_full | 0.15184 | $17,934.63 | 0.0923 |
| cand_tabpfn_v2 | 0.11508 | $12,772.64 | 0.8845 |
| best_new | 0.11508 | $12,772.64 | 0.8845 |
| equal_blend | 0.11932 | $13,454.32 | 0.8702 |
| weighted_blend | 0.11575 | $12,828.22 | 0.8805 |

The revised rule selected standalone TabPFNv2 in **9 of 10** outer folds and the weighted blend in **1**. The original run selected them in 7 and 3 folds. Corrected V1 XGBoost and nested V2 CatBoost missed in opposite directions on 17.7% of outer predictions. Their signed log errors had correlation 0.898. The final recipe selected on all development homes is **tabpfn_v2** alone. The separately evaluated weighted blend assigned weights 0.031, 0.121, 0.848 to V1 XGBoost, nested V2 CatBoost and TabPFNv2.

## What the experiment measures

The original threshold and single-model tie rule were registered before V3 outcomes. The audit found that the earlier implementation fitted and scored blend weights on the same inner out-of-fold rows. The correction was specified after those results were public. Each inner validation fold is now split in two. Both halves receive predictions from base models trained outside the entire fold. Weights learned from the first half score the second, and vice versa. Deployment weights are fitted on all inner out-of-fold member predictions after selection. Outer folds evaluate the full selection process. [Original protocol and correction](protocol.md)

The revision reran all ten outer splits, the 2010 check and the final fit with corrected V1 calendar ages and room counts. It reused the original, unaffected V2 searches (four families × 20 trials × 12 splits), fixed-candidate inner predictions and cached TabPFN inference. The 2010 test uses 136 development homes sold after the training period. The final 292 homes were previously observed. These corrected evaluations describe previously observed sales. [Archived V3 evaluation before correction](https://github.com/alperdigital/ML/blob/1f8864c518794adbb4217ace49b246d8c18f326d/ames-v3/README.md)

The primary gate requires a pooled outer RMSLE reduction of at least 2% and a dollar MAE increase below 1%. The interval resamples the 1,168 properties 5,000 times with both predictions for each property kept together. The separate historical release comparison resamples 292 houses 20,000 times. Both intervals are conditional on the fitted models. [Paired predictions, metrics and intervals](results/three-version-holdout.json)

## Data decisions

The CatBoost and Ridge candidates under study derive total floor area, full- and half-bath equivalents, and house and remodel ages. Their missing-value recipes vary how they encode absence, impute street frontage and flag missing values. All 547 missing fireplace-quality values in the development rows coincided with zero fireplaces. The final **TabPFNv2** model receives the original **79 predictor columns** after removal of ID and target, including missing values. Its internal processing is separate from those custom missing-value and feature recipes. [Ames feature definitions](../data_description.txt) · [Model implementation](models_v3.py)

Inputs describe sales in Ames from 2006–2010, including the recorded month, sale type and condition. A prospective appraisal would require inputs known as of the prediction date and evidence from later sales. For example, historical house 524 lists a 2008 remodel against a partial sale in 2007. The benchmark retains the source record.

## Foundation model

The eighth candidate fits the original [TabPFNv2 regression model](https://huggingface.co/Prior-Labs/TabPFN-v2-reg) using training examples as context. It predicts log1p price and returns dollars after the inverse transform. The model's authors describe pretraining on synthetic tasks in their [paper](https://doi.org/10.1038/s41586-024-08328-6). We tested the newer 3.5 checkpoint on a development subset first; the official client required acceptance of its access terms. We used the publicly accessible V2 checkpoint locally. The eight-estimator configuration runs locally. The first two CPU batches used one Torch thread. A development-only eight-estimator check returned identical prices with one and four threads, so subsequent CPU batches used four threads. [Runtime and checkpoint verification](results/tabpfn-runtime.json) The complete search and inference environment uses two Python interpreters so that the V2 comparison retains its verified runtime.

## Reproduce from the repository root

The verified machine was Apple Silicon, eight logical CPU cores, 8 GiB RAM. The V2/V3 Python environment was 3.9.6. The TabPFN worker used Python 3.12.14, Torch 2.15.0 and CPU inference. On macOS, XGBoost and LightGBM use the OpenMP runtime.

```sh
python3.9 -m venv .venv-ames
.venv-ames/bin/python -m pip install -r ames-v3/requirements.txt
python3.12 -m venv .venv-pfn
.venv-pfn/bin/python -m pip install -r ames-v3/requirements-tabpfn.txt
export AMES_TABPFN_PYTHON="$PWD/.venv-pfn/bin/python"
export TABPFN_MODEL_CACHE_DIR="$PWD/.ames-v3-work/weights"
export AMES_TABPFN_PRED_CACHE="$PWD/.ames-v3-work/foundation-cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p "$TABPFN_MODEL_CACHE_DIR"
curl --fail --location --output "$TABPFN_MODEL_CACHE_DIR/tabpfn-v2-regressor.ckpt" \
  "https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/02a44705ff7790b50cded404bc59adb94da456a1/tabpfn-v2-regressor.ckpt"
.venv-ames/bin/python -m unittest discover -s ames-v3 -v
.venv-ames/bin/python ames-v3/run.py --jobs 3
.venv-ames/bin/python ames-v3/compare_releases.py
.venv-ames/bin/python ames-v3/predict.py --version v2 --holdout-id 899
.venv-ames/bin/python ames-v3/predict.py --input data/test.csv --id 1461
```

The worker verifies checkpoint SHA-256 `2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736`. Each run uses local caches under `.ames-v3-work`. Use a fresh work directory for changed code or dependencies. The prediction command refits the frozen recommended recipe on the 1,168 development rows. Use `--save-bundle` to save that local fit or `--bundle` to reuse it.

The public `--version v2` uses the published CatBoost configuration. Use `--version v2-nested` to reproduce the research challenger. Both accept the same holdout ID. For sale 899 their respective predictions are **$412,405.27** and **$456,536.36**. The 292-house test fits the public V2 once and checks each house's single-row prediction against the portfolio to the cent.

## Results and provenance

The original timing record covers the 48 cold TabPFN batches. This correction reused those predictions. Elapsed times for the full original search and the cached rerun measure different work. [Original runtime record](results/tabpfn-runtime.json) · [Per-split audit for this revision](results/split-audits.json.gz)

- [All metrics, intervals, residuals and subgroup counts](results/summary.json)
- [Two outer predictions per home](results/outer-predictions.csv.gz), [2010 predictions](results/temporal.csv) and [historical holdout predictions](results/final.csv)
- [Frozen selection and full-precision weights](results/frozen-final.json)
- [Separate input example for Kaggle house 1461](results/example.json)
- [Original V1/V2 comparison reproduced before this correction](results/reference-check.json)
