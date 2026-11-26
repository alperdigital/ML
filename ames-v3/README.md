# Ames Housing V3

The [release comparison](../README.md#v1-v2-ve-v3-aynı-292-evde) reports all three published versions on 292 matched sales. The [interactive portfolio](https://alperdigital.github.io/ML/) also shows each house's sale price alongside its three estimates.

**Use V3 for this Ames benchmark.** Each of the 1,168 development homes received two out-of-sample predictions on different training splits. On these paired predictions, V3's RMSLE was **0.11577** versus **0.14102** for V2. The relative RMSLE gain was **+17.91%**, with a conditional 95% interval of **[+8.82%, +28.50%]**. A positive gain favors V3.

**Average dollar error** fell to **$12,870.93** versus **$16,062.17** for V2. V3 reduced the average error by **$3,191.24** per sale (-19.87%, conditional 95% interval [-30.10%, -12.07%]). The registered gate requires at least 2% lower RMSLE and less than 1% MAE increase.

Among neighborhoods with at least 20 distinct homes, Edwards had the highest V3 RMSLE (0.2133, 87 houses). The bottom and top price quartiles had V3 RMSLE of 0.1462 and 0.0996, with MAE of $10,018 and $20,373. Small neighborhood samples give imprecise rankings.

## Paired results

| Evaluation / model | RMSLE | MAE (USD) | R² |
|---|---:|---:|---:|
| Nested outer CV / V2 (1168 houses) | 0.14102 | $16,062.17 | 0.2486 |
| Nested outer CV / V3 (1168 houses) | 0.11577 | $12,870.93 | 0.8807 |
| Forward year 2010 / V2 (136 houses) | 0.11494 | $15,410.84 | 0.8936 |
| Forward year 2010 / V3 (136 houses) | 0.09706 | $12,066.78 | 0.9505 |
| Historical holdout / V2 (292 houses) | 0.12425 | $14,833.24 | 0.9042 |
| Historical holdout / V3 (292 houses) | 0.11810 | $12,764.41 | 0.9402 |

For continuity, the previously published five-inner-fold V2 CatBoost scored 0.12346 RMSLE on these same 292 holdout houses. Frozen V3's 0.11810 is a 4.34% improvement against that historical configuration, or 4.95% against the nested experiment's three-inner-fold V2 fit.

V2's inner selection picked CatBoost in nine outer folds and Ridge in one. In that Ridge fold, house 1299 sold for $160,000 and was estimated at $2.97 million. House 524 sold for $184,750 and was estimated at $1.11 million. Those predictions particularly affect dollar R². Keeping the model family fixed, the inner-tuned V2 CatBoost achieved **0.12692 RMSLE** and **$14,768 MAE**. V3 still improved RMSLE by **8.79%** on the same houses. All houses remain in the primary scores.

V3's median absolute error was **$7,952** and the 90th percentile was **$26,360** across the paired outer predictions. These figures describe the historical error distribution for the development houses.

## Which changes helped?

All figures below use the same outer validation houses. Each CatBoost feature comparison changes one element at a time. A positive gain means lower RMSLE.

| Change | Before RMSLE | After RMSLE | Relative gain |
|---|---:|---:|---:|
| One-hot → native categories | 0.12723 | 0.12888 | -1.30% |
| Generic → structural absence | 0.12888 | 0.12883 | +0.04% |
| Global → neighborhood frontage median | 0.12883 | 0.12959 | -0.59% |
| Add missingness indicator (global median) | 0.12883 | 0.12978 | -0.73% |
| Add missingness indicator (neighborhood median) | 0.12959 | 0.12860 | +0.76% |

Native categorical processing alone increased RMSLE from 0.12723 to 0.12888. Distinguishing confirmed structural absence made a small difference at that setting (0.12883). The largest observed gain came from the foundation model, whose standalone outer RMSLE was 0.11508. The registered selection rule, including blends chosen in three folds, scored 0.11577.

| Member or blend | Outer RMSLE | MAE (USD) | R² |
|---|---:|---:|---:|
| v1_xgb | 0.12857 | $14,987.15 | 0.8348 |
| v2_cat | 0.12692 | $14,768.21 | 0.8495 |
| cand_ridge_full | 0.15184 | $17,934.63 | 0.0923 |
| cand_tabpfn_v2 | 0.11508 | $12,772.64 | 0.8845 |
| best_new | 0.11508 | $12,772.64 | 0.8845 |
| equal_blend | 0.11890 | $13,432.08 | 0.8693 |
| weighted_blend | 0.11576 | $12,847.65 | 0.8804 |

V1 XGBoost and tuned V2 CatBoost missed in opposite directions on 17.8% of paired outer predictions. Their signed log errors had a correlation of 0.895. The inner rule selected best_new in 7 folds, weighted_blend in 3 folds. The final V3 recipe was `best_new` and its strongest new member was `tabpfn_v2`. Optimized final weights in the order V1 XGBoost, V2 CatBoost, best new were 0.073, 0.095, 0.832. These apply to the weighted blend only.

## What the experiment measures

Ames has 1,460 labeled sales from 2006–2010. The historical seed-42 split leaves 1,168 development houses and 292 comparison houses. We reproduced the previous V1/V2 comparison byte for byte before starting. Its V2 CatBoost holdout RMSLE was 0.12346 and V1 XGBoost MAE was $14,478.22. [Reference result](results/reference-check.json)

The primary result pools 2,336 paired outer predictions, two per development property, from two independently shuffled five-fold partitions. Inside each outer training partition, all fitting and selection used three inner folds. V2 repeated its four-family Optuna search (20 trials per family), dummy baseline and engineered-feature ablation. V3 reused that complete search, added the legacy raw-dollar XGBoost, seven fixed feature/model recipes, a locally evaluated regression foundation model and three-member log-price blends. All learned preprocessing statistics, category dictionaries, model choices and weights came from the training partition. The final V2 challenger uses three inner folds here; the historical publication used five.

The primary promotion gate and tie rule were committed before V3 outcomes. RMSLE is the square root of the mean squared difference of log1p prices with nonnegative predictions. MAE and R² use prices in dollars. The paired 95% percentile interval resamples 1,168 property IDs 5,000 times and keeps both predictions for a home together. It quantifies the variation across these homes conditional on the fitted models. The year-forward test trains using development sales from 2006–2009 and evaluates 2010 development sales. The final frozen procedure was evaluated once on the previously observed historical holdout. Both are descriptive checks. [Registered protocol](protocol.md)

## Data decisions

The [Ames data dictionary](../data_description.txt) describes absent fireplaces, garages, basements and pools. We assign a distinct absence token only where physical measurements support it: zero fireplaces, both zero garage size and capacity, zero basement area, or zero pool area. We preserve a conflicting observed descriptor, count inconsistencies and use an unknown token for unresolved missing categorical values. A missing garage build year receives a sentinel only for a confirmed absent garage. In the 1,168 development houses, all 547 missing fireplace quality values coincided with zero fireplaces; 217 street-frontage values were missing. The frontage ablation compares the training-set median with the neighborhood median, falling back to the training median for unseen/all-missing neighborhoods. The missingness flag is tested independently. Each recipe includes floor area, bathroom equivalents including half baths, and house/remodel ages. Native CatBoost receives categorical labels directly. The matched one-hot and native versions use identical fixed tree parameters.

Inputs follow the competition schema, including planned/known sale month, year, sale type and sale condition. The price example is conditional on the supplied descriptors and the historical Ames market.

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
.venv-ames/bin/python ames-v3/predict.py --input data/test.csv --id 1461
```

The worker verifies checkpoint SHA-256 `2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736`. Each run uses local caches under `.ames-v3-work`. Use a fresh work directory for changed code or dependencies. The prediction command refits the frozen recommended recipe on the 1,168 development rows. Use `--save-bundle` to save that local fit or `--bundle` to reuse it.

## Files and compute

The 12 fits comprised 10 outer splits, one chronological split and one final development selection, totalling 960 completed Optuna trials. Cumulative per-split elapsed time for V2 search plus refits was 81.4 minutes. V3 shared the same searches and used 306.7 cumulative minutes including inner candidate OOF computation and refits. These sums include overlapping work on parallel splits. The 48 TabPFN inference batches took 160.99 cumulative minutes, including 206.48 seconds for the final batch of 292 houses. Local inference timings, including cache effects, are recorded in the audits.

- [All scores, group counts, uncertainty and residual diagnostics](results/summary.json)
- [Two paired OOF predictions per development house](results/outer-predictions.csv.gz)
- [Year-forward predictions](results/temporal.csv) and [historical holdout predictions](results/final.csv)
- [All split configurations and trial histories](results/split-audits.json.gz) and [final frozen configuration](results/frozen-final.json)
- [Single-house prediction](results/example.json): Kaggle test house 1461 received **$129,940.41** with the recommended V3 recipe. The competition file supplies features only.

Release verification passed all 17 V3 tests and all 5 existing V2 tests. Scores and the promotion decision were independently recalculated from the published per-house predictions. Reloading the frozen single-house estimator reproduced the example price to the cent.
