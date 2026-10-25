# CatBoost instructions

Use these instructions to run two CatBoost ablations against the existing learned baselines. Create these folders first so the experiment numbering stays consistent:

```text
experiments/05_catboost_full_data/
experiments/06_catboost_undersampled/
```

The repository contains the definitions; the regional Parquet files are supplied separately in a ZIP.

## Setup

```bash
git clone https://github.com/williamdjting/rbc-team-gaw.git
cd rbc-team-gaw
pip install pandas numpy pyarrow scikit-learn catboost joblib
```

Unzip the supplied data into `experiments/02_hotspot_persistence_bc/regions/`. The structure should contain `cariboo`, `coastal`, `kamloops`, `northwest`, `prince_george`, and `southeast`, each with `data/features/features_2018.parquet` through `features_2023.parquet`. Do not modify the Parquet files.

## Split and target

Train on **2018–2021**, select thresholds/settings on **2022**, and evaluate on **2023**. Do not load 2024.

`target_recorded_detection` is 1 when at least one FIRMS detection was recorded in the cell on the target day, and 0 when none was recorded. It is not an ignition, unique-fire, severity, or verified-absence label.

## Configurations

Train both `cell_only` and `cell_plus_neighbours`.

Cell-only features:

```text
local_count_1d, local_count_3d, local_count_7d,
days_since_previous_detection, history_left_censored, day_of_year, quiet_stratum
```

Cell-plus-neighbours adds:

```text
neighbour_count_1d, neighbour_count_3d, neighbour_count_7d, in_region_neighbours
```

All features must use dates before the target date.

## Experiment 05: full-data CatBoost

Use every 2018–2021 training row. Do not undersample and do not apply the `0.01`
quiet-row odds correction. Run two configurations: unweighted CatBoost and
CatBoost with `auto_class_weights="Balanced"` (or an explicitly documented
`class_weights` value). This tests whether weighting the full dataset helps.

Save this experiment's outputs inside `experiments/05_catboost_full_data/`.

## Experiment 06: undersampled CatBoost

Use the undersampling procedure below. This is the fair comparison with
Experiments 03–04, which also use quiet-negative undersampling.

Save this experiment's outputs inside `experiments/06_catboost_undersampled/`.

## Class imbalance for Experiment 06

Quiet/active describes the previous seven days; positive/negative describes the target day.

| Category | Previous seven days | Target day |
| --- | --- | --- |
| Active positive | Some recorded activity | Detection recorded |
| Active negative | Some recorded activity | No detection recorded |
| Quiet positive | No recorded activity | Detection recorded |
| Quiet negative | No recorded activity | No detection recorded |

For cell-only, quiet is based on the cell. For cell-plus-neighbours, quiet requires zero activity in the cell and its immediate in-region neighbours.

1. Keep every positive row.
2. Keep every active row, including active negatives.
3. Randomly retain approximately 1% of quiet negatives only.
4. Train CatBoost without class weights.
5. Evaluate on every 2022 and 2023 row; do not undersample evaluation data.

Save `results/catboost_class_balance_before_after.csv` with:

```text
configuration,category,before_count,after_count,before_percent,after_percent
```

It must contain four categories for each configuration. Check that the four counts sum to the full training rows before sampling and to the rows passed to CatBoost afterward. Print both before and after summaries.

## Score correction for Experiment 06

Because only 1% of quiet negatives are retained, quiet examples look more positive in the training sample. For every quiet row at prediction time, correct the positive odds:

```python
odds = p_positive / (1 - p_positive)
corrected_odds = odds * 0.01
corrected_probability = corrected_odds / (1 + corrected_odds)
```

Do not multiply probability directly by `0.01`. Active-row scores are unchanged. This is a sampling correction, not separately fitted probability calibration.

## Thresholds and outputs

For each configuration and Fire Centre, predict all 2022 rows after correction, select the threshold maximizing F1, then apply it unchanged to 2023. Report average precision, precision, recall, F1, threshold, TP, FP, FN, and TN.

Save the outputs separately in the relevant experiment folder:

```text
results/catboost_metrics_2023.csv
results/catboost_predictions_2023.parquet
results/catboost_class_balance_before_after.csv
results/catboost_manifest.json
```

For Experiment 05, also save separate outputs for unweighted and weighted CatBoost.
For Experiment 06, include the before/after four-category class-balance table.
Record the CatBoost version, seed, features, class-handling method, source hashes,
sampling counts or class weights, threshold, and runtime. Compare Experiments 05
and 06 against Experiments 03 and 04 using the same 2023 rows and metrics.
Interpret positives as recorded hotspot activity, not confirmed ignitions.
