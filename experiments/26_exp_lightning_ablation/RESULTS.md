# Experiment 26: how much does lightning contribute?

**The 2024 F1 difference is not clearly distinguishable from zero in this matched run.**

Adding lightning changes 2024 pooled F1 from **0.5547 to 0.5586**:
**+0.385 F1 percentage points**, or +0.69% relative.
The paired 95% interval is [-0.0022, +0.0099] in F1 units.

## Matched design

Both models use the same S-NPP-only, one-day-gap-filled activity label, existing
H3-resolution-7 grids, and full-grid evaluation rows. Training data, row order,
non-lightning scaling, architecture, random initialization (seed 42), optimizer,
batch order and early-stopping policy are held fixed. The ablation is retrained
from scratch with all six lightning features and their missing flags set to zero
after scaling throughout training and inference. It is not inference-only masking.

The original with-lightning checkpoint and predictions from Experiment 25 are
unchanged. The retained scaler statistics were independently checked against
the identical training sample. Perturbing all lightning fields leaves the
ablated network input exactly unchanged.

Training: 2018-2021 (1,510,565 fitting and 251,393
held-out training-season rows). The no-lightning model selects epoch
6; the original with-lightning model selected epoch 6.
Each model selects its own thresholds on 2022 using the same per-centre F1
criterion. No 2023/2024 threshold tuning is performed.

## Full-grid comparison

| Year | Model | F1 | Precision | Recall | Median centre AP |
|---|---|---:|---:|---:|---:|
| 2023 | without lightning | 0.5833 | 0.6271 | 0.5452 | 0.5790 |
| 2023 | with lightning | 0.5813 | 0.6401 | 0.5324 | 0.5910 |
| 2024 | without lightning | 0.5547 | 0.6127 | 0.5068 | 0.4674 |
| 2024 | with lightning | 0.5586 | 0.6298 | 0.5018 | 0.4765 |

| Year | F1 gain from lightning | Relative F1 change | Paired 95% interval (F1 points) | AP change |
|---|---:|---:|---|---:|
| 2023 | -0.201 points | -0.35% | [-0.891, +0.486] | +0.0120 |
| 2024 | +0.385 points | +0.69% | [-0.216, +0.994] | +0.0091 |

F1 points above mean 100 times the difference in F1, not classification accuracy.
AP is the median per-centre average precision, not pooled AP. AP does not depend
on the selected classification threshold.

At the frozen operating points, adding lightning changes 2024 true positives by
**-100**, false positives by **-511**, and missed positive
cell-days by **+100**. These are cell-days, not distinct fires.

## Regional F1, 2024

| Fire Centre | F1 without lightning | F1 with lightning | Difference |
|---|---:|---:|---:|
| Cariboo | 0.3717 | 0.4010 | +0.0293 |
| Coastal | 0.4303 | 0.4105 | -0.0198 |
| Kamloops | 0.5742 | 0.5890 | +0.0148 |
| Northwest | 0.5709 | 0.5779 | +0.0071 |
| Prince George | 0.5575 | 0.5605 | +0.0030 |
| Southeast | 0.5479 | 0.5405 | -0.0074 |

## Interpretation limits

This ablation was requested after the original 2024 results had been inspected.
It is a follow-up analysis on an already-used test year, not a fresh independent
holdout. No settings were tuned from the new 2023/2024 results.

Intervals pair both models on the same 2,000 whole-ISO-week resamples. They measure
within-year variability, not uncertainty from training seeds or future seasons.
One matched seed is insufficient to establish a universal contribution. The
measured difference includes each model's training-based epoch choice and
2022-selected operating point. These results do not estimate a causal effect of
lightning on wildfire ignition.

This comparison addresses lightning's contribution to this MLP. Whether the
complete model beats persistence is a separate comparison; see
[Experiment 25](../25_exp_snpp_lightning_2024/RESULTS.md).

## Artifacts

`results/metrics_2023.csv`, `metrics_2024.csv`, `summary_*.csv`, and
`bootstrap_*.csv` contain the comparisons. `frozen_before_followup_evaluation.json`
records hashes and thresholds; `completion.json` confirms source artifacts were
unchanged and the original Experiment 25 metrics were reproduced. Ablated score
arrays preserve the row order of their corresponding Experiment 25 full-grid
Parquet files, whose hashes are frozen.
