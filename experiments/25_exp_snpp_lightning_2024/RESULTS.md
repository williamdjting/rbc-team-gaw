# Experiment 25: S-NPP + lightning results

The 2024 evaluation is complete. This is a separately trained S-NPP/lightning
MLP, not the earlier two-satellite model. No 2024 threshold tuning was performed.

## Data and method

- Inputs: S-NPP FIRMS history and official CanCPLD cloud-to-ground lightning,
  downloaded for 2018-2024. No NOAA-20, weather or incident inputs.
- 21 features, plus missing flags; model architecture 256/256/128, seed 42.
- Target: next-day S-NPP detection, with one-day label gaps filled. This measures
  hotspot activity, not verified ignitions or confirmed fire absence.
- All six existing BC H3-resolution-7 grids, May 8-September 30.
- Training 2018-2021: 1,761,958 sampled rows;
  1,510,565 fitting rows, 251,393 held-out training-season
  rows for early stopping. Best epoch: 6.
- Thresholds selected once on full-grid 2022; 2023 used as a development check.
- Final 2024 test: 34,244,884 cell-days, 20,082 positive labels.
- Frozen UTC: 2026-11-12T00:30:47Z; completed UTC: 2026-11-12T00:35:57Z.

## Full-grid, pooled exact-cell results

| Year | Model | F1 | Precision | Recall | Median centre AP |
|---|---|---:|---:|---:|---:|
| 2023 | mlp_snpp_lightning | 0.5813 | 0.6401 | 0.5324 | 0.5910 |
| 2023 | snpp_persistence | 0.5878 | 0.6087 | 0.5684 | 0.3474 |
| 2024 | mlp_snpp_lightning | 0.5586 | 0.6298 | 0.5018 | 0.4765 |
| 2024 | snpp_persistence | 0.5670 | 0.5856 | 0.5496 | 0.3201 |

F1 is not accuracy. AP is the median of six per-centre average precision values,
not pooled provincial AP. All evaluation negatives are retained.

2024 MLP minus persistence F1: **-0.0084**;
paired whole-week bootstrap 95% interval **[-0.0184, +0.0023]**
from 2,000 resamples. This interval reflects within-year variation, not uncertainty
across future fire seasons or training seeds.

## 2024 by Fire Centre

| Fire Centre | MLP F1 | Persistence F1 | MLP precision | MLP recall | MLP AP |
|---|---:|---:|---:|---:|---:|
| Cariboo | 0.4010 | 0.3606 | 0.5031 | 0.3333 | 0.3239 |
| Coastal | 0.4105 | 0.4321 | 0.4144 | 0.4067 | 0.3134 |
| Kamloops | 0.5890 | 0.5812 | 0.6874 | 0.5152 | 0.5722 |
| Northwest | 0.5779 | 0.5730 | 0.6129 | 0.5468 | 0.5585 |
| Prince George | 0.5605 | 0.5717 | 0.6373 | 0.5002 | 0.5188 |
| Southeast | 0.5405 | 0.5586 | 0.6045 | 0.4888 | 0.4343 |

## Interpretation and limits

The MLP has higher precision but lower recall than persistence on 2024. Its pooled
F1 is slightly lower, and the paired confidence interval includes zero. This run
does not establish an overall improvement over the simple persistence rule.

Compare with the S-NPP persistence baseline in this table. The earlier 0.626 MLP
result used a different label, feature set and evaluation year, so it is not a
like-for-like comparator. No S-NPP-only model ablation was run here, so these
results do not isolate the added value of lightning. One seed and one fixed
architecture were evaluated. Missing satellite detections remain uncertain
negatives; lightning coverage gaps remain missing features.

No hyperparameters or thresholds were adjusted after seeing 2024. The 2024
holdout has now been used for this experiment and should not be reused as an
untouched test during further model selection.

## Files

- `results/metrics_2024.csv`: per-centre metrics and frozen thresholds.
- `results/summary_2024.csv`: pooled performance.
- `results/bootstrap_2024.json`: paired uncertainty interval.
- `results/frozen_before_2024.json`: model/scaler/source hashes and protocol.
- `results/test_completion.json`: completed test record.
- `data/snpp_lightning_h3r7/evaluation/2024/`: full-grid joined features,
  target, MLP score and persistence prediction, partitioned by Fire Centre.

Lightning source: [official CanCPLD directory](https://crd-data-donnees-rdc.ec.gc.ca/CDAS/products/CanCPLD/CLDN/cg/).

## Follow-up: lightning contribution

A matched retraining ablation is now available in [Experiment 26](../26_exp_lightning_ablation/RESULTS.md). On 2024, lightning increases F1 from 0.5547 to 0.5586 (+0.385 percentage points), with a paired 95% interval including zero. This follow-up used an already-inspected year and does not change the original persistence comparison above.
