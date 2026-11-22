# Experiment 38: our models scored the CanadaFireSat way

CanadaFireSat (Porta, Dalsasso, McCarty and Tuia, [arXiv 2506.08690](https://arxiv.org/abs/2506.08690), v2 November 2025) reports **60.3% F1** for forecasting fire across Canada in 2023. Our F1 on the full BC grid is 0.03–0.12. Their F1 is measured on a **sample** of about two non-fire places per fire place, not on every place and day. This experiment applies their protocol to our saved models, with no retraining, so the two numbers can be compared directly.

## Their protocol, as applied here

- **Unit and dates:** res-7 cell (about 5 km²; their patches are 2.64 × 2.64 km, about 7 km²). Forecast dates on a fixed 8-day grid, May 8 to September 22. Each date predicts days t to t+7 from data up to t−1.
- **Labels (four):**
  - `burn_8d`: an S-NPP or NOAA-20 detection in the cell on t to t+7. This is closest to theirs ("the patch burns in the window", from NBAC burned-area polygons), and it includes existing fires spreading.
  - `ignition_8d`: a BCWS wildfire starts in the cell (Experiment 36's label, over 8 days).
  - `new_burn_8d`: a new satellite fire (Experiment 34's fire start: a detection after 7 days with none in the cell or its six neighbours).
  - `big_lightning_8d`: a lightning wildfire that grows to 10 ha or more starts there (Experiment 37's label).
- **Test set:**
  - every positive cell-date;
  - plus 2 negatives per positive, per Fire Centre and year;
  - negatives are drawn from dates between the year's first and last positive;
  - cells that burned in earlier years (satellite or BCWS) are left out, and for 2023 so are the 2022 negative cells;
  - equal numbers come from each decile of the fire weather index (FWI).
- **Test Hard (2023):** 2 negatives per positive within each Fire Centre × land cover (dominant fuel class) × month, matched to the positives' FWI deciles. Weather and land cover then cannot tell the classes apart.
- **Threshold:** maximizes F1 on a 2022 sample drawn the same way, one per model and label.
- **Repeats:** five negative draws. Tables report the mean and standard deviation.
- **Full-grid comparison:** the same cell-dates are also scored with no sampling, the way Experiments 36–37 score.

## What is scored

| Name | What it is |
|---|---|
| `exp22_ign_full` | Experiment 36's CatBoost, trained on all BCWS ignitions |
| `exp23_ign_full`, `exp23_ign_lightning` | Experiment 37's CatBoost models, trained on lightning ignitions of 10 ha or more |
| `exp16_spread` | Experiment 21's next-day satellite model |
| `exp20_start` | Experiment 34's satellite fire-start model |
| `fwi` | Ranking by station FWI (their baseline is an FWI threshold) |
| `recent_fire_nearby` | Days in the past week with a satellite detection in the cell or its neighbours |
| `lightning_14d` | Lightning in the 3x3 block over 14 days, then the 9x9 block |
| `everything_is_fire` | Flags every sample: F1 = 2p / (1 + p) at positive share p |

```bash
python run_experiment.py > run.log 2>&1   # --resume reuses finished region-year score tables
```

## Imported archive

Imported from `bc_labels_neural_ignitions_canadafiresat.zip` (source Experiment 24), renumbered to Experiment 38.
Folder paths and report text use this repository's numbering; internal module aliases
retain their source names for compatibility. Raw data, trained models and intermediate
results were not in the archive, so the code cannot be rerun from this folder alone.
