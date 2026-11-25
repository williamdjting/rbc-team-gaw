# Experiment 39: new-fire-only model with class-imbalance handling

**Why.** Mentor suggestion (Sam, Nov 25, 2026):
- Train a model that predicts **only new fires**, so it can't fall back on "fire today = fire tomorrow".
- Handle the extreme class imbalance with a weighted loss (GBDT), focal loss (MLP), or 2 non-fires per fire.
- Judge it with the watch list and the **top-k most confident predictions** (what % are real fires).
- Do this before adding new data.

## Setup
- **Rows:** quiet cells only, meaning no S-NPP or NOAA-20 detection in the cell or its 6 neighbours for 7 days.
  - Training: 2018–2021, from `data/pair_lightning_h3r7/training_samples` (every quiet fire plus 1% of quiet non-fires).
  - Test: every quiet cell-day of 2023 (33.8M; 5,332 new fires, about 1 in 6,000).
- **Label:** v1 (S-NPP or NOAA-20 detection, 1-day gaps filled).
- **Inputs:** current data only.
  - 15 fire-history inputs and 6 lightning inputs (Exp 25).
  - 8 jump features: distance to, size and FRP of fires up to about 24 km away (Exp 32).

## Models

| Name | Model | Imbalance handling |
|---|---|---|
| Q_pop | CatBoost | none: population weights (sampled non-fires × 100) |
| Q_wce | CatBoost | weighted cross-entropy: fires up-weighted so both classes carry equal total weight |
| Q_1to2 | CatBoost | trained on 2 random non-fires per fire (CanadaFireSat-style) |
| Q_focal | MLP 256/256/128 | focal loss (γ = 2, α = 0.75) |
| Reference | Exp 32 stacked + jumps | general model on all cells (current best watch list) |

CatBoost uses learning rate 0.05, depth 6 and seed 42. Rounds are picked by early stopping on 2021 PR-AUC, then the
model is refit on 2018–2021.

## Evaluation (2023)
- Average precision on quiet cells.
- **Watch list:** top N quiet cells per Fire Centre per day (N = 25, 50, 100, 200). Reports the share of new fires on
  the list the day before, the share of listed cells that burned, and results by group: genuine ignition, jump of an
  older BCWS fire, other burn.
- **Top-k most confident:** BC-wide per day (k = 10, 50, 100, 500, 1,000) and over the whole season (k = 100 to
  50,000). Reports the % that are real new fires.
- **Total F1 if routed:** quiet cells decided by the new model and all other cells by the best stacked model.
  Thresholds are chosen on 2022.
- **Bootstrap:** week bootstrap (2,000 resamples) against the reference.

## Run
```powershell
.venv/forecast/Scripts/python.exe -I experiments/39_exp_new_fire_only_bc/run_experiment.py .
```
- **Prerequisites:** `data/pair_lightning_h3r7/` (Exp 32 `build_pair_table.py`), Exp 32 `jumps/results/scores/`,
  and FIRMS/BCWS inputs as for Exp 32.
- **Outputs:** `results/summary.csv` and `results/metrics.json` (committed). Models and score files are git-ignored.
- **Results:** [RESULTS.md](RESULTS.md).
