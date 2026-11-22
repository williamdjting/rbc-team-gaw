# Experiment 35: fire starts with an MLP and an MLP + GRU

Experiment 34 trained CatBoost on fire starts only. This experiment trains Experiment 23's regularised MLP and MLP + 14-day GRU on the same fire starts, to see whether a network ranks the next fire start better than the trees.

## What is reused

- **From Experiment 34:**
  - the fire-start definition and the quiet cell-days;
  - the training rows: every fire start of 2018–2021 plus the 1% draw of other quiet cell-days, same seeds;
  - the scoring function itself (thresholds from 2022, 2023 evaluation, watch lists, onset groups).
- **From Experiment 23:**
  - the network layers: 256-256-128, plus a 64-unit GRU over the past 14 days for the MLP + GRU;
  - the training settings: learning rate 1e-4, dropout 0.3, AdamW weight decay 0.1 on weights, up to 40 epochs, early stopping with patience 5 on held-out weeks of the training seasons.
- **From Experiment 22:** the input scaling method (signed log, standardise, clip, missing flags) and the 14-day sequence channels.
- **Inputs:** Experiment 21's `spread_noday` inputs, the inputs of Experiment 34's best fire-start model.

## What changes

- The input scaling is refit on the fire-start rows.
- The output bias starts at the training base rate (about 0.4% fire starts) instead of 50/50, so the first epochs are not spent learning that starts are rare.

## Forecasts

| Name | What it is |
|---|---|
| `start_mlp` | MLP trained on fire starts (new) |
| `start_gru` | MLP + 14-day GRU trained on fire starts (new) |
| `start_drivers` | Experiment 34's fire-start CatBoost, loaded |
| `catboost_spread_noday` | Experiment 21's next-day CatBoost, loaded |
| `mlp` | Experiment 22's next-day MLP, loaded |
| `nearby_activity`, `lightning`, `random` | Experiment 34's simple rankings |

Experiment 34's two CatBoosts and simple rankings must reproduce Experiment 34's numbers exactly. Bootstrap intervals compare every forecast with Experiment 34's fire-start CatBoost and with Experiment 21's model.

```bash
python run_experiment.py > run.log 2>&1   # --resume reloads the networks and finished Fire Centres after an interruption
python build_report.py                    # writes RESULTS.html
```

## Imported archive

Imported from `bc_labels_neural_ignitions_canadafiresat.zip` (source Experiment 21), renumbered to Experiment 35.
Folder paths and report text use this repository's numbering; internal module aliases
retain their source names for compatibility. Raw data, trained models and intermediate
results were not in the archive, so the code cannot be rerun from this folder alone.
