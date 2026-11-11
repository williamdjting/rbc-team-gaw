# Experiment 22: deep learning models against CatBoost

Experiments 03–16 used gradient-boosted trees (and, in Experiment 04, a small GNN and MLP on fire history only). Experiment 21's CatBoost, given weather, lightning and upwind spread and no `day_of_year`, beat the two-satellite rule with the one-cell margin (0.824 vs 0.810) but not on the exact cell (0.607 vs 0.624). This experiment asks whether a different learner gets more out of the same information:

| Model | What it is |
|---|---|
| `catboost_spread_noday` | Experiment 21's model, loaded, not retrained: the reference to beat (must reproduce Experiment 21) |
| `mlp` | three hidden layers (256, 256, 128) on the same inputs, signed-log + standardized, missing flags |
| `gru` | the MLP plus a 64-unit GRU over each cell's last 14 days (25 daily channels: fire, NOAA-20, weather, lightning, upwind fire) |

Label, training draw, thresholds (2022-tuned per Fire Centre, or the rule's daily volume), evaluation on 2023 and the week-block bootstrap are Experiment 21's. Bootstraps compare every model with both the two-satellite rule and Experiment 21's CatBoost. Neural models hold out one 7-day block in six of each training season to choose the epoch.

Needs torch (CPU is enough; about 1–1.5 h for the full run) plus the datasets used by Experiment 21.

```bash
python run_experiment.py --regions Kamloops --max-epochs 2 --results /tmp/exp22_smoke   # smoke test
python run_experiment.py > run.log 2>&1
python build_report.py                                                                    # writes RESULTS.html
```

2024 is never loaded.

## Imported archive

Imported from `../latest.zip` (source Experiment 17), renumbered to Experiment 22.
Folder paths and report text use this repository's numbering; internal module aliases
retain their source names for compatibility. Saved reports, metrics and models are
included. Rerunning also requires Experiment 21's weather and lightning inputs; see
its README for the missing preprocessing dependencies.
