# Experiment 23: regularised networks

In Experiment 22 the MLP beat Experiment 21's CatBoost on the exact cell (BC-wide 2023 F1 0.626 vs 0.607) and tied the two-satellite rule. With the one-cell margin it was slightly behind CatBoost (0.820 vs 0.824). Both networks reached their best held-out loss after one epoch, then overfit the training seasons. This experiment retrains them more slowly and with stronger regularisation:

| Setting | Experiment 22 | Here |
|---|---|---|
| Learning rate | 1e-3 | 1e-4 |
| Dropout | 0.1 after the first two layers | 0.3 after every layer and on the GRU summary |
| AdamW weight decay | 1e-4, all parameters | 0.1, weights only (not biases) |
| Epochs (max, patience) | 20, 3 | 40, 5 |

Everything else is Experiment 22's: label, training draw, inputs, input scaling (rebuilt from Experiment 22's manifest), layer sizes, thresholds, scoring and bootstrap. One seed per model.

| Model | What it is |
|---|---|
| `catboost_spread_noday` | Experiment 21's model, loaded (must reproduce Experiment 21) |
| `mlp` | Experiment 22's MLP, loaded (must reproduce Experiment 22) |
| `mlp_reg` | the MLP with the settings above |
| `gru_reg` | the MLP + 14-day GRU with the settings above |

Needs torch and Experiment 22's `results/` (manifest and `models/mlp.pt`).

```bash
python run_experiment.py --regions Kamloops --max-epochs 2 --results /tmp/exp23_smoke   # smoke test
python run_experiment.py > run.log 2>&1
python build_report.py                                                                    # writes RESULTS.html
```

2024 is never loaded.

## Imported archive

Imported from `../latest.zip` (source Experiment 18), renumbered to Experiment 23.
Folder paths and report text use this repository's numbering; internal module aliases
retain their source names for compatibility.

The archive includes the standalone `RESULTS.html`, figures, model checkpoints,
training sample and training curves, but omits `results/manifest.json`,
`metrics_by_year.csv`, `summary_2023.csv`, `bootstrap_2023.csv`,
`new_vs_continuing_2023.csv`, and `fit_checks.csv`. The supplied report can be viewed,
but `build_report.py` cannot regenerate it until those original outputs are restored
or the experiment is rerun. Rerunning also requires Experiment 21's missing weather
and lightning preprocessing inputs.
