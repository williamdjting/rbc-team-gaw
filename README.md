# rbc-team-gaw
Wildfire hotspot forecasting project for British Columbia.

## Project layout

```text
docs/
  hotspot-forecasting-design.md   forecasting definition and model plan
  roadmap.md                       project schedule and responsibilities
  week-1.md ... week-8.md          weekly progress notes

experiments/
  firms_summer_2018_2024/          FIRMS preprocessing and H3 transformation
  01_hotspot_persistence_kamloops/ single-region persistence pilot
  02_hotspot_persistence_bc/       persistence benchmark across six Fire Centres
  03_exp_learned_baselines_bc/     logistic regression and gradient boosting
  04_exp_gnn_bc/                   GNN and MLP comparison
  EXPERIMENTS_OVERVIEW.html        concise comparison of Experiments 01–04

notebooks/
  01_region_weather_exploration.ipynb
  02_h3_firms_exploration.ipynb
```

Generated regional tables, model files, figures, caches, raw archives, and local
working notes are ignored by Git. The experiment reports are standalone HTML files
with figures embedded, so they can be opened without the generated datasets.

## Workflow

1. Prepare the FIRMS summer data and assign each detection to an H3 resolution-7 cell in `experiments/firms_summer_2018_2024/`.
2. Run the persistence workflow in `01_hotspot_persistence_kamloops/`.
3. Repeat it across all six Fire Centres with `02_hotspot_persistence_bc/`.
4. Fit learned cell-only and neighbour-feature baselines with `03_exp_learned_baselines_bc/`.
5. Train the GNN and MLP comparison with `04_exp_gnn_bc/`.

The experiments use 2018–2021 for training or diagnostics, 2022 for rule,
threshold, or checkpoint selection, and 2023 for development evaluation. 2024 is
reserved for a later final test and is not loaded by the current experiments.

The target is **next-day recorded hotspot activity**: whether at least one FIRMS
detection is recorded in a cell. It is not a verified ignition date, unique-fire
label, severity measure, or confirmed absence of fire. Observation coverage remains
unknown, so the current zero class means “no recorded detection.”

Class imbalance is handled only in the learned experiments. They keep all positives
and recent-activity rows, retain a random 1% of quiet negatives during training,
and correct quiet-row positive odds by 0.01 at prediction. Persistence experiments
use all cell-days without sampling or class weights.

For the current results, open [EXPERIMENTS_OVERVIEW.html](experiments/EXPERIMENTS_OVERVIEW.html).
