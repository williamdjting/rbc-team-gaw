# Experiment 03a: Learned Baselines with CEMS Fire Weather for BC

This experiment extends **Experiment 03** by using your joined FIRMS + CEMS dataset (`datasets/firms_with_cems_18_24.csv`) alongside the full BC CEMS grid (`datasets/dataset_CEMS_18_24.csv` for quiet/0-detection days) to add the four shortlisted Copernicus CEMS Fire Weather Index features (`fwinx`, `ffmcode`, `drtcode`, `infsinx`) onto every H3 resolution-7 cell and `history_end_date`.

Everything else from Experiment 03 is held strictly constant so you can directly measure the predictive gain of adding CEMS fire weather to the FIRMS-only baselines. You do **not** need to re-run Experiments 01–04 manually; `run_experiment.py` automatically builds the base H3 grids from `datasets/firms_with_cems_18_24.csv` if not already cached and pulls your teammate's Experiment 03 baseline metrics from `04_exp_gnn_bc/RESULTS.html`.

## Run

From the repository root (`rbc-team-gaw/`):

```bash
/Users/arshdeepkaur/wildfire_detection/.venv/bin/python experiments/03a_exp_learned_cems_bc/run_experiment.py
/Users/arshdeepkaur/wildfire_detection/.venv/bin/python experiments/03a_exp_learned_cems_bc/build_report.py
```

## Experimental Design

- **Same Regions & Grids:** All six BC Fire Centres on the exact same H3 resolution-7 grids (`234,554` total cells).
- **Same Target:** Next-day recorded FIRMS hotspot detection (`target_recorded_detection` on `target_date`).
- **Zero Temporal Leakage:** Each H3 cell receives CEMS fire weather from **`history_end_date`** ($t - 1$, strictly before `target_date`), mapped via a spatial KD-Tree to the nearest valid $0.25^\circ$ CEMS BC land cell.
- **Same Temporal Split:**
  - **Train:** `2018–2021`
  - **Validation (Threshold Selection):** `2022`
  - **Evaluation (Test):** `2023`
  - **Holdout:** `2024` is never loaded.
- **Feature Sets Compared:**
  - `cell_only_cems`: Experiment 03 `cell_only` features + `["fwinx", "ffmcode", "drtcode", "infsinx"]`
  - `cell_plus_neighbours_cems`: Experiment 03 `cell_plus_neighbours` features + `["fwinx", "ffmcode", "drtcode", "infsinx"]`
- **Models Compared:**
  - `logistic_regression`: Log-transformed counts, age, and skewed weather indices (`fwinx`, `drtcode`, `infsinx`), standardized `ffmcode`, and 6-knot spline on `day_of_year`.
  - `gradient_boosting`: `HistGradientBoostingClassifier` with the exact same fixed hyperparameters as Experiment 03.
- **Sampling & Prior Correction:** Identical to Experiment 03 (keeps all positives and 7-day active cells, samples quiet negatives at `1%`, and applies the exact $+\ln(0.01)$ log-odds shift to quiet rows at inference).

