# Experiment 33: final test on 2024

Every experiment so far trained on 2018–2021, chose thresholds on 2022 and compared models on 2023. Because 2023 was used to compare many models, it is no longer an untouched test. 2024 was never loaded. This experiment scores the existing models on 2024 once. Nothing is retrained or tuned.

## Fixed before any 2024 score was computed (2026-11-11)

- **Forecasts:**

  | Name | What it is | Comes from |
  |---|---|---|
  | `rule_snpp_yesterday` | S-NPP detection in the cell yesterday | Experiment 20 |
  | `rule_pair_yesterday` | S-NPP or NOAA-20 detection in the cell yesterday (the baseline) | Experiment 20 |
  | `catboost_spread_noday` | CatBoost with weather, lightning and spread inputs | Experiment 21 |
  | `mlp` | MLP | Experiment 22 |
  | `gru` | MLP + 14-day GRU | Experiment 22 |
  | `mlp_reg` | regularised MLP | Experiment 23 |
  | `gru_reg` | regularised MLP + GRU | Experiment 23 |

  The models are loaded from their experiments' saved files, unchanged.
- **Label:** S-NPP (FIRMS) or NOAA-20 (CWFIS) VIIRS detection in the cell on the target day, with one-day gaps filled; the same as the 2023 evaluation label. NOAA-21, which first appears in 2024, is not used, so the label means the same thing as in training.
- **Thresholds:** the 2022-F1 threshold per Fire Centre, model and scoring definition, recomputed from 2022 exactly as before. Second method: each day, flag as many cells as the two-satellite rule.
- **Metrics:** F1, precision and recall on the exact cell and with a one-cell margin, BC-wide and per Fire Centre. Week-block bootstrap intervals against the two-satellite rule and against Experiment 21's CatBoost.
- **Checks:** every forecast must reproduce its 2022 numbers from Experiments 21 and 22. Experiment 23 has no complete 2022 reference.

## Data for 2024 (downloaded 2026-11-11)

Each source goes to its own 2024 file; the 2018–2023 files are untouched.

| Data | File | Script |
|---|---|---|
| S-NPP detections (FIRMS) | already in `firms_h3_detections.csv` | — |
| Per-cell feature tables | `data/regions/<centre>/features_2024.parquet` | `build_features_2024.py`. Rebuilding 2023 with this script reproduces Experiment 02's tables exactly (`--check 2023`). |
| NOAA-20 (CWFIS) | `datasets/cwfis_hotspots/bc_hotspots_2024.parquet` | `scripts/download_cwfis_hotspots.py --test-year` |
| Station weather (BCWS) | `datasets/bcws_weather/station_daily_2024.parquet`, `stations_2024.parquet` | `scripts/download_bcws_weather.py --test-year` |
| Lightning (CanCPLD) | `datasets/cancpld_lightning/bc_cg_daily_2024.npz` | `scripts/download_cancpld_lightning.py --test-year` |
| BCWS incident records | `13_exp_fire_context_bc/data/bcws_incidents_2024.csv` | `13_exp_fire_context_bc/fetch_bcws_incidents.py --test-year` |

```bash
python build_features_2024.py
python run_experiment.py > run.log 2>&1
python build_report.py      # writes RESULTS.html
```

## Imported archive

Imported from `bc_labels_neural_ignitions_canadafiresat.zip` (source Experiment 19), renumbered to Experiment 33.
Folder paths and report text use this repository's numbering; internal module aliases
retain their source names for compatibility. Raw data, trained models and intermediate
results were not in the archive, so the code cannot be rerun from this folder alone.
