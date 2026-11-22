# Experiment 34: predicting fire starts

The next-day models mostly learn that a burning cell keeps burning. This experiment keeps only the start of each fire and asks whether a model trained on starts alone can say where the next one will appear.

## Definition

A **fire start** on day t is a cell that:
- is positive on the label: an S-NPP or NOAA-20 VIIRS detection from 2021, S-NPP before 2021 (no complete NOAA-20 record), with one-day gaps filled;
- had no detection from the same source in the cell or its six neighbours on days t-7 to t-1.

The cell-day must also be in Experiment 13's quiet stratum (no S-NPP count in the cell or its neighbours over the 7 days). The two definitions differ on a few cell-days at Fire Centre borders (90 of 2.6 million in Southeast 2021, none of them fire starts). Before 2021 this is Experiment 16's onset definition. A **quiet cell-day** is any cell-day meeting the second condition and in that stratum. Only quiet cell-days can be fire starts, so every model is trained and scored on quiet cell-days only.

## Forecasts

| Name | What it is |
|---|---|
| `start_drivers` | CatBoost trained on fire starts, with Experiment 21's inputs (fire history, BCWS incidents, weather, lightning) |
| `start_static` | `start_drivers` plus Experiment 18's static layers (terrain, fuels, farmland reserve, oil and gas, cutblocks) |
| `start_static_day` | `start_static` plus day of year |
| `catboost_spread_noday` | Experiment 21's next-day model, trained on all cell-days, loaded unchanged |
| `nearby_activity` | Experiment 18's ranking on the past 3 days: nearest detection, then active cells within 5 rings |
| `lightning` | Lightning flashes in the 3x3 block (about 30 km) over 3 days, then the 9x9 block |
| `random` | A random list of quiet cells (watch lists only) |

## Training, thresholds and scoring

- **Training:** quiet cell-days of 2018–2021, all fire starts plus Experiment 21's 1% draw of the other quiet cell-days, with the same seeds and CatBoost settings.
- **Thresholds:** the threshold that maximizes 2022 F1 per Fire Centre and model.
- **Evaluation:** 2023. 2024 is not loaded.
- **Metrics:**
  - F1, precision and recall on the exact cell, and with a one-cell margin (a flag counts if a start is in the cell or a neighbour, and a start counts if the cell or a neighbour is flagged);
  - average precision against the base rate;
  - daily watch lists: the top 10, 25, 50 or 100 quiet cells per Fire Centre, with the share of starts listed on the day, and listed at least once in the 7 days up to it;
  - catch rates by Experiment 16's onset groups (new BCWS wildfire, older wildfire flaring up, no BCWS record);
  - week-block bootstrap intervals against Experiment 21's model.

```bash
python run_experiment.py > run.log 2>&1   # --resume reloads models and finished Fire Centres after an interruption
python build_report.py                    # writes RESULTS.html
```

## Imported archive

Imported from `bc_labels_neural_ignitions_canadafiresat.zip` (source Experiment 20), renumbered to Experiment 34.
Folder paths and report text use this repository's numbering; internal module aliases
retain their source names for compatibility. Raw data, trained models and intermediate
results were not in the archive, so the code cannot be rerun from this folder alone.
