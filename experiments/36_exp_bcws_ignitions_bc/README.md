# Experiment 36: will a wildfire start here in the next 7 days? (BCWS ignitions as the label)

The project's aim is to predict whether a fire will start in an area. Experiments 34–35 defined a fire start from satellite detections. Satellites see only about a fifth of reported wildfires (7% of fires under 1 ha), usually a day or more after ignition, and many satellite "starts" are burns that were never reported as wildfires. This experiment uses the BC Wildfire Service's own record of where and when each wildfire started.

## Label

- **What counts:** a BCWS incident of type "Fire" (wildfire). Nuisance fires, smoke chases, field activities, duplicates and agency assists are left out.
- **When and where:** its point falls in the cell, and its ignition date (Pacific time) is on any of days t to t+6. The forecast is made with data up to day t-1.
- **Season:** forecast days May 8 to September 30. Ignitions up to October 6 count, so late-September windows are complete.
- **Size of the label:** about 1,500–2,100 such fires a season in active years, and 550–700 in quiet years (2019, 2020). 72% are lightning-caused and 25% person-caused. Each fire makes up to 7 cell-days positive.
- **Why 7 days:** pinning the exact day is very hard. A next-day version, smoke-tested on Southeast, reached F1 of only 0.03–0.07 for 250 km² areas.
- **No quiet-week condition:** every cell-day is a candidate.
- **No leakage:** inputs are known by the end of day t-1. Experiment 13's BCWS inputs only see records dated t-2 or earlier, so they never see a fire being predicted.

## Area size

"An area" can mean different scales, so every forecast is scored at three:

| Area | Approximate size | Score of the area |
|---|---|---|
| res-7 H3 cell | 5 km² | the cell's own probability |
| res-6 H3 parent | 36 km² | 1 − ∏(1 − p) over its cells (simple rankings: maximum) |
| res-5 H3 parent | 250 km² | as above |

Parents are formed within each Fire Centre. Probabilities are corrected for the 1% training draw before they are combined.

## Forecasts

| Name | What it is |
|---|---|
| `ign_drivers` | CatBoost on BCWS ignitions, Experiment 21's inputs (fire history, BCWS incidents, weather, lightning, upwind fire) |
| `ign_lightning` | + lightning over the past 7 and 14 days (fires can smoulder for days after a strike) |
| `ign_full` | + Experiment 18's static layers (terrain, fuels, farmland reserve, oil and gas, cutblocks) |
| `catboost_spread_noday` | Experiment 21's next-day satellite model, loaded (the baseline for the bootstrap) |
| `start_drivers` | Experiment 34's satellite fire-start model, loaded |
| `lightning_14d` | Ranking by lightning in the 3x3 block over 14 days, then the 9x9 block |
| `near_active_bcws` | Ranking by distance to the nearest active BCWS wildfire |
| `random` | A random list (watch lists only) |

## Training, thresholds and scoring

- **Training:** 2018–2021, with every positive cell-day plus a 1% draw of the other cell-days (Experiment 21's seeds) and Experiment 21's CatBoost settings.
- **Thresholds:** maximize 2022 F1 per Fire Centre, area size and model.
- **Evaluation:** 2023. 2024 is not loaded.
- **Metrics:**
  - F1, precision and recall per area size;
  - average precision against the base rate;
  - daily watch lists of the top 0.1%, 1% and 5% of each Fire Centre's areas, with the share of positive area-days listed;
  - per fire, for lightning- and person-caused fires: was its area on the top-1% list (or above the threshold) for any of the 7 forecasts covering it, and for the first of them, a full week ahead?
  - week-block bootstrap intervals against Experiment 21's model.

```bash
python run_experiment.py > run.log 2>&1   # --resume reloads models and finished Fire Centres after an interruption
python build_report.py                    # writes RESULTS.html
```

## Imported archive

Imported from `bc_labels_neural_ignitions_canadafiresat.zip` (source Experiment 22), renumbered to Experiment 36.
Folder paths and report text use this repository's numbering; internal module aliases
retain their source names for compatibility. Raw data, trained models and intermediate
results were not in the archive, so the code cannot be rerun from this folder alone.
