# Experiment 18: static layers, a quiet-cell watch list and a wildfire flag

Experiments 16 and 17 found that fires in quiet cells never reach the forecast's alert
threshold. Most of them are burns with no BC Wildfire Service (BCWS) record, at flat, low
farmland and oil and gas sites. The rest are older wildfires flaring up and new, mostly
lightning-caused, wildfires. This experiment adds per-cell static layers from
`scripts/build_static_layers.py` and tests them in three ways.

```powershell
python ../../scripts/build_static_layers.py   # once; writes datasets/static_layers/
python run_experiment.py
python build_report.py
```

Dependencies: as Experiment 17. Uses Experiment 13's features and model, Experiment 14's
scoring, Experiment 16's onset groups and Experiment 17's history features and wildfire link.

- **Static features:**
  - Elevation (mean and standard deviation) and slope.
  - Fuel-type shares, plus the conifer percent of mixedwood.
  - Agricultural Land Reserve share.
  - Oil and gas facilities and wells, in the cell and in the cell plus its neighbours.
  - Cutblock area harvested in the three previous years.

  Facilities, wells and cutblocks count only if dated before each season, so no later
  records are used. Terrain, fuels and the reserve are current layers.
- **A. Next-day forecast:**
  - CatBoost on Experiment 13's features, with and without the static layers, against the
    "fire in this cell yesterday" rule.
  - Scored on the exact cell and with the one-cell margin.
  - Thresholds are chosen on 2022 and applied unchanged to 2023.
  - Week-block bootstrap intervals for each Fire Centre and BC-wide.
- **B. Quiet-cell watch list:**
  - A quiet cell has no detection in the cell or its neighbours in the past 7 days.
  - Each day, the list holds the N highest-ranked quiet cells per Fire Centre, for N = 10,
    25, 50 and 100.
  - Rankings come from the two forecast models, a model trained only on quiet rows (adding
    Experiment 17's history features), two simple rankings (recent fire nearby, burned in
    an earlier season) and a random list.
  - Reported: the share of quiet fires on that day's list, the share listed on any of the 7
    days up to the fire, the share of listed cells with a detection next day and within 7
    days, and results by Experiment 16 onset group.
- **C. Wildfire flag:**
  - For each detected cell-day, the chance it is not linked to a BCWS wildfire (Experiment
    17's after-the-fact link).
  - Inputs: the static layers, the detection itself (intensity, night share, non-vegetation
    type, fire-complex size, recent activity), earlier seasons and location. It uses no
    BCWS inputs.
  - Compared with a simple rule: any oil and gas site nearby, or half the cell in the
    reserve.

`results/` contains forecast metrics, bootstrap intervals, watch-list metrics, wildfire-flag
metrics and predictions, feature importance, fit checks, models and a manifest.
`RESULTS.html` is the standalone report (not published).
