# Experiment 21: weather, lightning and wind-aligned spread inputs

Every model so far has seen only fire history. On the two-satellite label (Experiments 19 and 20), the rule "S-NPP or NOAA-20 detection yesterday" misses about 47,000 fire cells in 2023: fire spreading into new cells, or new starts. It also raises about 39,000 false alarms on fires that stop. This experiment adds inputs aimed at both kinds of error:

- **Weather:** BC Wildfire Service station values from noon local standard time on the history day (20:00 UTC, before that UTC day ends): temperature, humidity, wind, 24 h and 3-day rain, and the Fire Weather Index system codes. Each cell takes its nearest station with a reading, among the three nearest within 100 km.
- **Lightning:** CanCPLD cloud-to-ground flashes on the history day and the two days before, at 0.1°: the cell itself, the 3×3 and 9×9 blocks, and days since lightning in the 3×3 block.
- **Spread:** burning neighbours one and two cells away, weighted by how far upwind they sit (cosine of the angle between the direction to the neighbour and the station's noon wind direction). Also multiplied by wind speed.

Label, sampling, training years, CatBoost settings and scoring are those of Experiment 20. NOAA-20 inputs are blank before 2021, because the CWFIS archive doesn't have full NOAA-20 coverage for those years. Thresholds are chosen two ways:
- maximise 2022 F1 per Fire Centre;
- flag each day as many cells as the two-satellite rule flags (no tuning at all).

The models below are trained on 2018–2021 with the stopgap label. Experiment 20's model is loaded and must reproduce Experiment 20's numbers.

| Model | Inputs |
|---|---|
| `catboost_stopgap` | Experiment 20's model (Experiment 13 inputs) |
| `catboost_n20` | + NOAA-20 and two-satellite history |
| `catboost_drivers` | + weather + lightning |
| `catboost_spread` | + wind-aligned spread |
| `catboost_spread_noday` | as `catboost_spread`, without `day_of_year` |

Data, built by scripts in `rbc-team-gaw/scripts/` (git-ignored):
- `datasets/bcws_weather/`, from `download_bcws_weather.py`. Rerun to add noon wind direction.
- `datasets/cancpld_lightning/`, from `download_cancpld_lightning.py`.

```bash
python run_experiment.py --regions Kamloops --results /tmp/exp21_smoke   # smoke test
python run_experiment.py > run.log 2>&1
python build_report.py                                                     # writes RESULTS.html
```

2024 is never loaded.

## Imported archive

Imported from `../latest.zip` (source Experiment 16), renumbered to Experiment 21.
Folder paths and report text use this repository's numbering. Internal Python module
aliases retain their source names, as in Experiments 13?20.

The saved report and outputs are included. Rerunning requires inputs not supplied by
this archive: `datasets/bcws_weather/station_daily_2018_2023.parquet`,
`datasets/bcws_weather/stations_2018_2023.parquet`, and
`datasets/cancpld_lightning/bc_cg_daily_2018_2023.npz`. The referenced
`download_cancpld_lightning.py` is absent from this checkout, and the existing
`download_bcws_weather.py` produces a different path/schema. Obtain the original
preprocessing inputs/scripts before rerunning.
