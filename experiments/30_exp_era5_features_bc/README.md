# Experiment 30: ERA5-Land features (incl. soil moisture), overall vs new fires

**Code only. Results are in [Experiment 32](../32_exp_two_satellite_bc/)**, which runs this script on the two-satellite
(S-NPP or NOAA-20) table. A first run on the S-NPP-only table was deleted on November 21, 2026 to avoid confusion.

**Question.** Does adding ERA5-Land weather and soil moisture to fire history + lightning improve next-day fire
**overall** (total F1) and on **new fires**? A new fire is a cell with no detection in the cell or its ring-1
neighbours over the previous 7 days.

## Setup
- **Base table:** `data/pair_lightning_h3r7/` (built by Exp 32): 15 fire-history + 6 lightning features, using the
  two-satellite gap-filled target.
- **ERA5-Land, 22 features** (`build_era5_features.py`, from `dataset_ERA5_18_24.csv`), taken from day t − 1:
  temperature, RH, VPD, wind, skin temperature, soil moisture layers 1–3, soil-moisture anomaly, 7-day soil-moisture
  change, rain (1/3/7/14 days), dry days, 7-day temperature/RH/VPD statistics, potential evaporation, solar radiation,
  snow depth.
- **Mapping:** each H3 cell uses its nearest ERA5 point. Cells more than 10 km from any point get missing values.
- **Caveat:** the CSV's daily value is the mean of 5 snapshots, so rain, evaporation and radiation are proxies, not
  true daily totals.
- **Four CatBoost models:** A all rows, no ERA5; B all rows, + ERA5; C quiet rows only, no ERA5; D quiet rows only,
  + ERA5.
- **Train / threshold / test:** train 2018–2021; thresholds per Fire Centre on 2022; test the full 2023 grid.
- **Reported:** new-fire AP, watch lists, and a week bootstrap.

## Run
```powershell
.venv/forecast/Scripts/python.exe -I experiments/30_exp_era5_features_bc/build_era5_features.py .
bash experiments/32_exp_two_satellite_bc/run_all.sh
```
