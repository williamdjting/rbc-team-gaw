# Experiment 11: daily ignition dataset (Week 3, Task 2)

One row per H3 resolution-6 cell per target day (May 8 - September 30, 2018-2023)
over the six BC Fire Centre grids from Experiments 02-07 (33,435 cells, about 36 km²
each). Labels come from Experiment 10 (`LABEL_SPEC.md`, default label = tiers A+B).

```powershell
# prerequisites: Experiment 10 labels
& C:/Users/sgurl/anaconda3/python.exe scripts/download_bc_fire_perimeters.py
& C:/Users/sgurl/anaconda3/python.exe experiments/10_ignition_label_fusion/build_labels.py
# dataset (about 50 s); --resolution 5|6|7
& C:/Users/sgurl/anaconda3/python.exe experiments/11_ignition_dataset/build_dataset.py
# label counts at resolutions 5/6/7
cd experiments/11_ignition_dataset; & C:/Users/sgurl/anaconda3/python.exe resolution_sensitivity.py
```

Output (gitignored, rebuild locally): `data/ignition_dataset/res6/`

- `ignition_<year>.parquet`: 4,881,510 rows per year.
- `grid.parquet`: cell, region (majority of res-7 children), children in grid, centre.
- `summary_by_year.csv`, `summary_by_region_year.csv`, `checks.csv`, `manifest.json`.

## Columns

| Prefix | Columns | Rule |
|---|---|---|
| keys | `h3_cell`, `date`, `region`, `year`, `split` | Split: 2018 `train_extra` (small fires under-recorded), 2019-2021 `train`, 2022 `valid`, 2023 `test_dev`. 2024 untouched. |
| `y_` | `y_ignition` (main), `y_ignition_ge1ha`, `y_n_ignitions`, `y_n_lightning`, `y_n_person` | Events on day t. |
| `info_` | `info_n_tier_a`, `info_max_size_ha` | Retrospective. Use for weights or analysis only, **never as features**. |
| mask | `masked`, `mask_reason` (bit 1 footprint, bit 2 onset) | Drop masked rows from training and scoring. Positives are never masked. |
| `f_` | `f_lat`, `f_lon`, `f_doy`; `f_firms_cell_1d/7d`, `f_firms_ring1_1d/7d`, `f_firms_ring2_7d`, `f_days_since_firms_cell`; `f_bc_ign_cell_season`, `f_bc_ign_ring1_1d/7d`, `f_bc_ign_ring2_7d` | Data through t-1 only. History starts May 1 each year, with no cross-season history. `ring1` includes the cell itself for BC counts and excludes it for FIRMS counts. |

## Add-on inputs (added for Task 3)

| File | Builder | Columns |
|---|---|---|
| `weather_<year>.parquet` | `scripts/download_bcws_weather.py`, then `add_weather.py` (about 5 min) | `w_ffmc`, `w_dmc`, `w_dc`, `w_isi`, `w_bui`, `w_fwi`, `w_temp_max`, `w_rh_min`, `w_precip_mm`, `w_wind_max`, `w_fwi_station_km`. Same row order as `ignition_<year>.parquet`. |
| `static_human.parquet` | `scripts/download_bc_wui.py`, then `add_static_human.py` | `s_hib_frac`, `s_hib_dist_km`, `s_wui1km_frac` per cell. |

**Weather.** Source is the BC Wildfire Service Data Mart: 260-302 stations per year,
with daily FWI-system codes from the 12:00 LST observation, plus daily maximum
temperature, minimum RH, total precipitation and maximum wind from hourly
observations. Values are inverse-distance weighted from the 4 nearest reporting
stations within 150 km.

- Every `w_*` value for target day t is the station value from **day t-1**.
- 4-6% of cell-days are missing (more than 150 km from a reporting station).
- The median distance to the nearest FWI station is 34 km.
- The CEMS gridded product used in Experiments 03a/04a is not available in this
  checkout. BCWS stations are the provincial operational source.

**Human access.** The BC Wildfire WUI Human Interface Buffer (buffers around
structures) and the WUI 1 km buffer are current static products. Each cell gets the
share of its area covered, and the distance from its centre to the nearest buffer
(sinusoidal equal-area projection).

Lightning and fuel/terrain layers are still to do in Task 4.

## Counts (resolution 6)

| Year | Split | Positive cell-days | Of which >= 1 ha | Masked | Prevalence |
|---|---|---:|---:|---:|---:|
| 2018 | train_extra | 908 | 598 | 83,322 | 0.019% |
| 2019 | train | 553 | 105 | 17,927 | 0.011% |
| 2020 | train | 438 | 83 | 5,603 | 0.009% |
| 2021 | train | 1,284 | 384 | 62,761 | 0.027% |
| 2022 | valid | 1,331 | 283 | 17,258 | 0.027% |
| 2023 | test_dev | 1,964 | 572 | 185,464 | 0.042% |

Checks (`checks.csv`):

- Every in-scope default event is counted. 3 of 6,795 events fall outside the grid.
- No positive row is masked.
- 2,000 random rows per year have `f_firms_cell_7d` recomputed from the raw daily array (past-only).

## Resolution sensitivity (`results/resolution_sensitivity.csv`)

| Resolution | Rows | Positive cell-days | Events sharing a cell-day | Prevalence |
|---|---:|---:|---:|---:|
| 5 | 4.3M | 6,102 | 688 | 0.147% |
| 6 | 29.3M | 6,478 | 312 | 0.022% |
| 7 | 202.5M | 6,667 | 122 | 0.003% |

Resolution 6 keeps 97% of resolution 7's positive cell-days with 7x fewer rows.
Resolution 5 merges more same-day fires (lightning busts).

## First signal check (2021-2023, unmasked rows)

| Feature > 0 | Ignition rate lift | Share of positives |
|---|---:|---:|
| `f_bc_ign_ring1_1d` (ignition in cell or ring 1 yesterday) | 40x | 6% |
| `f_bc_ign_ring1_7d` | 17x | 16% |
| `f_bc_ign_ring2_7d` | 9.5x | 17% |
| `f_firms_cell_7d` | 6x | 1% |
| `f_firms_ring1_7d` | 4x | 4% |

Recent nearby ignitions (lightning busts) carry signal. FIRMS activity touches very
few ignitions, which confirms this is a different problem from persistence. Rates
vary 13x by region (Kamloops 0.105%, Northwest 0.008%) and 5x by month (July peak),
so the climatology baseline in Task 3 should be strong.
