# Next-day fire model: current setup and results

*November 21, 2026. All numbers use the two-satellite label (Exp 32). Details:
[32_exp_two_satellite_bc/RESULTS.md](32_exp_two_satellite_bc/RESULTS.md).*

## 1. Data we use

- **Grid**
  - H3 resolution-7 hexagons: about 5.2 km² each, about 2.4 km between neighbouring cell centres.
  - 231,192 cells covering the six BC Fire Centres: Cariboo, Coastal, Kamloops, Northwest, Prince George, Southeast.
- **Season:** May 1 – Sep 30 each year. Forecast dates run May 8 – Sep 30, so every forecast has 7 days of history.
- **Fire detections**
  - **S-NPP VIIRS** (NASA FIRMS, 375 m pixels), 2018–2024.
  - **NOAA-20 VIIRS** (NASA FIRMS, 375 m pixels), 2018–2026.
- **Lightning:** CanCPLD cloud-to-ground flashes, 0.1° grid, 3-hourly, summed into UTC days.
- **Years**
  - Train on 2018–2021.
  - Choose thresholds on 2022.
  - Test on 2023, every cell and every day (34.2 million cell-days).
  - 2024 is kept as a frozen holdout and is not used.
- **Tested but not used in the best model**
  - ERA5-Land weather and soil moisture (0.1°, daily): no gain.
  - Copernicus 90 m elevation: no gain.
- **Not used:** BCWS station weather and incidents (Exp 21 only), CEMS, MODIS, satellite imagery.

## 2. Labelling

- **A cell "has fire" on a day** if at least one S-NPP **or** NOAA-20 detection falls inside it on that UTC date.
- **Label for day t** = the cell has fire on day t.
  - **One-day gaps are filled:** if the cell had fire on t − 1 and t + 1 but none on t, day t also counts as fire,
    because the satellite most likely missed it.
- **Everything else is 0.** No detection means "not seen", not "confirmed no fire". Clouds and smoke can hide fires.
- **Inputs use only data up to day t − 1.** The model forecasts tomorrow from what was known by the end of today.
- **This is satellite hotspot activity, not ignition.** It includes prescribed and agricultural burns, and a
  long-burning fire counts on every day it is detected.

## 3. Features

All features for forecast day t are computed from data up to the end of day t − 1. "Detections" means S-NPP or NOAA-20.

### Main model: 22 inputs

**Fire history in the cell**

| Feature | Meaning | Importance |
|---|---|---|
| `local_count_1d` / `_3d` / `_7d` | Detections in the cell over the last 1 / 3 / 7 days | 0.3 / 1.5 / 0.2% |
| `frp_sum_1d`, `frp_max_1d` | Total and maximum fire radiative power (FRP, fire intensity) in the cell yesterday | 0.8 / 0.4% |
| `frp_sum_7d` | Total FRP in the cell over the last 7 days | 3.0% |
| `night_count_1d` | Night-time detections in the cell yesterday | 0.0% |
| `active_days_7d` | Days with a detection in the cell, out of the last 7 | 0.1% |
| `days_since_detection` | Days since the cell's last detection this season | 5.2% |
| `history_left_censored` | No detection in the cell yet this season | 0.1% |

**Fire history in the neighbourhood**

| Feature | Meaning | Importance |
|---|---|---|
| `neighbour_count_1d` / `_3d` / `_7d` | Detections in the 6 neighbouring cells over the last 1 / 3 / 7 days | 0.3 / 0.2 / 1.7% |
| `disk1_active_days_7d` | Days with a detection in the cell or its neighbours, out of the last 7 | 6.8% |
| `quiet_stratum` | No detection in the cell or its neighbours for 7 days (quiet cell) | 0.9% |

**Lightning (CanCPLD)**

| Feature | Meaning | Importance |
|---|---|---|
| `ltg_cell_1d` | Flashes yesterday in the 0.1° box holding the cell | 0.2% |
| `ltg_3x3_1d` / `_3d` | Flashes in the surrounding 3 × 3 boxes (~30 km), last 1 / 3 days | 0.4 / 0.3% |
| `ltg_9x9_1d` / `_3d` | Flashes in the surrounding 9 × 9 boxes (~90 km), last 1 / 3 days | 5.0 / 3.6% |
| `ltg_days_since_3x3` | Days since the last flash in the 3 × 3 block (capped at 30) | 66.1% |

**Spread**

| Feature | Meaning | Importance |
|---|---|---|
| `nd_score` | Next-door spread model's probability that this cell burns tomorrow. Only for cells next to yesterday's fire; missing elsewhere. | 2.9% |

**How to read importance.** It is CatBoost's PredictionValuesChange on the training data. Training holds a 1% sample
of quiet cells, each weighted 100×, so importance mostly shows what separates the millions of quiet non-fire cells.
That is why days since lightning looks dominant. Exp 26's ablation showed that removing lightning barely changes
results. Importance is not proof of value.

### Next-door spread model: the inputs behind `nd_score`

- Trained only on cells next to yesterday's fire: 126,976 cell-days in 2018–2021, 8.7% of which burn.
- Uses the 21 fire-history and lightning features above, plus the groups below.

**Fire shape (the only group that improved results)**

| Feature | Meaning | Importance |
|---|---|---|
| `sp_frp_nb` | Total FRP of the burning neighbours yesterday | 21.6% |
| `sp_burned_15d` | The cell itself burned in the previous 15 days | 6.6% |
| `sp_size5` | Burning cells within 5 rings (~12 km): fire size | 3.9% |
| `sp_growth5` | Change in that count since the day before: fire growth | 3.2% |
| `sp_n20_burn_nb` | Neighbours detected by NOAA-20 yesterday | 2.4% |
| `sp_burn_nb` | Burning neighbours yesterday (of 6) | 2.4% |
| `sp_cos_growth` | Whether the cell lies in the direction the fire has been growing | 1.1% |
| `sp_new_nb` | Neighbours that started burning yesterday (active front) | 0.6% |

**Terrain (Copernicus 90 m; no gain)**

| Feature | Meaning | Importance |
|---|---|---|
| `tr_elev_mean`, `tr_elev_std`, `tr_elev_range` | Cell elevation: mean, spread and range | 9.8 / 1.8 / 2.2% |
| `tr_dz_mean`, `tr_dz_max`, `tr_dz_min`, `tr_uphill_share` | Elevation difference between the cell and its burning neighbours (positive = cell is uphill), and the share of burning neighbours below it | 0.5 / 1.1 / 1.3 / 0.3% |

**Wind (ERA5-Land, day t − 1; no gain)**

| Feature | Meaning | Importance |
|---|---|---|
| `wd_speed` | Wind speed | 4.4% |
| `wd_align`, `wd_align_x_speed` | How much the wind blows from the burning neighbours toward the cell, and that multiplied by speed | 2.5 / 2.8% |

**Ablation (next-door AP gain over the previous step):**

| Step | 95% interval |
|---|---|
| Fire shape | **+0.044 to +0.065** |
| Terrain | −0.008 to +0.003 |
| Wind | −0.008 to +0.004 |

Elevation ranks high in importance but adds nothing in the ablation. As a level it probably stands in for location.

### Tested and dropped

| Input | Result |
|---|---|
| **ERA5-Land, 22 features** (temperature, humidity, VPD, wind, soil moisture layers 1–3 and anomaly, rain over 1–14 days, dry days, evaporation, radiation, snow) | Lowers total F1 (−0.010 to +0.001) |
| **Target-day wind** (perfect forecast, upper bound only) | Next-door AP +0.003 to +0.021. Small even with perfect knowledge. |

## 4. Fire cells by type

**Definitions.** Every fire cell-day is one of four types, based on what the satellites saw before it.

| Type | Definition |
|---|---|
| **Already burning** (continuing) | The cell itself had fire yesterday |
| **Next-door spread** | Not in the cell yesterday, but in at least one of its 6 neighbours yesterday |
| **Recent fire nearby** | Not next door yesterday, but fire in the cell or a neighbour at some point in the last 7 days |
| **New fire, quiet area** | No fire in the cell or its neighbours for the past 7 days |

**Counts by year:**

| Year | Fire cell-days | Already burning | Next-door spread | Recent fire nearby | New, quiet area |
|---|---|---|---|---|---|
| 2018 | 36,345 | 24,419 (67%) | 6,455 (18%) | 2,399 (7%) | 3,072 (8%) |
| 2019 | 1,927 | 671 (35%) | 185 (10%) | 278 (14%) | 793 (41%) |
| 2020 | 1,495 | 668 (45%) | 136 (9%) | 223 (15%) | 468 (31%) |
| 2021 | 21,900 | 13,928 (64%) | 4,293 (20%) | 1,679 (8%) | 2,000 (9%) |
| 2022 | 5,077 | 2,952 (58%) | 831 (16%) | 475 (9%) | 819 (16%) |
| **2023 (test)** | **60,907** | **36,839 (60%)** | **12,188 (20%)** | **6,548 (11%)** | **5,332 (9%)** |

**How likely each kind of cell is to burn the next day (2023):**

| Candidate cells | Cell-days | Burn rate |
|---|---|---|
| Burning yesterday | 56,746 | 65% (35% stop) |
| Next to a fire | 120,873 | 10% |
| Recent fire nearby | 250,088 | 2.6% |
| Quiet | 33.8 million | 0.016% (1 in 6,000) |

- **Big fire years** (2018, 2021, 2023) are dominated by burning and spreading fires.
- **Quiet years** (2019, 2020) have a much larger share of new fires.

## 5. Best model by fire type (2023)

**Best model = stacked CatBoost.**
- **Inputs:**
  - fire history from both satellites (counts, FRP, neighbours, days since detection);
  - lightning;
  - a score from a dedicated next-door spread model, based on burning neighbours' FRP, NOAA-20 neighbours, fire
    size and growth, and "burned in the last 15 days".
- **Thresholds:** one per Fire Centre, chosen on 2022.

### Overall

| Model | F1 | Precision | Recall |
|---|---|---|---|
| Rule: fire in the cell yesterday (either satellite) | 0.626 | 0.649 | 0.605 |
| General model (no spread score) | 0.625 | 0.658 | 0.594 |
| **Stacked model (best)** | **0.629** | 0.645 | 0.613 |

- **Stacked vs general model:** +0.002 to +0.006 (95% range, resampling weeks). A real gain.
- **Stacked vs rule:** −0.001 to +0.007. Ahead, but not significantly.
- **By Fire Centre:** stacked is best in 5 of 6.

### By fire type

Recall = share of that type's fires caught. Precision = share of the model's flags on that type of cell that really
burned.

| Type (fires) | Rule: recall / precision | General: recall / precision | **Stacked: recall / precision** |
|---|---|---|---|
| Already burning (36,839) | 100% / 65% | 96.4% / 67% | **97.1% / 67%** |
| Next-door spread (12,188) | 0% / — | 5.1% / 29% | **12.5% / 34%** (1,524 caught) |
| Recent fire nearby (6,548) | 0% / — | 0.7% / 39% | **0.6% / 39%** |
| New, quiet area (5,332) | 0% / — | **0%** / — | **0%** / — |

- **Already burning:** every model catches almost all of them. The rule's 19,907 false alarms are fires that stopped;
  the models drop a few of these.
- **Next-door spread:** the only type where the models add real catches, and the stacked model doubles the general
  model's.
- **New fires:** the yes/no forecast flags none. Even the highest-scored quiet cells burn only about 1–2% of the time,
  far below the threshold, and flagging them would lower F1.

## 6. Watch list

**What it is.**
- Each day, each Fire Centre gets a list of its **N highest-scored quiet cells**, whatever their scores.
- It is a ranked list for prioritising patrols or attention, not a yes/no forecast.
- It answers: "if we can check N cells per centre per day, how many new fires were on the list the day before?"
- A Fire Centre has about **38,600 quiet cells per day**, so a 50-cell list is about 0.13% of them.

**How it looks with the current model (2023; 5,332 new fires in quiet areas):**

| List size per centre per day | Cells listed (season, BC) | New fires on the list | Share caught | Listed cells that burned | Random list catches |
|---|---|---|---|---|---|
| 10 | 8,760 | 148 | 2.8% | 1.7% | 0.04% |
| 25 | 21,900 | 258 | 4.8% | 1.2% | 0.08% |
| **50** | **43,800** | **406** | **7.6%** | **0.9%** | **0.13%** |
| 100 | 87,600 | 586 | 11.0% | 0.7% | 0.26% |

- **Better than chance, far from reliable.** With 50 cells per centre, the list catches 7.6% of new fires, about
  **58× a random list**, but only about 1 in 110 listed cells burns the next day.
- **The spread score doesn't change it.** The general model's list scores about the same (7.5% at 50), as expected,
  since the spread score only applies next to a fire.
- **Room to improve.** New fires depend on where lightning strikes, where people are, and how dry the fuel is in that
  spot. Our inputs don't see these at cell level, which is the gap satellite embeddings (fuel, land cover, roads)
  are meant to fill.
