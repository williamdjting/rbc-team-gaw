# Experiment 31: spread features, terrain and wind for next-door fires

**Code only. Results are in [Experiment 32](../32_exp_two_satellite_bc/)**, which runs these scripts on the
two-satellite (S-NPP or NOAA-20) table. A first run on the S-NPP-only table was deleted on November 21, 2026.

**Question.** Can features describing the nearby fire, terrain and wind improve prediction of next-door spread?
Next-door spread means a cell with no detection yesterday but a burning neighbour.

## Inputs (all known by day t−1)
- **Fire shape**, built from S-NPP or NOAA-20 detections (FIRMS):
  - burning and newly burning neighbours;
  - their FRP;
  - burning cells within 5 rings, and the change since the day before;
  - direction of recent growth;
  - whether the cell burned in the previous 15 days;
  - NOAA-20 burning neighbours.
- **Terrain**, from the Copernicus GLO-90 DEM (`scripts/build_terrain_h3r7.py`):
  - cell elevation mean, standard deviation and range;
  - elevation difference to the burning neighbours;
  - uphill share.
- **Wind**, from ERA5-Land u10/v10 on day t−1:
  - speed;
  - alignment between the wind and the direction from the burning neighbours to the cell;
  - alignment × speed;
  - the target-day version as an upper bound only.
- **ERA5:** Exp 30's 22 features.

## Models
- **Feature sets** S0–S4 and U, one dedicated CatBoost each, trained on next-door rows (2018–2021):

  | Set | Inputs |
  |---|---|
  | S0 | Base features |
  | S1 | + fire shape |
  | S2 | + terrain |
  | S3 | + wind |
  | S4 | + all ERA5 |
  | U | Target-day wind (upper bound) |

- **`stack.py`:**
  - The next-door model's score becomes an input to the full model (out-of-fold on training years).
  - It is compared with the rule and Exp 30 model A by fire type: continuing, next-door spread, recent fire nearby,
    and new in a quiet area.
