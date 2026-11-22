# Experiment 32 results: two-satellite label, ERA5 and spread features (2023)

**Answer.**
- **Stacked model is the best total F1 so far on the full 2023 grid: 0.629.** The general model scores 0.625 and the
  "fire yesterday" rule 0.626.
  - Stacked − general model: +0.002 to +0.006 (95%, week bootstrap). Significant.
  - Stacked − rule: −0.001 to +0.007. Not significant.
- **Next-door spread:** the stacked model catches 1,524 of 12,188 next-door fires (12.5%), and 34% of its next-door
  flags are right. The rule catches none.
- **ERA5 does not help** total F1 (−0.010 to +0.001), and **no model catches new fires in quiet areas**.

**Setup.**
- Label, fire history, quiet flag and rule all use S-NPP **or** NOAA-20 (FIRMS) for 2018–2023.
- Train 2018–2021, thresholds per Fire Centre on 2022, test every 2023 cell-day (34.2M).
- 2024 is not used.

## Total F1, 2023

| Model | F1 | Precision | Recall |
|---|---|---|---|
| Rule: either satellite yesterday | 0.626 | 0.649 | 0.605 |
| General model (fire history + lightning) | 0.625 | 0.658 | 0.594 |
| General model + ERA5 | 0.620 | 0.660 | 0.584 |
| Two models side by side (general model elsewhere, next-door model S3 on next-door cells) | 0.629 | — | — |
| **Stacked** (next-door score as an input to the general model) | **0.629** | 0.645 | 0.613 |

F1 by Fire Centre (rule / general / stacked):

| Fire Centre | Rule | General | Stacked |
|---|---|---|---|
| Cariboo | 0.551 | 0.563 | 0.567 |
| Coastal | 0.623 | 0.632 | 0.635 |
| Kamloops | 0.639 | 0.642 | 0.645 |
| Northwest | 0.618 | 0.618 | 0.615 |
| Prince George | 0.628 | 0.624 | 0.629 |
| Southeast | 0.641 | 0.641 | 0.643 |

The stacked model is the best in 5 of 6 centres.

## Fire cells by type, 2023 (60,907 fire cell-days)

| Type | Fires | Share | Candidate cell-days | Burn rate | Rule: caught / flagged | General: caught / flagged | Stacked: caught / flagged |
|---|---|---|---|---|---|---|---|
| Continuing (burning yesterday) | 36,839 | 60% | 56,746 | 65% | 36,839 / 56,746 (65% right) | 35,524 / 52,714 (67%) | 35,757 / 53,267 (67%) |
| **Next-door spread** | **12,188** | **20%** | 120,873 | 10.1% | 0 | 621 / 2,175 (29%) | **1,524 / 4,468 (34%)** |
| Recent fire nearby (within 7 days, not next door yesterday) | 6,548 | 11% | 250,088 | 2.6% | 0 | 49 / 126 | 39 / 100 |
| New, quiet area | 5,332 | 9% | 33.8M | 0.016% | 0 | 0 | 0 |

The rule's 19,907 false alarms are all fires that stopped.

## Next-door model by feature set (next-door cells, 2023: 120,873 cell-days, 12,188 fires)

| Model | AP | Recall | Precision | Next-door F1 |
|---|---|---|---|---|
| General model | 0.204 | 5.1% | 29% | 0.087 |
| S0 dedicated, base features | 0.213 | 11.9% | 26% | 0.163 |
| S1 + fire shape | **0.267** | 14.3% | 33% | 0.201 |
| S2 + terrain | 0.265 | 11.4% | 34% | 0.171 |
| S3 + wind (day before) | 0.264 | 11.3% | 37% | 0.173 |
| S4 + all ERA5 | 0.256 | 19.1% | 28% | 0.228 |
| U: target-day wind (upper bound) | 0.275 | 17.4% | 33% | 0.228 |

AP differences (95%, week bootstrap):

| Comparison | Interval |
|---|---|
| Fire shape − general model | **+0.051 to +0.075** |
| Fire shape − S0 | **+0.044 to +0.065** |
| Terrain − shape | −0.008 to +0.003 |
| Wind − terrain | −0.008 to +0.004 |
| All ERA5 − wind | −0.016 to +0.002 |
| Target-day wind − wind | **+0.003 to +0.021** |

The fire-shape features are burning neighbours' FRP, NOAA-20 burning neighbours, fire size and growth, and "burned
in the last 15 days". **They are the only real gain.** Terrain and day-before wind add nothing. A perfect wind
forecast adds a little. Recall and precision at threshold jump around between sets; AP is the stable measure.

## New fires in quiet areas (5,332 fires among 33.8M cell-days)

| Model | AP | Daily top-50 watch list catches |
|---|---|---|
| General model | 0.0041 | 7.5% |
| General model + ERA5 | 0.0026 | 5.7% |
| Quiet-only model | 0.0020 | 6.8% |
| Quiet-only model + ERA5 | 0.0051 | 7.4% |

- Quiet-only model + ERA5 at its 2022 threshold: recall 9.6%, precision 1.5% (F1 0.025).
- ERA5 watch-list gain: −0.2 to +1.6 points, not significant.
- New fires remain essentially unpredictable with these inputs.

## Caveats
- **NOAA-20 copy.** NOAA-20 comes from the FIRMS archive (Exp 21 used CWFIS), so this year's positives (60,907) differ
  slightly from Exp 21's (58,914).
- **Approximate ERA5.** Values are means of 5 snapshots, and 7% of cells have no ERA5 value.
- **No tuning.** One seed, default CatBoost settings.
