# Experiment 27 results: quiet-cell watch list, all six Fire Centres

**Question.** Each day, list the N highest-ranked *quiet* cells per Fire Centre (no S-NPP
detection in the cell or a ring-1 neighbour for 7 days). How many quiet fires does the
list catch, and does a learned model beat simple rankings?

**Answer.** Yes for Experiment 13's CatBoost, in all six centres. With 50 cells per centre per
day it catches 18.4% of 2023 quiet fires within 7 days, against 5.7% for the best simple
ranking and about 0.1% for a random list (next day). The S-NPP + lightning MLP (Experiments
25/26) is clearly worse than CatBoost and barely beats the simple ranking outside 2023. Lightning
makes no difference. CatBoost's edge is largest on new BCWS wildfires, which points to its
BCWS incident features, not model complexity.

## 2023, BC-wide (pooled over 4,924 quiet fires)

| Ranking | Caught next day (top 50) | Caught within 7 days (top 10 / 25 / 50) | Listed cells burning next day (top 50) | Centres beating "recent fire nearby" (top 50) |
|---|---:|---:|---:|---:|
| Random | 0.1% | – | ~0.02% (quiet base rate) | – |
| Burned in an earlier season | 1.6% | 1.1% / 1.5% / 1.7% | 0.2% | 0 of 6 |
| Recent fire nearby (30 days) | 4.2% | 1.4% / 2.9% / 5.7% | 0.5% | – |
| MLP, S-NPP + lightning (Exp 25) | 7.4% | 4.8% / 8.5% / 12.1% | 0.8% | 6 of 6 |
| MLP, no lightning (Exp 26) | 7.7% | 5.6% / 9.1% / 11.9% | 0.9% | 5 of 6 |
| **CatBoost, fire context (Exp 13)** | **11.7%** | **7.2% / 12.9% / 18.4%** | **1.3%** | **6 of 6** |
| Gradient boosting, fire context (Exp 13) | 12.2% | 6.3% / 11.9% / 19.0% | 1.4% | 6 of 6 |

Week-block bootstrap, BC-wide, top 50, caught within 7 days (95% interval):

| Comparison | 2023 difference |
|---|---|
| CatBoost − recent fire nearby | +12.7 pts [+10.1, +15.4] |
| MLP (lightning) − recent fire nearby | +6.4 pts [+4.0, +8.9] |
| MLP (lightning) − CatBoost | **−6.3 pts [−9.7, −3.2]** |
| MLP with − without lightning | +0.1 pts [−0.6, +0.9] |

## By Fire Centre, 2023, top 50, caught within 7 days

| Centre | Quiet fires | CatBoost | MLP (lightning) | Recent fire nearby |
|---|---:|---:|---:|---:|
| Cariboo | 180 | 38.3% | 15.6% | 13.9% |
| Coastal | 294 | 30.6% | 26.9% | 16.0% |
| Kamloops | 448 | 27.2% | 17.2% | 8.0% |
| Northwest | 426 | 27.2% | 18.5% | 13.8% |
| Prince George | 3,347 | 12.5% | 8.4% | 2.4% |
| Southeast | 229 | 39.7% | 22.3% | 13.1% |

Prince George holds 68% of 2023 quiet fires (mostly burns with no BCWS record), so pooled
figures lean on it.

## By onset type, 2023, top 50, caught within 7 days (Experiment 16 groups)

| Onset type | Cells | CatBoost | MLP (lightning) | Recent fire nearby |
|---|---:|---:|---:|---:|
| New BCWS wildfire | 885 | **18.8%** | 3.6% | 1.8% |
| Older wildfire flaring up | 1,690 | 21.4% | 20.3% | 9.2% |
| No BCWS record | 2,341 | 16.0% | 9.3% | 4.3% |

The MLP matches CatBoost on flare-ups but misses new wildfires. CatBoost has BCWS incident
inputs (distance to active fires, new fires within 25 km, recently declared out); the MLP
has only S-NPP history and lightning.

## Other years (MLP and simple rankings only; no Experiment 13 scores exist)

| Year | Quiet fires | MLP (lightning), top 50 | Recent fire nearby, top 50 | Difference [95% CI] |
|---|---:|---:|---:|---|
| 2022 | 601 | 17.8% | 18.6% | −0.8 pts [−4.4, +2.6] |
| 2024 | 2,801 | 14.0% | 12.0% | +2.0 pts [−1.1, +5.5] |

Outside 2023 the MLP is not distinguishable from the simple ranking.

## Checks

- Kamloops 2023 reproduces Experiment 18: CatBoost 20.8% / 27.2% (Exp 18: 20.7% / 27.3%),
  recent fire nearby 6.0% / 8.0% (5.8% / 7.8%), burned before 1.1% / 1.3% (1.3% / 1.3%);
  448 quiet fires (450). Small differences come from random tie-breaking and border cells.
- Quiet-cell truth from Experiment 13 and Experiment 25 differs on 2-24 cell-days per centre
  out of millions (border-cell assignment).

## Caveats

- CatBoost was scored on 2023 only. 2023 was already used for model comparison, and Kamloops
  2023 for Experiment 18's watch list.
- Experiment 22's MLP/GRU (which also have BCWS incident inputs) could not be scored: their
  weather and lightning preprocessing inputs are missing.
- BCWS incident features assume incidents are known two days later; real reporting delays
  were not audited.
- Hit rates per listed cell stay low (about 1 in 75 next day): a watch list improves recall
  of quiet fires, not alert precision.
- 2024 has been inspected before (Experiments 25-26).

Files: `results/watch_summary_bc.csv`, `watch_metrics.csv` (per centre/year/size),
`bootstrap_bc.csv`, `watch_onsets_2023.csv`, `quiet_average_precision.csv`, `checks.csv`,
`manifest.json`. Run: `python run_experiment.py` (about 5 minutes, no training).
