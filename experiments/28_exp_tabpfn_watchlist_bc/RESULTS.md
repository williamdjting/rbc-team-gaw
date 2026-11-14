# Experiment 28 results: TabPFN as a quiet-cell watch-list ranking

**Question.** Does TabPFN rank quiet cells better than the MLP that sees the same information
(Experiment 25), and better than Experiment 13's CatBoost?

**Answer.** No. With the same inputs as the MLP (S-NPP history + lightning), TabPFN is slightly
worse than the MLP in 2023 and ties it in 2022 and 2024. It is far behind CatBoost, which also
sees BCWS incident reports. It does beat the simple "fire nearby recently" ranking in 2023.
This supports Experiment 27: what the model is told matters more than which model it is.

## Setup

- Rows: every quiet cell-day (no S-NPP detection in the cell or a ring-1 neighbour for 7 days),
  six Fire Centres, 2022-2024. About 34M quiet cell-days per year.
- Inputs (as Experiment 25, those not zero on quiet cells): days since detection,
  history-censored flag, six CanCPLD lightning features.
- Context: 1,000 quiet positives and 2,000 quiet negatives from 2018-2021 (of 5,348 positives
  available). Balanced so positives are visible; scores are only used to rank.
- TabPFN package 9.2.0 (as Experiment 07) with TabPFN v2 weights (newer weights need a licence
  login), `n_estimators = 1`, CPU. Unique input rows scored once: 0.17-0.33M per year,
  150-210 rows/s, 76 minutes in total.
- Watch list, metrics, tie-breaks and bootstrap: Experiment 27's.

## 2023, BC-wide, top 50 cells per centre per day

| Ranking | Inputs | Quiet fires caught within 7 days | Listed cells burning next day |
|---|---|---:|---:|
| Random | – | ~0.1% (next day) | ~0.02% |
| Recent fire nearby (rule) | FIRMS | 5.7% | 0.5% |
| **TabPFN** | FIRMS + lightning | **10.4%** | 0.6% |
| MLP (Experiment 25) | FIRMS + lightning | 12.1% | 0.8% |
| CatBoost (Experiment 13) | FIRMS + BCWS incidents | 18.4% | 1.3% |

Week-block bootstrap, caught within 7 days, top 50 (95% interval):

| Comparison | 2022 | 2023 | 2024 |
|---|---|---|---|
| TabPFN − MLP | −0.5 pts [−1.1, +0.2] | **−1.7 pts [−3.2, −0.6]** | −0.3 pts [−2.3, +2.1] |
| TabPFN − rule | −1.3 pts [−4.6, +1.8] | **+4.7 pts [+2.8, +6.8]** | +1.8 pts [−2.1, +7.0] |
| TabPFN − CatBoost | – | **−8.0 pts [−11.3, −5.3]** | – |

## By onset type, 2023, top 50

| Onset type | TabPFN | MLP | CatBoost |
|---|---:|---:|---:|
| New BCWS wildfire | 3.1% | 3.6% | 18.8% |
| Older wildfire flaring up | 18.8% | 20.3% | 21.4% |
| No BCWS record | 7.0% | 9.3% | 16.0% |

Without BCWS inputs, neither TabPFN nor the MLP finds new wildfires; both mostly catch flare-ups.

## Caveats

- TabPFN v2 weights, one context draw and one estimator, as Experiment 07. Newer weights or
  averaging several contexts might score a little higher; the gap to CatBoost is unlikely to close
  without BCWS inputs.
- 2023 was used before for model comparisons; 2024 was inspected in Experiments 25-26.

Files: `results/watch_summary_bc.csv`, `watch_metrics.csv`, `bootstrap_bc.csv`,
`watch_onsets_2023.csv`, `manifest.json`. Scores are cached in `results/scores/` (git-ignored).
