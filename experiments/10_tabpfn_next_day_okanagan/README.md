# Experiment 10: next-day fire, Okanagan region (TabPFN, Colab)

The same pipeline as `experiments/09_tabpfn_next_day_h3_res6`, restricted to H3 resolution-6
cells whose centroid is inside **lat 49.0–51.0, lon −120.5 to −118.5**. This covers Kelowna,
Vernon, Penticton and southern Kamloops.

**Notebook:** `tabpfn_next_day_okanagan_colab.ipynb`. Same upload and GPU instructions as
Experiment 09.

## Why this box

Smaller boxes had too few fires to test on. Kelowna's immediate area (lat 49.3–50.4, lon
−119.9 to −118.9) has 735 fire cell-days over 2018–2024, and only **5 in 2024**. The chosen box
has:

| Year | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|---|
| Fire cell-days | 781 | 81 | 62 | 1,146 | 69 | 791 | 87 |

361 cells in the box have fire at least once. Total: 3,017 fire cell-days.

## What changes from Experiment 09

- Rows are restricted to the box before sampling, so the odds correction, weights and calibration
  all describe the regional population.
- Map days are chosen from fire in the region, not across BC.
- Output folder: `tabpfn_next_day_outputs_okanagan`.
- Quiet negatives in training are kept at **0.1%** (Experiment 09 uses 0.2%). Each kept quiet row
  gets weight 1,000. The odds correction uses the same rate, so it still applies.
- Everything else is the same: features, splits (train 2018–2022, validate 2023, test 2024),
  sampling, models, and metrics.

## Caveats

- **The 2024 test has only 87 fire cell-days.** PR-AUC on that set is noisy. Treat differences of
  a few hundredths as unreliable.
- The regional model sees far fewer examples (about 1,200 training fire rows, compared with
  23,721 for BC), so TabPFN and XGBoost may behave differently here than in Experiment 09.
- A regional model can't learn from fire elsewhere in BC. Its neighbour and fire-history features
  still use the whole province's FIRMS data where a neighbouring cell is outside the box. This
  is intended, because fire doesn't stop at the box edge.

## Status

The pipeline was run locally in quick mode with scikit-learn stand-ins for TabPFN and XGBoost
(2021–2022 training, one bag). The numbers from that run are **not results**; they only show the
code works. The full run needs a Colab GPU.
