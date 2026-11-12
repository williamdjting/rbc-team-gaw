# Experiment 09: next-day fire per H3 res-6 cell with TabPFN (Colab)

Predicts whether each H3 resolution-6 cell in BC will have a FIRMS detection on day *t+1*, using
ERA5 weather and FIRMS fire history up to day *t*.

**Notebook:** `tabpfn_next_day_fire_colab.ipynb`. Open it in Google Colab with a GPU runtime and upload:

- `datasets/interim/era5_h3_res6_cell_date_18_24.parquet` (2.4 GB)
- `datasets/interim/firms_h3_res6_cell_date_18_24.parquet`

Set `DATA_DIR` in the config cell if the files are on Google Drive. Run once with `QUICK = True` as
a smoke test.

## Design

- **Join:** ERA5 (every cell, every day) is the spine; FIRMS is left-joined so cell-days without
  a detection become 0. Label `y` = fire in the same cell on the next day. The last day of each
  season (30 September) is dropped because it has no next-day label.
- **Split:** train 2018-2022, validation 2023 (calibration, threshold), test 2024.
- **Features (54):** day-*t* ERA5 weather plus derived humidity, VPD and wind speed; 3- and
  7-day rainfall and means; cell centroid and day of year; this cell's fire today and over the
  last 3 / 7 / 30 days; earlier years' fire rate; ring-1 and ring-2 neighbour fire, and ring-1
  fires with the wind blowing toward the cell.
- **Sampling:** all positives; negatives split into *active* (fire in or near the cell recently)
  and *quiet*, kept at different rates. Every kept row carries weight = 1 / keep probability,
  so weighted metrics describe the whole BC grid.
- **Odds correction:** each model sees far more fire than the real rate, and differently for
  active and quiet rows. Scores are corrected per stratum, then isotonic-calibrated on 2023.
- **Models:** persistence baseline, XGBoost on GPU, TabPFN on GPU (3 contexts of 10k rows,
  averaged).
- **Metrics:** weighted PR-AUC (main), ROC-AUC, Brier, and precision / recall / F1 at the
  best-F1 threshold from 2023 applied unchanged to 2024.

## Data facts (checked 2026-11-11)

- ERA5: 30,793 cells × 1,071 days (May-September 2018-2024), one row per cell-day, no nulls.
- FIRMS: 53,953 fire cell-days in 6,188 cells; all cells exist in ERA5.
  Per year: 2018 11,990 · 2019 950 · 2020 723 · 2021 7,993 · 2022 2,085 · 2023 21,262 · 2024 8,950.
- In 2023, 94% of next-day fire cells had fire in or near the cell in the previous days; 6% were
  new ignitions with no recent fire nearby.

## Status

The data pipeline, label checks, metrics and plots were run locally in quick mode on the real
files, with scikit-learn stand-ins for TabPFN and XGBoost. Neither real model has been run yet.
That needs a Colab GPU.
