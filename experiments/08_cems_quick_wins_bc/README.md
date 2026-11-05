# Experiment 08 — All 5 CEMS & FIRMS Quick Wins (`08_cems_quick_wins_bc`)

This experiment implements and evaluates **all 5 Quick Wins** for next-day wildfire hotspot forecasting across all 6 British Columbia Fire Centres (`Cariboo`, `Coastal`, `Kamloops`, `Northwest`, `Prince George`, `Southeast`) on the exact `234,554`-cell H3 resolution-7 grid (`34,244,884` cell-days per summer).

## Temporal Split & Data Hygiene
- **Training years:** `2018–2021` (`May 8 – Sep 30` target days; `May 1 – Sep 29` history windows)
- **Validation / threshold selection:** `2022` (and out-of-fold `2021 + 2022` / Empirical-Bayes shrinkage in Quick Win 4)
- **Development evaluation:** `2023` (`34,244,884` cell-days, `93,654` positive cell-days)
- **Holdout year:** `2024` is strictly excluded (`assert year < 2024`).

## The 5 Quick Wins Tested

1. **Quick Win 1 — Active vs. Quiet / Spread / New-Ignition AP Breakdown (`results/active_vs_quiet_ap_2023.csv`):**
   - Evaluates 2023 Average Precision separately on:
     - `all_rows`: All 34.24M cell-days
     - `active_local_7d`: Active persistence rows (`local_count_7d > 0`)
     - `no_local_7d`: Unburned cell-days (`local_count_7d == 0`), split into:
       - `neighbour_spread`: `local_count_7d == 0` and `neighbour_count_7d > 0` (fire spreading from an adjacent H3 cell into an unburned cell)
       - `pure_ignition`: `local_count_7d == 0` and `neighbour_count_7d == 0` (new ignitions far from active fires)

2. **Quick Win 2 — Fire Radiative Power (`FRP`), Brightness & 2-Hop Spatial Features (`M2_frp_brightness`):**
   - Adds 9 strictly past-only intensity and 2-hop spatial features from `datasets/firms_h3_cell_date_with_cems_18_24.csv`:
     - `local_frp_sum_1d`, `local_frp_sum_3d`, `local_frp_sum_7d`, `local_frp_max_1d`
     - `local_brightness_max_1d`
     - `neighbour_frp_sum_1d`, `neighbour_frp_sum_3d`
     - `neighbour_2hop_count_1d`, `neighbour_2hop_frp_1d`

3. **Quick Win 3 — Engineered CEMS Temporal, Climatological Anomaly & Interaction Features (`M3_engineered_cems`):**
   - Adds 11 strictly past-only CEMS weather build-up, anomaly, and fire-weather interaction features:
     - `fwinx_mean_3d`, `fwinx_max_7d`, `fwinx_delta_3d`
     - `drtcode_delta_7d`, `ffmcode_mean_3d`, `infsinx_max_3d`
     - `fwinx_anom`, `drtcode_anom` (relative to 2018–2021 cell-month training climatology)
     - `fwinx_x_local_1d`, `infsinx_x_neighbour_1d`, `fwinx_x_frp_1d`

4. **Quick Win 4 — Multi-Year Validation (`results/multi_year_validation_2023.csv`):**
   - Compares `2022`-only regional F1 threshold selection against out-of-fold `2021 + 2022` validation and Empirical-Bayes shrinkage toward the pooled BC threshold.

5. **Quick Win 5 — Multi-Seed Ensemble (`M5_ensemble_5seeds`), GBDT+GNN Blend (`M6_gbdt_gnn_blend`) & Paired Day-Block Bootstrap 95% CIs (`results/bootstrap_significance_2023.csv`):**
   - Averages predictions across 5 random seeds (`42..46`), blends with the Experiment `04a` message-passing GNN, and computes 1,000 paired day-block bootstrap replicates across the 146 target days of 2023.

## Usage
```bash
python experiments/08_cems_quick_wins_bc/run_experiment.py
python experiments/08_cems_quick_wins_bc/build_report.py
```

