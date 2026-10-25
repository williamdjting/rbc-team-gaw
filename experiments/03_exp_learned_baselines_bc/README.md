# Experiment 03: learned baselines for BC

Steps 1 and 2 of the planned comparison in `hotspot-forecasting-design.md`: a cell-only
learned baseline, then the same with neighbouring-cell features. Fixed rules from
Experiment 02 are replaced by fitted models; everything else is held constant so
the results compare directly.

Run from this folder after Experiment 02 has completed:

```powershell
python run_experiment.py
python build_report.py
```

Dependencies: pandas, numpy, scikit-learn, pyarrow, joblib, matplotlib.
Inputs are read from `../02_hotspot_persistence_bc/regions/<centre>/data/features/`;
nothing in Experiment 02 is modified.

- Same six Fire Centres, H3 resolution-7 grids and per-centre feature tables as Experiment 02.
- Same next-day recorded-activity proxy target; coverage-supported labels remain missing.
- Same split: train on 2018-2021, choose thresholds on 2022, evaluate on 2023. 2024 is never loaded.
- Feature sets: `cell_only` (past 1/3/7-day local counts, days since previous
  detection, history-censored flag, day of year) and `cell_plus_neighbours`
  (adds past 1/3/7-day in-region neighbour counts and in-region neighbour count).
  Day of year is derived from the target date; no new data sources are used.
- Models: logistic regression (log counts, day-of-year spline) and histogram
  gradient boosting with fixed, untuned settings. One pooled model per
  configuration across all six centres; metrics reported per centre.
- Training keeps all positives and every row with recent activity (past 7 days in
  the cell, plus neighbours for `cell_plus_neighbours`). The remaining quiet
  negatives are sampled at 1%. Models train unweighted, and quiet-row log-odds
  are shifted by log(0.01) at prediction. The shift is exact because the quiet
  flag is itself a model input. Inverse-rate weights were tried first. They made
  gradient boosting diverge, because at a ~0.03% base rate its first Newton steps
  saturate and scikit-learn has no step cap. Every fit is now checked for
  saturated log-odds.
- Average precision (threshold-free) is the primary metric. For F1, each centre
  and configuration uses the threshold that maximized 2022 F1, as Experiment 02
  selected rules on 2022.

Checks: past-only feature history, no 2024 rows, every positive kept in
training, no saturated training log-odds, Experiment 02's previous-day rule
reproduced exactly on every evaluated region-year, and input hashes unchanged.

`results/` contains metrics by year, the 2023 comparison against Experiment 02
rules, precision-recall curves, logistic coefficients, training-sample counts,
fitted models, per-centre 2023 predictions and a manifest. `RESULTS.html` is the
standalone report.
