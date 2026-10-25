# Experiment 06: undersampled CatBoost

The fair CatBoost comparison with Experiments 03-04. Follows `william/instructions.md`.

```bash
python run_experiment.py
```

CatBoost settings and helpers are imported from `../05_catboost_full_data/`.
Runs in about two minutes.

- Training sample identical to Experiment 03: every positive, every active row
  (activity in the past 7 days in the cell, or the cell and its in-region
  neighbours for `cell_plus_neighbours`), and the same seeded ~1% draw of quiet
  negatives. Kept rows match Experiment 03 exactly (1,514,696 / 1,758,947).
- Unweighted CatBoost. At prediction, quiet-row odds are multiplied by 0.01;
  active rows are unchanged. This is a sampling correction, not calibration.
- Evaluated on every 2022 and 2023 row. Thresholds maximize 2022 F1 per Fire
  Centre and configuration and are applied unchanged to 2023.

`results/` contains `catboost_metrics_2023.csv`, `catboost_predictions_2023.parquet`,
`catboost_class_balance_before_after.csv`, `catboost_comparison_2023.csv`
(Experiments 03-04, and 05 once its results exist), fitted models and
`catboost_manifest.json`.
