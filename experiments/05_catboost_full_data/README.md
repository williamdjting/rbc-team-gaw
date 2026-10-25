# Experiment 05: full-data CatBoost

CatBoost on every 2018-2021 training row (136,979,536 cell-days), with no
quiet-negative undersampling and no quiet-row odds correction. Follows
`william/instructions.md`.

Run from this folder once the regional feature Parquet files are unzipped into
`../02_hotspot_persistence_bc/regions/`:

```bash
python run_experiment.py
```

Dependencies: pandas, numpy, pyarrow, scikit-learn, scipy, catboost, joblib.
Shared loading, metric and threshold functions are imported from
`../03_exp_learned_baselines_bc/run_experiment.py`; nothing earlier is modified.

- Feature sets: `cell_only` and `cell_plus_neighbours`, as in Experiment 03
  (`quiet_stratum` included).
- Two class-handling variants per feature set: `unweighted` and
  `auto_class_weights="Balanced"`. Fixed, untuned CatBoost settings shared with
  Experiment 06 (300 iterations, learning rate 0.1, depth 6, seed 42).
- Thresholds maximize 2022 F1 per Fire Centre and configuration and are applied
  unchanged to 2023. 2024 is never loaded.
- The training pool is a column-major float32 array that CatBoost shares without
  copying (~6 GB for `cell_plus_neighbours`). On a 10-core laptop one fit takes
  about 70 minutes (about 14 s per iteration), so expect about 5 hours for all four fits.

`results/` will contain `catboost_metrics_2023.csv` (plus `_unweighted` and
`_balanced` splits), `catboost_predictions_2023.parquet` (plus per-variant files),
`catboost_class_balance_before_after.csv` (before = after; no sampling),
`catboost_comparison_2023.csv` against Experiments 03-04, fitted models and
`catboost_manifest.json`. After this run, re-run Experiment 06 so its comparison
also includes Experiment 05.
