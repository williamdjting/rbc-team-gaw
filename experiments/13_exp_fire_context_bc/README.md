# Experiment 13: fire-context features for BC

Adds new information to Experiment 03's cell + neighbour baseline. Experiments 03–07
all reach F1 of about 0.45–0.58, close to what their fire-history inputs allow. Here
the rows, target, split, training sample and quiet-row correction are Experiment
03's exactly; only the inputs change.

Run from this folder after Experiments 02 and 03 have completed:

```powershell
python fetch_bcws_incidents.py   # once; writes data/bcws_incidents_2018_2023.csv
python run_experiment.py         # --reuse-sample skips rebuilding the training sample
python f1_checks.py              # bootstrap and rule-volume checks on the saved predictions
python build_report.py
```

Dependencies: pandas, numpy, scipy, scikit-learn, pyarrow, joblib, matplotlib,
h3 4.5.0, catboost. Inputs are Experiment 02's feature tables and grids,
`../firms_summer_2018_2024/data/firms_h3_detections.csv` and the BCWS download.
Loading, sampling, gradient-boosting settings and metric helpers are imported
from `../03_exp_learned_baselines_bc/run_experiment.py`. No earlier experiment
is modified.

- **FRP and day/night:** FRP sum and max yesterday, FRP over 7 days, night
  detections, for the cell and for the cell plus its ring-1 neighbours.
- **Wider area:** detections 2 and 3 H3 rings out over 1 and 7 days, grid distance
  to the nearest detection over 1 and 3 days (up to 5 rings), and the number of
  active cells within 5 rings.
- **Fire complex and duration:** cells detected in the past 3 days are joined by
  H3 adjacency into complexes. Features are the largest complex size and 3-day FRP
  within 2 rings, plus the number of days with detections in the past week.
- **BCWS incidents:** DataBC "Fire Incident Locations – Historical", records typed
  "Fire", fire years 2018–2023 only. Features are the distance to and number of known
  active fires, the age and lightning cause of the nearest, fires declared out in
  the last 3 days, and new fires within 25 km. Incidents are used one day after
  FIRMS: target day t sees ignition and out dates up to t−2. Final size and
  perimeters are never used.
- **Ablation:** gradient boosting on the base set, each group added alone, all
  FIRMS groups, and all four groups. CatBoost (Experiments 05–06 settings) on the
  base and full sets.
- **Thresholds:** each centre's threshold maximizes 2022 F1, as before. A second
  rule pools leave-one-year-out 2018–2021 scores on the training sample (quiet
  negatives weighted ×100) with all 2022 rows. Best-in-hindsight 2023 F1 is also
  stored as an upper reference.
- **Label check:** the same 2023 predictions are re-scored against a target that
  ignores FIRMS static-source and offshore detections (types 2 and 3).

Checks:
- Rebuilt local counts and target equal Experiment 02's on every interior cell. Border
  cells may count more, because Experiment 02 assigns detections to a centre by point.
- Base gradient boosting reproduces Experiment 03's 2023 AP in every centre.
- Past-only history, no 2024 rows, no saturated training log-odds, and source
  hashes unchanged.

`f1_checks.py` resamples whole 2023 weeks (2,000 draws) for 95% intervals of model F1
minus rule F1. It also scores a "rule volume" operating point: each day, the k
highest-scoring cells, where k is the number of cells the rule flags that day.

`results/` contains metrics by year (both threshold rules and the vegetation-target
check), the 2023 ablation, best-in-hindsight F1, precision-recall curves, CatBoost
feature importance, fit checks, training-sample counts, models, per-centre 2023
predictions and a manifest. `RESULTS.html` is the standalone report.
`data/training_sample.parquet` is a regenerable cache and is git-ignored.
