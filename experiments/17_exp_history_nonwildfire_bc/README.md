# Experiment 17: history features and a non-wildfire flag

Experiment 16 found that most "quiet" onsets are either burning with no BC Wildfire
Service fire on record, recurring at the same places, or older wildfires flaring up
beyond the 7-day, one-cell history window. This experiment adds longer history features
and a wildfire-linked target, for next-day forecasts.

```powershell
python fetch_bcws_perimeters.py   # once; writes data/bcws_perimeters_2018_2023.geojson
python run_experiment.py
python build_report.py
```

Dependencies: as Experiment 13, plus shapely 2. Uses Experiment 13's features and
incident download, Experiment 14's scoring, and Experiment 16's onset groups.

- **History features (past only):**
  - 30-day detections in the cell and in the cell plus its neighbours.
  - Active cells within 5 rings over 30 days, and the distance to the nearest detection
    in 30 days.
  - Days with activity since May 1.
  - Recent detection days with and without a known BCWS fire nearby (2–29 days back).
  - From earlier seasons: years the cell and its neighbours burned, and their wildfire-linked
    and other detection days.
- **Wildfire link:**
  - A detection is linked if it falls inside, or within 2 km of, the final perimeter of a
    BCWS fire active within 3 days. It is also linked if it is within 10 km of an active
    BCWS fire's ignition point.
  - Final perimeters are after-the-fact, so they set labels and prior-year features only.
  - Current-season features use the ignition-point link with incidents already reported.
- **Models:** CatBoost with Experiment 13's features, with the history features added, and
  the latter retrained on the wildfire target. Each training set keeps every positive of its
  own target, so the quiet-row correction stays exact.
- **Rules:** fire in the cell yesterday, and the same rule limited to cells with a known
  active BCWS fire within 10 km.
- **Scoring:** exact cell and one-cell margin, for both the recorded and wildfire targets.
  Thresholds are chosen on 2022 and evaluated on 2023. Quiet fires caught are reported by
  Experiment 16 onset group.

`results/` contains metrics by year, a 2023 summary, quiet-fire catches by group, feature
importance, fit checks, training-sample counts, models and a manifest. `RESULTS.html` is
the standalone report (not published).
