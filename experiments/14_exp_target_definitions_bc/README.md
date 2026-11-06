# Experiment 14: target definitions and the F1 ceiling

Experiment 13 found that, with a detection in the exact H3 res-7 cell as the target, no
model beats the previous-day rule, and the rule sits near F1 0.55. About 40% of the rule's
false alarms and 49% of its misses are one-cell shifts. This experiment scores the same
inputs under four definitions of a correct forecast, to see which can reach F1 0.70–0.80.

Run from this folder after Experiments 02, 03 and 13 have completed:

```powershell
python run_experiment.py
python build_report.py
```

Dependencies: as Experiment 13 (pandas, numpy, scipy, scikit-learn, pyarrow, joblib,
matplotlib, h3 4.5.0, catboost). Features, loading and models come from
`../13_exp_fire_context_bc/`; nothing earlier is modified.

- **Definitions:**
  - `res7_exact` is the target of Experiments 02–07 and 13.
  - `res7_margin1` keeps res-7 forecasts. A flagged cell is correct if a detection falls in
    it or a neighbour; a detected cell is caught if it or an in-centre neighbour was flagged.
  - `res6_exact` uses H3 res-6 cells (~36 km²). Each res-6 cell's score is its highest
    res-7 child score, and its truth is any child detection.
  - `res6_margin1` is res-6 with the same one-cell margin.
- **Models:** Experiment 13's gradient boosting and CatBoost on all fire-context features,
  trained on the exact-cell target. The same two are also retrained on "a detection in the
  cell or a neighbour tomorrow", with Experiment 03's sampling rule and every positive of
  that target kept. The previous-day rule is scored the same way.
- **Thresholds:** per Fire Centre, definition and model, maximizing 2022 F1 over score
  quantiles that cover the top 5% of rows, then applied unchanged to 2023. The best F1 at
  any 2023 threshold is stored as an after-the-fact reference.

Checks: the exact-cell rule reproduces Experiment 02's F1 in every region-year, source
hashes are unchanged, and 2024 is never loaded.

`results/` contains metrics by year, the 2023 median summary, fit checks, training-sample
counts, retrained models and a manifest. `RESULTS.html` is the standalone report.
