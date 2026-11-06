# Experiment 15: forecast lead time

Asks whether learned models pull away from persistence when forecasting further
ahead. Rows, features (Experiment 13, all groups), sampling and quiet-row correction
are unchanged; history ends on day e and the target is a recorded detection on day
e + L, for L = 1, 2, 3, 5 and 7 days.

```powershell
python run_experiment.py
python build_report.py
```

Dependencies and inputs as Experiment 13; scoring helpers come from Experiment 14.

- One CatBoost model per lead (Experiments 05–06 settings). Each lead keeps
  Experiment 03's active rows and drawn quiet rows, plus all of its own positives.
- Baseline: the best of six persistence rules, chosen on 2022 per Fire Centre, lead
  and definition. The rules flag a cell active in the past 1, 3 or 7 days, either
  in the cell alone or in the cell plus its in-centre neighbours.
- Scored on the exact res-7 cell and with Experiment 14's one-cell margin.
  Thresholds maximize 2022 F1 and are applied unchanged to 2023. The best F1 at any
  2023 threshold is stored as a ceiling.
- Rows whose target day falls after September 30 are dropped. 2024 is never loaded.

`quiet_cell_catch.py` rescores 2023 and splits fires by the cell's state on day 0:
burning, nearby activity in the past 7 days, or quiet. It reports the share of each
group the model and rules catch (`results/quiet_cell_catch_2023.csv`).

`results/` contains metrics by year, the 2023 comparison, fit checks,
training-sample counts, models and a manifest. `RESULTS.html` is the report.
