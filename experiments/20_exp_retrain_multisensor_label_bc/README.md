# Experiment 20: retraining on the multi-satellite label

Experiment 19 found that a label from both VIIRS satellites (S-NPP and NOAA-20), with one-day
gap filling, is much more consistent than the S-NPP label. This experiment retrains the
next-day forecast on that label and tests whether the model now beats the rule.

```powershell
python ../../scripts/download_cwfis_hotspots.py   # once; writes datasets/cwfis_hotspots/
python run_experiment.py
python build_report.py
```

- **Label:**
  - A positive is an S-NPP (FIRMS) or NOAA-20 (CWFIS) VIIRS detection in the cell, plus a
    cell positive on the day before and the day after.
  - **Stopgap for training:** 2021 has the full label. 2018–2020 use gap-filled S-NPP only,
    because NOAA-20 is absent in 2018 and partial in 2019–2020. The FIRMS NOAA-20 request will
    replace this.
  - Inputs never use gap filling.
- **Rules:**
  - S-NPP detection in the cell yesterday (the old rule).
  - S-NPP or NOAA-20 detection in the cell yesterday (the new bar).
- **Models:** CatBoost on Experiment 13's features.
  - Experiment 13's model (old label).
  - Trained on 2018–2021 with the stopgap label.
  - Trained on 2021 only.
  - Trained on 2021 only, adding NOAA-20 and two-satellite inputs.
- **Scoring:**
  - Against the new label, on the exact cell and with the one-cell margin. The margin uses
    neighbourhoods within each Fire Centre grid, as in Experiment 19.
  - Thresholds are chosen on 2022 and applied unchanged to 2023.
  - Week-block bootstrap against the two-satellite rule.

`results/` contains metrics by year, a 2023 summary, bootstrap intervals, training-sample counts,
feature importance, fit checks, models and a manifest. `RESULTS.html` is the standalone report
(not published).
