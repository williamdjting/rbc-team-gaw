# Experiment 07: TabPFN hybrid

TabPFN on active rows, Experiment 06 CatBoost on quiet rows. Follows `william/week2/instructions-tabpfn.md`.

```bash
python run_experiment.py --quick   # smoke test, prints TabPFN rows/s
python run_experiment.py
```

Requires Experiments 05 and 06 to have run: it imports Experiment 05's helpers
and loads Experiment 06's `results/models/catboost_unweighted__*.cbm`.

- TabPFN context: uniform draws of 10,000 rows from every 2018-2021 active row
  (3 draws, averaged), so active scores need no odds correction.
- Quiet rows keep Experiment 06's CatBoost score with the 0.01 odds correction.
- Evaluated on every 2022 and 2023 row. Thresholds maximize 2022 F1 per Fire
  Centre and configuration and are applied unchanged to 2023.
- `active_ap_tabpfn` vs `active_ap_catboost_06` compares the two models on the
  rows TabPFN scores.
