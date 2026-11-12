# Experiment 27: quiet-cell watch list for all six Fire Centres

Extends Experiment 18's watch list (Kamloops only) to Cariboo, Coastal, Kamloops, Northwest,
Prince George and Southeast. Each day, each centre lists its N highest-ranked quiet cells
(N = 10, 25, 50, 100). A quiet cell has no S-NPP detection in it or an in-region ring-1
neighbour over the 7 days to the history day; rules based on yesterday's fire never flag them.

No model is trained. Rankings come from saved scores:

| Ranking | Source | Years |
|---|---|---|
| `catboost_13`, `gbm_13` | Experiment 13 fire-context models (`results/predictions/*_2023.parquet`) | 2023 |
| `mlp_lightning` | Experiment 25 full-grid scores (`data/snpp_lightning_h3r7/evaluation/`) | 2022-2024 |
| `mlp_no_lightning` | Experiment 26 matched ablation scores | 2022-2024 |
| `nearby_activity`, `burned_before` | Experiment 18's simple rankings, rebuilt from FIRMS cell-days | 2022-2024 |
| `random` | Expected share for a random list | 2022-2024 |

Experiment 22's MLP/GRU are not scored: their weather and lightning inputs are missing.

Metrics, as Experiment 18: share of quiet fires on the list the day before, share listed on
any of the 7 days up to the fire, share of listed cells burning next day, results by
Experiment 16 onset type, and a week-block bootstrap of BC-wide differences.

```powershell
python run_experiment.py   # about 5 minutes; writes results/
```

Needs pandas, numpy, scipy, scikit-learn, pyarrow and h3 4.5. Results: [RESULTS.md](RESULTS.md).
