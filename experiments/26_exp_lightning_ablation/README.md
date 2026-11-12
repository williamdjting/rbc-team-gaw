# Experiment 26: matched lightning ablation

Follow-up analysis after Experiment 25's 2024 results were inspected. This is
not another untouched holdout evaluation. Protocol specified before this run.

Question: how much does lightning contribute to the Experiment 25 MLP?

- Reference: the existing S-NPP + lightning model and saved scores from 25.
- Ablation: retrain from scratch with all six lightning inputs AND their missing
  flags fixed to zero after scaling. Keep the 42-column network shape so seed 42
  produces the exact same initial parameter tensors. No lightning information
  reaches the ablated model. This is not masking only at prediction time.
- Use the identical saved 2018-2021 sampled rows, row order, fitting/validation
  split, non-lightning scaling, architecture, optimizer, batches, epoch limit,
  patience and seed. Reuse fitted scaling from 25; its retained-feature statistics
  are independent of lightning. Keep the existing reference model unchanged.
- Early stopping selects the ablation's epoch using training-season validation
  weeks. Select its own per-centre F1 thresholds on full-grid 2022, using the same
  method as 25. Save the model/threshold hashes before scoring 2023 or 2024.
- Evaluate exactly the same full-grid cell-days and S-NPP labels as 25.
- Report reference minus ablation: pooled F1, precision, recall, median per-centre
  average precision, regional changes, and paired ISO-week bootstrap intervals
  for F1 (2,000 resamples). Positive differences favour including lightning.
- No tuning based on 2023 or 2024, no additional seeds or architectures selected
  from outcomes. This one-seed ablation estimates contribution for this model/run,
  not a universal or causal effect of lightning on wildfire occurrence.

Run from the repository root:

```powershell
.venv/forecast/Scripts/python.exe experiments/26_exp_lightning_ablation/run_ablation.py
```

Local models, scores, thresholds and metric tables go in `results/`; the
human-readable result will be `RESULTS.md`. Experiment 25 artifacts remain intact.
