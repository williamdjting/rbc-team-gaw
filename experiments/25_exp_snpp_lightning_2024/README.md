# Experiment 25: S-NPP plus lightning, final-year evaluation

Separate experiment authorized November 12, 2026. This does not reuse or modify
Experiment 24's frozen two-satellite model. It uses only S-NPP observations and
CanCPLD cloud-to-ground lightning; there are no weather, incident, NOAA-20,
combined-satellite, wind-spread or calendar inputs.

Completed: see [RESULTS.md](RESULTS.md). On the full 2024 grid, pooled exact-cell
F1 is 0.5586 for the MLP and 0.5670 for persistence. The paired 95% interval for
their difference includes zero. No settings were changed after seeing 2024.
`verify_results.py` independently recounts the saved predictions and checks all
frozen hashes and thresholds.

## Protocol fixed before the test

- Same six H3-resolution-7 Fire Centre grids as Experiment 02; no grid expansion
  based on 2024 detections. Targets May 8-September 30; history begins May 1.
- Target: S-NPP detection on day t, or detections on both t-1 and t+1 that fill
  one missing day. Gap filling is label-only. No observation means an uncertain
  negative, not confirmed absence of fire. This is hotspot activity, not ignition.
- Inputs: 15 S-NPP history features plus six lightning features. All are computed
  through UTC day t-1. Lightning uses the containing 0.1-degree cell, surrounding
  3x3/9x9 blocks, three-day windows and time since nearby lightning.
- Raw CanCPLD three-hour counts are summed into UTC days. Any missing interval
  makes that cell-day unknown; missing neighbourhood values are not silently
  treated as zero. Missing flags accompany all model inputs.
- Train 2018-2021. Keep all recently active rows and positives, plus seeded 1%
  quiet-negative sampling. Apply log(0.01) to quiet-row output log odds.
- MLP: 256/256/128 hidden units, SiLU, dropout 0.1 after the first two layers,
  AdamW learning rate 0.001, weight decay 0.0001, batch size 2048, seed 42.
- Maximum 20 epochs, patience 3 on held-out seven-day blocks within the training
  seasons. Fit signed-log/scaling parameters only on fitting rows, not held-out
  rows. No architecture or hyperparameter search.
- Select exact-cell F1 thresholds on the full 2022 grid separately by centre.
  Check 2023 with those thresholds; do not tune from its results.
- Freeze model/scaler/threshold/source hashes before building 2024 features or
  scoring 2024. Primary metric: pooled exact-cell F1. Also report precision,
  recall, per-centre AP and paired whole-ISO-week bootstrap difference from the
  previous-day S-NPP persistence rule (2,000 resamples).
- Full-grid evaluation, no negative subsampling. Report no spatial-margin metric
  in this experiment. No claim that lightning improves skill without an ablation.

## Reproduce

From the repository root, using `.venv/forecast/Scripts/python.exe`:

```powershell
python scripts/download_cancpld_lightning.py
python experiments/25_exp_snpp_lightning_2024/run_experiment.py check
python experiments/25_exp_snpp_lightning_2024/run_experiment.py develop
python experiments/25_exp_snpp_lightning_2024/run_experiment.py test
```

`develop` creates the trained model, 2022/2023 results and frozen protocol.
`test` verifies the frozen hashes and refuses to repeat an already completed
2024 evaluation. Existing source experiment code remains untouched.

Raw inputs/provenance: `data/cancpld_lightning/`. Joined training samples and
full-grid evaluation tables: `data/snpp_lightning_h3r7/`. Local model, daily
counts, metric tables and bootstrap results: `results/` (Git-ignored).

Source: [official CanCPLD CG download directory](https://crd-data-donnees-rdc.ec.gc.ca/CDAS/products/CanCPLD/CLDN/cg/).
The [dataset paper](https://www.nature.com/articles/s41597-025-05924-7)
describes the three-hour UTC count convention and masked coverage.
