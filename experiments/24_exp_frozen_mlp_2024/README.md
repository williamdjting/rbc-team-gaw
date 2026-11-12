# Experiment 24: frozen MLP evaluation on 2024

Status: model and evaluation protocol frozen; evaluation blocked on missing
inputs and a year-specific evaluation adapter. No 2024 predictions or metrics
have been produced.

Update November 12, 2026: a separate S-NPP/lightning model has now been tested on
2024 in Experiment 25. This two-satellite experiment remains blocked and its
frozen artifacts are unchanged, but 2024 S-NPP outcomes have now been inspected;
future work must not describe 2024 as wholly untouched by project development.

`frozen_protocol.json` preserves Experiment 22's input order, fitted scaler,
network configuration, 2022 thresholds, and hashes of the checkpoint and source
code. `artifacts/mlp.pt` is a local copy of the original trained MLP checkpoint
(gitignored). The protocol contains 24 thresholds: six centres, two scoring
definitions, and the MLP plus two-satellite persistence rule.

Run `py -3 experiments/24_exp_frozen_mlp_2024/verify_freeze.py` from the repository
root to check the checkpoint, original source hashes and threshold completeness.
This is an integrity check, not an evaluation runner.

## Fixed evaluation

- Primary: pooled exact-cell F1 over all six centres on May 8-September 30, 2024.
- Baseline: S-NPP or NOAA-20 detection on the previous day.
- Additional reporting: per-centre precision, recall, F1 and average precision;
  median per-centre AP; one-cell-margin F1 at its separately frozen thresholds;
  paired whole-week bootstrap intervals versus persistence.
- Preserve the original H3 grids, feature order, label, season boundaries,
  sampling correction and fitted scaler. No training, threshold tuning, scaler
  refitting or model selection on 2024. Evaluate the entire grid.
- Reproduce the saved 2023 MLP results before reading 2024 labels, to validate the
  new evaluation adapter against the original implementation.

## Inputs needed to finish

The saved S-NPP detection file and MLP artifacts exist locally. The following
matching inputs are missing; filenames below describe the existing development
contract, with a corresponding 2024 extension required:

| Input | Required source/schema |
|---|---|
| NOAA-20 detections | `datasets/cwfis_hotspots/bc_hotspots_2018_2023.parquet`, plus 2024 using the same CWFIS preprocessing |
| Station observations | `datasets/bcws_weather/station_daily_2018_2023.parquet`, plus 2024 with previous-day noon values, 24-hour precipitation and wind direction |
| Station locations | `datasets/bcws_weather/stations_2018_2023.parquet`, plus matching 2024 locations and identifiers |
| Lightning | `datasets/cancpld_lightning/bc_cg_daily_2018_2023.npz`, plus 2024 on the same 0.1-degree grid |
| Incident context | Extend the Experiment 13 incident CSV through 2024 with the same date/status/cause handling |
| Cell-day inputs | Build 2024 rows on the frozen Experiment 02 region grids, preserving history and feature timing |

The existing `data/bcws_weather/` files have a different path/schema. The
`william/week4/2024_BCWS_WX_STATIONS.csv` file alone supplies no daily weather.
Neither is a drop-in replacement for the missing feature inputs. The original
lightning downloader is absent. Recover the original preparation scripts/data
before substituting or rebuilding any sources.

The active Python 3.13 environment also lacks torch, numpy, pandas, scipy,
scikit-learn, h3, pyarrow and catboost. Restore a compatible environment alongside
the original preprocessing pipeline.

The existing runners train models and hard-code 2018-2023 in several loaders.
Do not execute Experiment 22's `main()` as a holdout run or simply change its
evaluation-year constant. A separate evaluation-only adapter must load the
frozen checkpoint/scaler/thresholds and extend feature construction to 2024
without changing the development experiment files.
