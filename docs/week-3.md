# Week 3

**Date:** October 29, 2026

## Milestone

Clean and join data sources, engineer time-based features, and create labels for candidate prediction windows.

## Deliverable

Labeled daily dataset.

## Direction change: predict new fire starts, not continued burning

Experiments 08-10 showed the current FIRMS target ("any detection in the cell tomorrow") is mostly repetition of fires that are already burning:

- In 2021-2023, 60-72% of FIRMS positive cell-days are day 8 or later of a BC-recorded fire. Fewer than 7% fall on a fire's first two days.
- 20 fires produce about half of all positives in big years (2018, 2021, 2023).
- Every model converges on the "active yesterday" persistence rule (F1 about 0.54). Quiet-cell recall is 0%.

The main task becomes **ignition prediction**: will a new fire start in this cell tomorrow? The label combines BC incidents and FIRMS onsets (Experiment 10). Continued burning becomes a smaller, separate spread task. Details are in `experiments/10_ignition_label_fusion/FINDINGS.md`.

## Tasks

### 1. Lock the ignition label

- Download BC historical fire perimeters and redo fire attribution using real fire shapes instead of point + final size.
- Spot-check 30-50 FIRMS-only onsets that keep growing (tier C) on maps to confirm they look like real, unrecorded fires.
- Freeze the onset definition (about 5 km and a 7-day quiet period), the tiers and the mask rules in one label-building script.
- Known source facts to carry forward:
  - BC `IGNITION_DATE` is local Pacific time stored as if it were UTC. Use the date as-is.
  - BC records cover only 30% of fires in 2018 and 80-85% in 2019-2020.

### 2. Build the ignition dataset

- Target: a tier A/B/C ignition event in the cell on day t+1, with masked cell-days excluded.
  - Main option: all ignitions, with tier and size as sample weights.
  - Secondary option: fires of at least 1 ha only, which can be checked against satellite data.
- Start at H3 resolution 6; keep resolution 7 as a sensitivity check.
- Split: train 2019-2021, select on 2022, evaluate on 2023. 2024 stays untouched. Treat 2018 BC negatives as unknown.

### 3. New baselines

- Climatology: historical ignition rate per cell and season.
- Fire-weather rule: FWI or its components alone.
- Human-access proxy: distance to roads and settlements.
- Metrics:
  - Average precision.
  - Recall when each fire centre gets a fixed number of alerts per day (as in Experiment 08).

### 4. Features for ignition

- Full daily CEMS fire-weather grid for every cell. This is currently joined only at detection points.
- Lightning: 74% of ignitions are lightning-caused. Use CLDN if accessible, or ERA5 CAPE and convective rainfall as a proxy.
- Static layers: fuel type, elevation, slope, and distance to roads and towns.

### 5. Models

- CatBoost first, reusing the existing pipeline.
- Move to the GNN only once the tabular model beats climatology.

### 6. Spread task, kept small

- Keep the FIRMS target, restricted to cells near a fire that is already burning.
- Weight each fire equally so the largest fires don't dominate.
- Persistence + GNN is the reference result.

## Progress

- [x] Task 1: Lock the ignition label. See `experiments/10_ignition_label_fusion/LABEL_SPEC.md`.
  - Downloaded 2,464 BC fire perimeters (`scripts/download_bc_fire_perimeters.py`) and redid attribution with them. Unexplained FIRMS positives in big years fell from 15-40% to about 3%.
  - Perimeters recovered 515 fires missing from the incident file, 385 of them in 2018. 2018 fires of at least 1 ha are now well covered; small 2018 fires are still missing.
  - Spot-checked 56 FIRMS-only onsets in two rounds. Most were gas flares, industrial sites or slash-pile burning, so FIRMS-only events (tier C) are off by default. FIRMS is used to confirm fires, correct 191 late BC dates, locate perimeter-only fires and build masks.
  - Final label: 6,795 ignition events (455-2,049 per season), plus fire-footprint and onset masks. Old FIRMS label: 828-45,861 positives per season.
- [x] Task 2: Build the ignition dataset. See `experiments/11_ignition_dataset/README.md`.
  - H3 resolution 6 over the six Fire Centre grids: 33,435 cells x 146 days x 6 years = 29.3M rows in `data/ignition_dataset/res6/` (gitignored; rebuild with `build_dataset.py`, about 50 s).
  - Columns use `y_` (targets), `info_` (label-only, never features), `masked`, and `f_` (past-only features: FIRMS history, earlier BC ignitions nearby, day of year, location).
  - 438-1,964 positive cell-days per season; prevalence 0.009-0.042%. All checks pass: event counts, no masked positives, past-only features.
  - Resolution 6 keeps 97% of resolution 7's positive cell-days with 7x fewer rows.
  - First signal: an ignition nearby in the last 7 days lifts the rate 17x; FIRMS history touches only 1-4% of ignitions. Region (13x) and month (5x) differences are large, so expect a strong climatology baseline.
- [x] Task 3: New baselines. See `experiments/12_ignition_baselines/README.md`.
  - New inputs:
    - BC Wildfire station weather from the BCWS Data Mart (260-302 stations, daily FWI codes, temperature, RH, rain, wind), interpolated to cells using previous-day values (`scripts/download_bcws_weather.py`, `add_weather.py`). The CEMS grid file is not in this checkout.
    - WUI human-interface buffer coverage and distance per cell (`scripts/download_bc_wui.py`, `add_static_human.py`).
  - Bar to beat: climatology x FWI x recent-ignitions rule. 2023 AP 0.0028 (6.7x prevalence), AUC 0.80, 11.8% recall with 100 alerts per centre per day. Climatology alone: 3.7x. FWI alone: 2.0x.
  - Recent nearby ignitions (lightning storms) are the strongest short-term signal. Human-access proxies are the best predictor of person-caused fires (6.8x, AUC 0.89) but don't help with lightning fires.
  - Absolute skill is low: ignition prediction is hard. Lightning data (Task 4) is the most promising addition.
  - Head-to-head on 2023 new fires: all Experiment 02-07 models (persistence, boosting, GNN, TabPFN, CatBoost) are close to random (AUC 0.51-0.63, 1.1-2.1x lift, about 4.6% recall with 100 alerts). The new baseline is 6.7x lift, AUC 0.80, 11.8% recall: 2.4-3.2x more new fires found at the same alert budget.
- [ ] Task 4: Features for ignition
- [ ] Task 5: Models
- [ ] Task 6: Spread task
