# Week 5 tasks: labels, watch list and features (all Gurleen)

**Update November 21, 2026:** all tasks, including the features track (section 6, Exp 30 and 32), are reassigned to Gurleen. Exp 30 (ERA5) is in `experiments/30_exp_era5_features_bc/`.

**Written:** November 14, 2026. **Read this first** before starting any Week 5 work in this repo
(people and coding assistants alike). The two tracks are independent: each person works in their
own experiment folder and only meets the other through the shared file contract in section 4.

---

## 1. Where the project stands (end of Week 4)

**Task.** Each day, for every H3 res-7 cell (~5 km²) in the six BC Fire Centres (May 8 – Sep 30),
predict whether the cell has a VIIRS satellite fire detection **tomorrow**.

| Item | Value |
|---|---|
| Train / threshold / dev / holdout years | 2018–2021 / 2022 / 2023 / 2024 |
| Main label | two-satellite (S-NPP + NOAA-20) detection, one-day gap filled (Exp 19–20) |
| Persistence rule ("fire in this cell yesterday"), 2023 | **F1 0.624** |
| Best model (Exp 22 MLP), 2023 | F1 0.626, a statistical tie with the rule |
| 2024 holdout | MLP 0.559 vs rule 0.567 |

**What we learned in Weeks 1–4**

- Bigger models (CatBoost, GNN, MLP, GRU, TabPFN) and extra inputs (FWI, weather stations,
  lightning, terrain/fuels) move F1 by about ±0.02. None beats the rule clearly.
- About 60% of positives are fires that are still burning, ~32% are spread to a neighbouring cell,
  and ~8% are new fires in quiet cells. The rule already gets the first group right.
- An extra alert only raises F1 if it is right more than ~31% of the time. Alerts in quiet cells
  are right ~1.3% of the time, so new fires cannot raise F1. They need their own output.
- **Watch list (Exp 27):** each day, list the top 50 quiet cells per Fire Centre. Catch rate
  within 7 days, 2023:

  | Ranking | Caught within 7 days |
  |---|---|
  | Fire-nearby rule | 5.7% |
  | TabPFN (Exp 28) | 10.4% |
  | MLP (Exp 25) | 12.1% |
  | CatBoost with BCWS incident inputs | **18.4%** |

  The inputs drive the gain, not the model size.
- **Label problems.** About 45% of quiet-cell onsets have no BCWS record. Many are industrial
  heat. BCWS fires are 63% lightning-caused and 34% person-caused.
- **Keep the current target.** New fires are a subset of it. We add outputs and evaluations;
  we don't swap the target.

Full history: `docs/final-experiments-summary.md`, `docs/data-guide.md`,
`experiments/Week4_progress_brief.html`.

---

## 2. Ground rules for both tracks

1. **Always compare against the persistence rule** on the same label, cells and days. A model
   that beats only other models is not a result.
2. **Splits.**
   - Train 2018–2021. Choose thresholds and early stopping on 2022.
   - Report 2023.
   - Score 2024 only once, at the end. No tuning on 2024.
3. **Metrics to report.**
   - F1, precision and recall on the full grid, never on a sampled test set.
   - Recall split by onset type: continuing / spread / quiet (new).
   - Watch-list catch rate (top 50 per centre, caught within 7 days).
   - Week-block bootstrap 95% intervals for differences against the rule. Exp 27 has the code.
4. **No future data in inputs.**
   - Features for target day *t* use data up to day *t − 1* only.
   - A weather value from day *t* or *t + 1* can only appear in a run labelled "perfect
     forecast (upper bound)". Never use it in the main result.
5. **Folders and numbering.**
   - Gurleen: `experiments/29_*`, `experiments/31_*`.
   - Gurleen (features track, reassigned Nov 21): `experiments/30_*`, `experiments/32_*`.
   - Each folder holds `README.md` (how to run), `RESULTS.md` (answer first, then tables) and
     `results/` (small CSVs only).
   - Large data stays out of git and goes under `data/`.
6. **Git.** `git pull` before starting and before pushing. Only commit files in your own
   folders, plus the shared contract in section 4.

---

## 3. Papers

| Paper | Where | What to take from it |
|---|---|---|
| Porta et al., *CanadaFireSat: Toward high-resolution wildfire forecasting with multiple modalities*, ISPRS J. 2026 | [arXiv 2506.08690](https://arxiv.org/abs/2506.08690) · data code [eceo-epfl/CanadaFireSat-Data](https://github.com/eceo-epfl/CanadaFireSat-Data) · model code [eceo-epfl/CanadaFireSat-Model](https://github.com/eceo-epfl/CanadaFireSat-Model) · dataset [Hugging Face](https://huggingface.co/datasets/EPFL-ECEO/CanadaFireSat) | See below |
| Hollmann et al., *TabPFN* | `william/week2/TabPFN.pdf` | Background for Exp 07 and Exp 28. It ties the MLP when both get the same inputs. |

**CanadaFireSat in brief**

- **Task.**
  - Will a 100 m patch burn in the next **8 days**?
  - Label: NBAC final burned-area polygons.
  - One positive per fire event per grid cell, dated at the fire's start. So there are no
    "still burning" positives (see `src/preprocess/burned_area.py` and `temporal_freq.py`).
  - Negatives: 2 per positive, sampled evenly across FWI levels, never in cells that burned in
    earlier years (`src/sampling/negative.py`).
- **Inputs.** Sentinel-2 images (64 days), MODIS (NDVI, EVI, LAI, FPAR, LST), ERA5-Land
  (8-day mean/max/min, plus RH, VPD and wind speed), CEMS FWI and drought code. No
  fire-history input.
- **Results, 2023 test.**
  - FWI ≥ 6 baseline: F1 50.3%.
  - Weather-only deep model: 53.5%.
  - Images + weather: 60.3%.
  - "Test Hard" (negatives FWI-matched to the fires): all drop to 33–37%.
  - Their models beat FWI mainly on low-FWI days.
- **Not comparable with our 0.62.** Their test set is ~39% fire by sampling; ours is ~0.1% on
  the full grid. They forecast 8 days ahead, not one day.
- **What we borrow.**
  - FWI baseline, FWI-matched hard evaluation, time since last burn → Gurleen.
  - ERA5 window statistics, MODIS vegetation → Gurleen (reassigned Nov 21).
- **What we don't borrow.** Sentinel-2 imagery, sampled test sets as the headline, an 8-day
  target as the main task.

---

## 4. Shared file contract (how the tracks meet)

The features track (reassigned to Gurleen Nov 21) writes the per-cell daily feature tables; the watch-list code reads them. Keep
these names and columns fixed so neither person waits on the other.

```
data/features_h3r7/era5_{year}.parquet     # one row per (h3_cell, date)
data/features_h3r7/modis_{year}.parquet    # one row per (h3_cell, date)
```

| Column | Meaning |
|---|---|
| `h3_cell` | H3 res-7 index, string, same cell universe as Exp 25 (`data/snpp_lightning_h3r7/`) |
| `date` | **The day the values describe.** To predict target day *t*, join `date = t − 1`. |
| feature columns | Prefix `era5_` or `modis_`, lower-case snake case, units in `README.md` |

- Years 2018–2024, May 1 – Sep 30. A few days before May 8 are needed for the history windows.
- MODIS composites: `date` must be the day the composite became **available**, which is its
  last day, not its first. Forward-fill to daily.
- Write a `data/features_h3r7/README.md` listing every column, its source and its units.

---

## 5. Gurleen: labels and watch list

### G1. Fill the NOAA-20 gap, 2018–2020 (label)

- **Why.** The two-satellite label is complete from 2021. NOAA-20 is missing in 2018 and partial
  in 2019–2020, so training years currently use a stop-gap label (Exp 20).
- **Do.**
  1. Request NOAA-20 VIIRS detections from NASA FIRMS (archive download) for BC, May–Sep
     2018–2020. CWFIS only has partial coverage.
  2. Rebuild the two-satellite gap-filled label for 2018–2024 with Exp 19's rules.
  3. Re-score the persistence rule on all years.
- **Done when.**
  - The label table covers every year, with a short consistency check per year (detections per
    satellite, overlap).
  - Rule F1 is reported per year on the new label.

### G2. Exp 29: watch list, hardened (ideas from CanadaFireSat)

Build on Exp 27 (`experiments/27_exp_quiet_watchlist_bc/run_experiment.py`).

1. **FWI baseline.** Rank quiet cells by CEMS FWI (yesterday, and the 8-day mean). Add it to the
   watch-list table next to the rule and CatBoost.
2. **Results by FWI band.**
   - Report the catch rate and hit rate within FWI bands: 0–5, 5–10, 10–20, 20–30, 30+.
   - Hard check: compare quiet onsets with non-onsets of the **same FWI**, as in CanadaFireSat's
     Test Hard. This shows whether the ranking does more than spot hot, dry days.
3. **Time since the cell last burned.**
   - Build it from BCWS perimeters (`data/bcws_perimeters/`), as of day *t − 1*.
   - Add it to the CatBoost watch-list model and test its effect with a week-block bootstrap.
4. **Skill relative to the baseline.** Report (score − score_rule) / (1 − score_rule) next to
   the raw numbers.
5. Optional, if time:
   - Score the watch list against the BCWS-wildfire-only label: onsets matched to BCWS records.
   - Once the feature tables exist (section 4), add ERA5 and MODIS to the watch-list model.

- **Done when.** `experiments/29_*/RESULTS.md` answers:
  - Does the watch list beat FWI ranking?
  - Does it still work within an FWI band?
  - Does time since last burn help?

  Each answer comes with 2023 numbers and intervals; 2024 is scored once.

---

## 6. Features (reassigned to Gurleen, November 21)

### A0. Housekeeping (first)

- Rename `experiments/09_tabpfn_next_day_h3_res6/`. It clashes with
  `09_firms_incident_diagnostics`. Suggested name: `experiments/30_exp_era5_features_bc/`, or keep
  the res-6 notebook as its own folder.
- Update its README: the models were run, and it should record the results table and the
  caveats.
- That notebook's results (res-6, S-NPP label) for reference:

  | 2024 | F1 | PR-AUC |
  |---|---|---|
  | Persistence | 0.582 | 0.471 |
  | TabPFN | 0.580 | 0.546 |
  | XGBoost | 0.569 | 0.540 |

  This matches the team finding: ranking improves, but yes/no alerts don't beat the rule.

### A1. ERA5 feature table (res-7)

- **Fix the daily accumulations first.**
  - `scripts/check_ERA5_coverage.py` sums `tp`, `ssrd`, `ssr`, `str`, `slhf`, `pev` and
    `evabs`/`evatc`/`evavt` over 5 snapshots (00/05/10/15/20 UTC).
  - ERA5-Land accumulations are **running totals since 00 UTC**, so summing them over-counts.
  - The daily total for day *d* is the value at **00 UTC on day d + 1**.
- **Drop or flag non-informative columns.**
  - `lai_hv` and `lai_lv` are a fixed climatology: the same every year.
  - `cvh`, `cvl`, `z` and `lsm` are static.
- **Map 0.1° ERA5 to res-7 cells.** Use the nearest grid point or area weighting, and record
  which in the README.
- **Features.**
  - Daily values.
  - RH, VPD and wind speed.
  - 3-, 7- and 8-day mean/max/min.
  - Dry days in the last 7.
  - Rain over 3, 7 and 14 days.
  - Soil-moisture change.
- Write the table per section 4.

### A2. Storm-potential features (cheap add-on)

- CAPE and convective precipitation come from **ERA5 single levels**, which is a different
  product from ERA5-Land.
- If the catalogue has it, also add the lightning flash density field. Check, don't assume.
- **Don't build a lightning-prediction model.** Even observed CanCPLD lightning barely moved
  results (Exp 24–26: watch list 12.1% with vs 11.9% without). A predicted map is noisier.

### A3. MODIS vegetation features

- Products: NDVI and EVI (MOD13A1, 16-day, 500 m), LAI and FPAR (MOD15A2H, 8-day, 500 m) and
  LST (MOD11A1, daily, 1 km), via Google Earth Engine. These are the products CanadaFireSat
  uses.
- **Why.** ERA5-Land's LAI is a fixed climatology. MODIS is the only input showing *this year's*
  vegetation state.
- **Features.**
  - Cell means.
  - Anomaly versus the cell's 2018–2021 average for that time of year.
  - Change since the previous composite.
- Use the availability date (section 4), never the composite start date.

### A4. Exp 30: does ERA5 / MODIS help? (ablation)

- **Setup.** Same label, splits, cells and thresholds as Exp 21/25. Use Exp 25's table
  (`data/snpp_lightning_h3r7/`) as the fire-history base.
- **Variants.**
  1. Fire history only.
  2. Fire history + ERA5.
  3. Fire history + ERA5 + MODIS.
  4. Optional: + storm features.
  5. Optional, labelled as an upper bound: + ERA5 for the target day itself ("perfect
     forecast").
- **Model.** CatBoost first, which is fast and strong on tables. TabPFN only as a check.
- **No day-of-year or lat/lon inputs.** Exp 21 found that day-of-year hurt transfer between
  years. Lat/lon lets the model memorise places.
- **Report.**
  - F1 vs the rule, with bootstrap intervals.
  - Recall on continuing / spread / quiet onsets.
  - Watch-list catch rate on quiet cells.
  - Feature importance.
- **Done when.** `experiments/30_*/RESULTS.md` answers whether ERA5 and MODIS add anything over
  fire history, overall and on new fires, with intervals.

---

## 7. Known data facts (save time)

- **FIRMS files contain only detections.** Label 0 comes from the full cell × day grid built in
  Exp 02. `observation_coverage` is "unknown" everywhere.
- **The boundary file moved.**
  - `datasets/bc_legal_boundary.geojson` was removed in the ERA5 commit.
  - A copy is at `data/static/bc_legal_boundary.geojson`.
  - Newer scripts look in `datasets/raw/`.
- **The ERA5 CSV is not a daily average.**
  - `dataset_ERA5_18_24.csv` (repo root, 4.4 GB, git-ignored): 13,334 BC land points per day,
    May–Sep 2018–2024.
  - Its daily value is the mean of 5 snapshots, not a true daily average. Rebuild from the raw
    files with the A1 fix.
- **No GPU on Gurleen's machine.** It has an AMD 780M integrated GPU, which PyTorch CUDA can't
  use. TabPFN v2.5+ weights need a licence login; v2 weights work without one.
