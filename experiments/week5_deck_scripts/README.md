# Week 5 deck scripts

Scripts that produced `experiments/Week5_GAW_detailed.html` and `experiments/Week5_error_analysis.html`
(November 18, 2026). Run from the repo root with `.venv/forecast/Scripts/python.exe -I`. `WORK` is any
scratch folder for intermediate files.

## Week 5 deck (CanadaFireSat comparison)

1. `python experiments/week5_deck_scripts/sample_map.py . WORK/sampled_map.png`
   Kamloops map of every-cell vs 1 : 2 sampled scoring (needs Exp 38 `results/scores/`). The PNG is
   already embedded in `week5_body.html`; rerun only to change the map, then re-embed it.
2. Edit `week5_body.html` (slides, data and chart code).
3. `python experiments/week5_deck_scripts/assemble_week5_deck.py experiments/week5_deck_scripts/week5_body.html experiments`
4. Check: `python experiments/week5_deck_scripts/verify_week5.py experiments/Week5_GAW_detailed.html WORK/preview`

## Error-analysis deck (Exp 25 MLP, 2023, every cell)

1. `python experiments/week5_deck_scripts/classic_metrics.py . WORK/classic.json` (confusion, PR, threshold, histogram, calibration, ROC)
2. `python experiments/week5_deck_scripts/err_analysis.py . WORK/err.json` (one-day map cells, F1 by Fire Centre and month)
3. `python experiments/week5_deck_scripts/err_map.py WORK/err.json WORK/err_map.png`
4. `python experiments/week5_deck_scripts/build_classic_deck.py WORK experiments`
5. Check: `python experiments/week5_deck_scripts/verify_err.py experiments/Week5_error_analysis.html WORK/preview`

Inputs: `data/snpp_lightning_h3r7/evaluation/2023/`, `experiments/25_exp_snpp_lightning_2024/results/metrics_2023.csv`,
`experiments/02_hotspot_persistence_bc/regions/*/data/h3_grid.parquet`. The verify scripts need Chrome at its default path.

## Error-analysis deck, CatBoost version (current `Week5_error_analysis.html`)

Exp 21 CatBoost (best model for now), next-day two-satellite target, 2023, from Exp 21's saved results only:
`python experiments/week5_deck_scripts/build_catboost_deck.py experiments experiments/week5_deck_scripts/cb_centres.json`
then `python experiments/week5_deck_scripts/verify_cb.py experiments/Week5_error_analysis.html WORK/preview`.
The Exp 25 MLP version with curves and map is kept as `experiments/Week5_error_analysis_exp25_mlp.html`
(built by `build_classic_deck.py`).
Slides 5–6 (one-day map, F1 by Fire Centre × month) are next-day, on the 18 dates Exp 38 scored: run
`python experiments/week5_deck_scripts/cb_nextday.py . experiments/week5_deck_scripts` first (rebuilds the next-day label from FIRMS S-NPP +
FIRMS NOAA-20, checks it against Exp 38's `burn_8d`, writes `cbn.json`, `cbn_map.png`). `cb8_analysis.py` is the older 8-day version (unused).

Week 5 deck, last slide (confidence under the paper's evaluation): `python experiments/week5_deck_scripts/paper_eval_confidence.py . experiments/week5_deck_scripts`
redraws Exp 38's seed-0 Test sample with its own code and seed (asserts it reproduces F1 0.8518) and writes `paper_conf.json`; the slide's data is embedded in `week5_body.html`.
