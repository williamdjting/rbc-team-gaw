# Experiment 37: will a lightning wildfire that grows to 10 ha start here in the next 7 days?

Experiment 36's forecast with the label restricted to **lightning-caused (natural) wildfires that grew to 10 ha or more**. About 72% of BCWS wildfires are lightning-caused and 25% person-caused; person-caused fires depend on human activity that the inputs do not describe. About 74% of fires stay under 1 ha and are caught at once, close to random events. Both are treated as noise for the label. The fires kept are the ones that matter for damage and response.

- **Label:** a BCWS wildfire (type "Fire") with cause "Lightning" and a recorded size (`CURRENT_SIZE`, the final size for past seasons) of at least 10 ha, whose point is in the cell and whose ignition date (Pacific time) is on any of days t to t+6. Size only selects which fires count; it is never an input.
- **Fires:** 1,048 such fires started May–September 2018–2023 (of 5,966 lightning wildfires). 601 are in the training years (351, 24, 19 and 207 for 2018–2021), 117 in 2022 (thresholds) and 330 in 2023 (evaluation).
- **Inputs unchanged:** Experiment 13's BCWS inputs still see every known active fire, whatever its cause, since that is known information on the forecast day.
- **Everything else is Experiment 36's:** the three area sizes (about 5, 36 and 250 km²), the three CatBoost models, the references, the training draw, thresholds from 2022, evaluation on 2023, watch lists, per-fire catch rates and the bootstrap. 2024 is not loaded.
- **Comparison:** Experiment 36 is the same run with fires of every cause in the label.

```bash
python run_experiment.py > run.log 2>&1   # --resume reloads models and finished Fire Centres after an interruption
```

## Imported archive

Imported from `bc_labels_neural_ignitions_canadafiresat.zip` (source Experiment 23), renumbered to Experiment 37.
Folder paths and report text use this repository's numbering; internal module aliases
retain their source names for compatibility. Raw data, trained models and intermediate
results were not in the archive, so the code cannot be rerun from this folder alone.
