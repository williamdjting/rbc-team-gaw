# Experiment 12: ignition baselines (Week 3, Task 3)

Non-learned baselines for next-day ignition on the Experiment 11 dataset (H3
resolution 6, masked rows dropped). These are the bars the Task 5 models must clear.

```powershell
& C:/Users/sgurl/anaconda3/python.exe experiments/12_ignition_baselines/run_baselines.py   # about 40 min
```

Needs `data/ignition_dataset/res6/` with `weather_<year>.parquet` and
`static_human.parquet` (see `experiments/11_ignition_dataset/README.md`).

Setup:

- Fit on 2018-2021. Climatology settings are selected on 2022 average precision; 2023 is development evaluation.
- Training on 2018-2021 beat 2019-2021 (2022 AP 0.00166 vs 0.00134), so 2018 is kept.
- Metrics:
  - Average precision (AP), pooled and as the mean of regions.
  - ROC AUC.
  - Recall with K alerts per Fire Centre per day (K = 5, 20, 100 cells of about 36 km²).
- Prevalence is 0.027% in 2022 and 0.042% in 2023, so "lift" = AP / prevalence.

## Baselines

| Name | Rule |
|---|---|
| `seasonal` | Region x day-of-year ignition rate (±15 days). |
| `climatology` | Cell rate from training ignitions summed over the 4-ring (about 20 km), shrunk toward the region rate, x the seasonal profile. |
| `wx_<index>` | One BCWS fire-weather variable from the previous day, alone. |
| `human_hib_frac`, `human_neg_dist` | Share of the cell inside the WUI human interface buffer, and minus the distance to it. |
| `recent_ignitions` | 10 x ignitions in ring 1 yesterday + 3 x ring 1 in the last 7 days + ring 2 in the last 7 days. |
| `clim_x_fwi` | Climatology x observed/expected multiplier by FWI decile, fit on the training years. |
| `clim_x_fwi_x_recent` | ... x observed/expected multiplier by recent-ignition bin. |

## Results: all ignitions

| Baseline | 2022 AP (lift) | 2023 AP (lift) | 2023 AUC | 2023 recall @20 | 2023 recall @100 |
|---|---:|---:|---:|---:|---:|
| **clim_x_fwi_x_recent** | **0.0047 (17.0x)** | **0.0028 (6.7x)** | **0.80** | **4.2%** | **11.8%** |
| recent_ignitions | 0.0037 (13.4x) | 0.0020 (4.8x) | 0.60 | 4.2% | 11.5% |
| clim_x_fwi | 0.0022 (7.9x) | 0.0018 (4.2x) | 0.79 | 1.6% | 7.6% |
| climatology | 0.0017 (6.1x) | 0.0015 (3.7x) | 0.75 | 1.9% | 8.3% |
| climatology refit on 2018-2022 | - | 0.0014 (3.4x) | 0.76 | 1.8% | 7.4% |
| wx_temp_max (best single weather variable) | 0.0014 (5.2x) | 0.0012 (2.9x) | 0.75 | 1.0% | 4.0% |
| wx_fwi | 0.0010 (3.6x) | 0.0008 (2.0x) | 0.70 | 0.4% | 3.6% |
| human_neg_dist | 0.0007 (2.6x) | 0.0010 (2.4x) | 0.73 | 1.2% | 5.9% |
| seasonal | 0.0004 (1.5x) | 0.0007 (1.8x) | 0.67 | 0.8% | 2.8% |

## Results by cause and size (2023 lift, AUC)

| Target | Positives | Best baseline | Climatology | FWI alone | Human proxy |
|---|---:|---|---:|---:|---:|
| Lightning | 1,543 | clim_x_fwi_x_recent 7.9x, 0.82 | 3.9x | 2.0x | 1.4-1.8x |
| Person-caused | 387 | **human_hib_frac 6.8x; human_neg_dist AUC 0.89** | 3.2x | 2.0x | 6.5-6.8x |
| >= 1 ha | 572 | clim_x_fwi_x_recent 5.2x, 0.79 | 3.1x | 2.2x | 1.2-1.5x |

The 2023 lift of the combined baseline by region ranges from 4.9x (Kamloops) to 12.9x (Northwest).

## Findings

1. **The bar to beat is `clim_x_fwi_x_recent`: 2023 AP 0.0028 (6.7x prevalence), AUC 0.80, 11.8% recall at 100 alerts per centre per day.** It needs no learning beyond binned rates.
2. **Lightning storms are the strongest short-term signal.** Recent nearby ignitions alone reach 4.8x lift and match the combined rule at small alert budgets. Their AUC is only 0.60 because most rows have no recent ignition nearby. A lightning-strike feed should help most (Task 4).
3. **Weather alone is weak; weather on top of location and season helps.** FWI alone gives 2.0x. Multiplying it into climatology lifts AP 14% in 2023 (31% in 2022). Previous-day maximum temperature and DMC/BUI (fuel dryness) beat FWI itself as single variables.
4. **Human access predicts human-caused fires, not lightning fires.** For person-caused ignitions, settlement-buffer coverage gives 6.8x lift and distance gives AUC 0.89, better than every other baseline. A learned model should combine human features (for person-caused fires) with weather and recent ignitions (for lightning fires), or model the two causes separately.
5. **Absolute skill is low; this is a hard target.** Even the best rule finds about 12% of ignitions with 100 alerts per centre per day. 2023 lift is lower than 2022, partly because the 2023 season was unusual (early extreme drought and lightning busts in the northeast). Adding 2022 to the climatology fit did not help.
6. These numbers are **not comparable** with Experiments 01-09: the target changed from next-day FIRMS hotspot activity (mostly persistence) to new ignitions.

## Head-to-head with Experiments 02-07 on new fires (2023)

`compare_previous.py` scores the saved 2023 predictions of every earlier model on the
new ignition target. Old resolution-7 scores are aggregated to resolution 6 by their
maximum. It uses the same unmasked rows and metrics as above. Output:
`results/compare_previous_2023.csv`.

| Model | AP lift | AUC | Recall @20 | Recall @100 |
|---|---:|---:|---:|---:|
| **NEW clim x FWI x recent ignitions** | **6.7x** | **0.80** | **4.1%** | **11.8%** |
| NEW climatology | 3.7x | 0.75 | 1.9% | 8.3% |
| 05 CatBoost full data (best old) | 2.1x | 0.60 | 1.4% | 4.6% |
| 06 CatBoost sampled | 1.7x | 0.60 | 1.2% | 4.6% |
| 03 boosting / logistic + neighbours | 1.5x | 0.60-0.63 | 1.0-1.1% | 3.8-4.6% |
| 07 TabPFN hybrid | 1.4x | 0.60 | 1.2% | 4.5% |
| 04 GNN / MLP | 1.3x | 0.55-0.57 | 1.4-1.5% | 4.5-4.9% |
| 02 persistence | 1.1x | 0.51 | 1.3% | 3.7% |

Previous models find new fires at close to random (AUC 0.51-0.63). The new baseline
finds 2.4-3.2x more new fires at the same alert budget. The gap holds for lightning,
person-caused and >= 1 ha fires.

There is one partial exception. On **satellite-visible** new fires (tier A, 475), old
CatBoost has the highest AP (13.5x), because a few of its top scores land on new fires
next to recent FIRMS activity. Its broader ranking is still weaker (AUC 0.62 vs 0.82;
recall @100 6.7% vs 9.9%). FIRMS neighbourhood history therefore deserves a place as a
feature in Task 5.

Outputs in `results/`:

- `metrics.csv`, `metrics_by_region.csv`
- `climatology_tuning_2022.csv`, `fwi_multipliers.csv`, `recent_multipliers.csv`, `selected.json`
- `baselines.png`

Runtime is dominated by the alert-budget metric over about 4.8M rows per year x 37 score/target pairs.
