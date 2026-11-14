# Wildfire forecasting: current experiment summary

**Lightning ablation update:** [Experiment 26](../experiments/26_exp_lightning_ablation/RESULTS.md) compares matched MLPs. On 2024, lightning adds 0.00385 F1 (0.385 percentage points), but the paired 95% interval includes zero. Median centre AP rises from 0.4674 to 0.4765. This is a follow-up on previously inspected 2024, with one seed.

**Update November 12, 2026:** the separate S-NPP/lightning Experiment 25 has now
completed a full-grid 2024 test: exact-cell F1 0.5586 for its retrained MLP versus
0.5670 for S-NPP persistence. It uses different inputs and an S-NPP-only label;
it is not the 2024 evaluation of Experiment 22's two-satellite model. See
[Experiment 25 results](../experiments/25_exp_snpp_lightning_2024/RESULTS.md).
The remaining summary describes the earlier two-satellite experiments.

Updated November 11, 2026. Based on saved reports, metrics and implementation;
training was not rerun for this review. Experiments 21 and 22 provide the latest
completed six-centre comparison. Experiment 23 does not yet provide a complete
six-centre result in the supplied archive. No final 2024 evaluation is available.

## Problem, input and output

Predict which H3 resolution-7 grid cells across British Columbia's six Fire
Centres will have hotspot activity on the next UTC day. Each example is a
cell-day. The six centres are Cariboo, Coastal, Kamloops, Northwest, Prince George
and Southeast.

The forecast uses past satellite observations, nearby incident history, station
weather and lightning. Its output is a model score interpreted as the probability
of the binary target, followed by a thresholded yes/no alert per cell. Probability
calibration has not been established. A map of these scores or alerts is the
natural output product.

The latest experiments forecast hotspot activity, including continuing burning
and spread. They do not identify verified new ignitions, individual fires, fire
size or severity. Experiments 10-12 investigated a separate incident-based
ignition task; those metrics should not be mixed with these results.

## Target label

Let D(c,t) mean at least one S-NPP or NOAA-20 VIIRS detection in cell c on day t.
The target is:

`y(c,t) = D(c,t) OR [D(c,t-1) AND D(c,t+1)]`

Thus a positive means a detection that day, or a single missing day bracketed by
detections. A zero means neither condition is met, rather than confirmed absence
of fire. Cloud, overpass timing and other observation gaps remain limitations.

Gap filling is used only for retrospective labels. Model inputs stop at t-1 and
never use filled future observations. Incident context uses an additional lag,
with dates through t-2. Historical incident availability is assumed by this lag;
it is not an audit of actual reporting timestamps.

For 2021-2023 the label combines both satellites. Training years 2018-2020 use
gap-filled S-NPP alone as a stopgap because the available NOAA-20 archive is
incomplete; NOAA-20-related model inputs are blank before 2021.

## Datasets and features in Experiments 21-23

| Dataset | Use |
|---|---|
| NASA FIRMS VIIRS S-NPP, processed into H3 cells | Labels and past hotspot counts, FRP, night detections, time since detection, neighbourhood and fire-complex activity |
| CWFIS hotspot archive, NOAA-20 VIIRS subset | Second satellite for labels and recent NOAA-20/combined-satellite history |
| DataBC/BCWS historical incident locations | Distance/count of nearby active and recently extinguished fires, incident age, cause and recent nearby starts |
| BCWS weather observations and station locations | Previous-day noon temperature, humidity, wind, rain and FFMC/DMC/DC/ISI/BUI/FWI; three-day rain and station distance |
| CanCPLD cloud-to-ground lightning | Daily and three-day activity at 0.1-degree resolution, surrounding 3x3/9x9 blocks, days since nearby lightning |
| H3 regional grids and Fire Centre assignments | Spatial joins and neighbourhoods; upwind-burning features combine satellite history with station wind |

Weather uses the nearest station with a reading among the three nearest within
100 km. Missing readings remain missing. It is historical observed weather, not
a forecast of the target day's weather. ERA5, CEMS, vegetation and road layers
are not inputs to these final model configurations, even though other project
work explored some of them.

The final tabular configuration has 63 columns including `quiet_stratum`, and
excludes `day_of_year`. The neural preprocessing produces 89 inputs after missing
flags. The GRU additionally receives 14 history days with 25 channels per day.

## Training and evaluation

- Seasons: May-September; target days May 8-September 30 (146 days/year).
- Train: 2018-2021. Select thresholds: 2022. Development evaluation: 2023.
- 2024 remains excluded from these experiments and reserved for final testing.
- Regional grids contain 234,554 cell entries in total. The training tables
  contain 136,979,536 cell-days before sampling.
- Keep all positive and recent-activity rows plus a random 1% of quiet negatives:
  1,759,606 sampled rows, including 53,611 positives. Predictions apply the
  implemented quiet-row sampling correction.
- Neural models reserve one seven-day block in six within training seasons for
  epoch selection, leaving 1,508,981 rows for fitting.
- Main operating point: maximise 2022 F1 separately by centre, model and scoring
  definition, then apply those thresholds unchanged in 2023.
- Additional check: flag the same number of cells each day as the persistence rule.

## Completed six-centre performance

2023 results, pooled across all six centres. Learned models use 2022-selected
thresholds. The two F1 columns use separate scoring definitions and thresholds;
they are not interchangeable measures of accuracy. Precision and recall below
refer to exact-cell scoring.

| Model | Experiment | Exact-cell F1 | One-cell-margin F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|
| S-NPP detection yesterday | Baseline | 0.584 | 0.785 | 0.664 | 0.522 |
| S-NPP or NOAA-20 detection yesterday | Main baseline | 0.624 | 0.810 | 0.645 | 0.603 |
| CatBoost, fire history | 20, rescored in 21 | 0.582 | 0.790 | 0.564 | 0.601 |
| CatBoost, weather/lightning/spread, no day-of-year | 21 | 0.607 | 0.824 | 0.646 | 0.572 |
| MLP, same features | 22 | 0.626 | 0.820 | 0.638 | 0.614 |
| MLP + 14-day GRU | 22 | 0.623 | 0.816 | 0.622 | 0.623 |

Exact-cell scoring requires the predicted cell to match. One-cell-margin scoring
allows spatial agreement within an immediate H3 neighbour, within each centre's
grid. F1 balances precision and recall; 0.824 F1 does not mean 82.4% accuracy.

The MLP improves exact-cell F1 over CatBoost by 0.019 (week-block bootstrap 95%
interval 0.011 to 0.027). Against persistence its gain is only 0.002, with interval
-0.003 to 0.007: no clear exact-cell improvement over the rule. Median per-centre
average precision improves from 0.617 for CatBoost to 0.642 for both networks.

CatBoost has the highest margin F1, improving over persistence by approximately
0.015 (95% interval 0.005 to 0.025). At the rule's daily alert volume its margin
F1 drops to 0.807 versus the rule's 0.810. Its headline margin gain therefore
depends on the operating threshold and number of alerts.

## Experiment 23: preliminary, not a final comparison

The supplied HTML covers Kamloops only and two regularised training epochs.
Its generic 'BC-wide' captions are misleading for that run. In this limited report:

| Kamloops model | Exact-cell F1 | Margin F1 |
|---|---:|---:|
| Original MLP reference | 0.645 | 0.838 |
| Regularised MLP | 0.599 | 0.816 |
| Regularised MLP + GRU | 0.610 | 0.820 |

The separate run log shows a longer run but stops during regional evaluation.
The archive lacks a completed manifest and evaluation tables. These artifacts
do not establish the performance of the intended full regularised experiment.

## What the experiments establish

Adding a second satellite improves the persistence baseline. Removing day-of-year
helps CatBoost transfer between seasons. The MLP improves on CatBoost's exact-cell
forecast, mainly by retaining continuing activity, while the GRU adds no clear
benefit in the completed comparison.

New activity remains difficult: the MLP detects 2,200 of 23,361 positive cell-days
without a previous-day paired detection (about 9.4%). These are newly active cells,
not verified ignition events. Strong aggregate hotspot scores should not be
presented as strong ignition forecasting.

The MLP is the strongest completed learned model by exact-cell F1; CatBoost leads
under spatial tolerance. Persistence remains an essential benchmark. These are
development findings: repeated comparisons on 2023, one seed per neural model,
incomplete early NOAA-20 coverage and station-weather gaps limit generalisation.
Freeze a model and operating point before evaluating the untouched 2024 data.

Source reports: [21](../experiments/21_exp_weather_lightning_spread_bc/RESULTS.html),
[22](../experiments/22_exp_neural_models_bc/RESULTS.html),
[23](../experiments/23_exp_regularised_neural_bc/RESULTS.html).
Experiment 21/22 figures were cross-checked against their saved CSVs and manifests.
Missing preprocessing dependencies and Experiment 23 outputs are documented in
the experiment READMEs.
