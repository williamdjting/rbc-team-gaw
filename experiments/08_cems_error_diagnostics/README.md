# Experiment 08: CEMS activity and error diagnostics

Determine where weather helps next-day recorded hotspot forecasting before adding
more predictors. Uses gradient boosting with local and neighbour history, the
existing 2018–2021 training / 2022 threshold selection / 2023 development split,
and no 2024 evaluation. Earlier experiments are read-only inputs.

## Inputs and current availability

The local checkout has all 36 regional feature Parquets and Experiment 03 baseline
predictions. It lacks Experiment 03a prediction/metric files and the complete CEMS
weather CSV. HTML summaries cannot substitute for per-row predictions.

Run from the repository root (use a Python environment with numpy, pandas, scipy,
scikit-learn, pyarrow, joblib, h3 and matplotlib):

```powershell
python experiments/08_cems_error_diagnostics/run_experiment.py --mode check
python experiments/08_cems_error_diagnostics/run_experiment.py --mode saved --variants firms --maps --output experiments/08_cems_error_diagnostics/results/baseline
```

On this machine the working interpreter is `C:/Users/sgurl/anaconda3/python.exe`;
the bare `python` command is a Windows Store alias.

## Paired weather experiment

Supply either `datasets/dataset_CEMS_18_23.csv` or
`datasets/dataset_CEMS_18_24.csv` with `valid_time`, `latitude`, `longitude_deg`,
`fwinx`, `ffmcode`, `drtcode`, and `infsinx`. Prefer the 2018–2023 file. The shared
loader reads a supplied CSV before filtering years below 2024; 2024 values never
enter training, model selection or evaluation. No detection-only joined CSV is needed.

```powershell
python experiments/08_cems_error_diagnostics/run_experiment.py --mode paired --seed 42 --maps --output experiments/08_cems_error_diagnostics/results/paired_42
```

This fits two pooled models from one shared sample, keeping every positive and
every locally/neighbour-active row plus 1% of quiet negatives. Keys are sorted
before seeded sampling. Both fits use exactly the same labels and rows; only
CEMS columns differ. Sample row IDs are saved per region/year. Each model uses
the existing quiet-row odds correction and its own region-specific threshold
selected on all 2022 rows. No activity-specific thresholds are fitted on 2023.
Use separate output directories for seeds 43 and 44 to assess run variability.
`--regions Kamloops` is a smaller smoke run; it trains on Kamloops only and is
not equivalent to the pooled six-centre experiment.

Weather joins reject duplicate or unmatched keys, check unchanged row counts,
and reject nonfinite values. Mapping distance summaries and annual weather
variation are saved. Mapping uses the same nearest-grid approximation as 03a;
large distances require review, not automatic reassignment. Historical weather
timestamps do not establish operational availability at forecast issuance time.

Alternatively copy your teammate's `03a/results/predictions/*_2023.parquet` and
`03a/results/metrics_by_year.csv` into the corresponding experiment folder and run
`--mode saved` for both variants. Saved mode verifies matching evaluation keys
and labels but cannot establish identical training samples.

## Outputs

Build the standalone, chart-first report after the diagnostic run with `--maps`:

```powershell
python experiments/08_cems_error_diagnostics/analyze_quiet_scores.py
python experiments/08_cems_error_diagnostics/build_report.py
```

`Result.html` is written directly inside this experiment folder. It embeds all
charts and 200 review maps, with expandable galleries, so no adjacent assets are
needed. Use `--input` and `--output` to report on a different run.

The quiet-score analysis reuses saved 2023 predictions and existing 2022-selected
thresholds; it does not retrain or retune. It writes class-specific quantiles and
log-score histograms, daily budget details, and regional/pooled budget summaries.
Default budgets are 10, 50, 100, 500 and 1,000 quiet alerts **per centre per day**;
the six-centre total daily budget is six times that value. Candidate pools contain
quiet cells only. The random comparator uses the same region-day pool and alert
count. Recall is total expected TP divided by total quiet positives; precision is
total expected TP divided by total alerts, including alerts on days with no positives.

Equal-score cutoff ties use expected uniform random selection, not label-based
or file-order tie breaking. Expected TP/FP can therefore be fractional. These are
descriptive budget curves, not a budget optimized on 2023. Score values at or below
1e-12 are clipped only in histogram display; ranking uses original probabilities.
For paired runs, pass the paired result folder with `--input` to both analysis and
report commands. CEMS appears only when its predictions actually exist in the run.

- `activity_metrics.csv`: all rows plus three disjoint groups: local activity in
  the preceding seven days; neighbours only; neither. Includes AP, positive count,
  prevalence, precision, recall, F1 and confusion counts. AP/recall are undefined
  when a group has no positives. Compare changes within each group, not raw AP
  across groups with different prevalence.
- `paired_deltas.csv`: CEMS minus FIRMS AP/recall/F1 when both variants run.
- `error_review.csv`: up to 100 false positives and 100 false negatives, allocated
  across model/region/activity strata, with blank review category/notes columns.
  This deliberate review sample does not estimate population error proportions.
- `case_*.png` with `--maps`: target cell outlined within a two-ring H3 neighbourhood,
  with recorded activity at offsets -2 through +2 days. Future days are used only
  for retrospective review. Outside-region or unavailable dates are marked missing.
- `REPORT.html`: offline metrics and review maps; keep it beside its PNG files.
- `input_inventory.csv`, `manifest.json`, and paired-mode mapping, variation,
  model and shared-sample artifacts for provenance.

Suggested review categories: spatial near-miss, temporal gap, new recorded activity,
isolated detection, unclear. These maps show recorded activity, not confirmed fire
extent. Do not automatically relabel errors. Sensor/confidence/cloud/overpass
information is not present in the feature tables: external observation data is
needed to establish whether a gap is a missed observation.

This experiment does not yet implement rolling-year validation, event/block
bootstrap confidence intervals, ignition verification or automatic label repair.
Use its diagnostics to choose the next investigation; small AP changes are not
automatically statistically significant.

## Checks

```powershell
python -m unittest discover -s experiments/08_cems_error_diagnostics -p "test_*.py" -v
```

Tests cover activity partitioning, unsafe joins, confusion metrics and paired
training/prediction with synthetic inputs. Generated results are Git-ignored.
