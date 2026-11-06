# Initial baseline diagnostic: 2023

Run against Experiment 03 gradient boosting with cell + neighbour features,
using its original per-region thresholds selected on 2022. All six regional
aggregate AP, confusion counts, recall and F1 reproduce the saved Experiment 03
metrics. These are region-cell-days; boundary duplicates are not removed.

| Previous seven-day activity | Positive region-cell-days | True positives | False negatives | Pooled recall |
|---|---:|---:|---:|---:|
| Cell itself | 33,362 | 25,559 | 7,803 | 76.6% |
| Neighbours only | 7,981 | 1,082 | 6,899 | 13.6% |
| Neither | 4,952 | 0 | 4,952 | 0.0% |

The current thresholded model mostly catches continuation in recently active
cells. It catches few neighbour-only positives and no quiet positives. Zero recall
at these thresholds does not imply zero ranking signal or prove that the sampling
correction is incorrect. Quiet positive activity is not necessarily a new ignition.

The next comparison should test whether CEMS improves ranking and recall within
the neighbour-only and quiet groups, using identical training rows. Local CEMS
weather and prediction files are missing, so no CEMS comparison has been run here.

The run selects 100 false positives and 100 false negatives across region/activity
strata for manual review. Maps use recorded detections within two H3 rings from
two days before to two days after each case. They do not establish cloud coverage,
true fire absence, or ignition. Review annotations are initially blank.

Generated details: `results/baseline/activity_metrics.csv`, `error_review.csv`,
and `REPORT.html` with adjacent map PNGs. See README for reproduction commands.

## Quiet-cell score and budget analysis

The original selected thresholds catch no quiet positives, but the model ranks
quiet candidates better than random selection within the same region-day pool.

| Quiet alerts per centre per day | Expected recall | Expected precision | Random recall |
|---|---:|---:|---:|
| 10 | 2.40% | 1.36% | 0.023% |
| 50 | 6.77% | 0.76% | 0.114% |
| 100 | 10.26% | 0.58% | 0.228% |
| 500 | 18.95% | 0.21% | 1.142% |
| 1,000 | 22.75% | 0.13% | 2.285% |

Budgets are separate for each of the six centres, on every evaluation day. Ties
at the cutoff use expected uniform random selection; reported TP may be fractional.
This is a quiet-only allocation, not an overall model alert budget. No budget was
selected by optimizing 2023 outcomes. Precision includes alerts on days without positives.

The maximum quiet-positive score per region is 0.017–0.030, below all existing
regional thresholds (0.113–0.332). Thus zero thresholded recall does not establish
zero ranking signal. Increasing quiet alerts recovers some positives at a substantial
false-alert cost; improved discrimination is still needed. CEMS comparison remains pending.

`Result.html` includes score distributions, threshold comparisons and budget curves.
Detailed exports are `quiet_score_summary.csv`, `quiet_score_histograms.csv`,
`quiet_daily_budgets.csv`, `quiet_budget_summary.csv`, and `quiet_analysis_manifest.json`.
