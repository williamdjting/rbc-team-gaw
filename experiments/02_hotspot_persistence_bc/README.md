# Experiment 02: BC hotspot persistence

Repeats Experiment 01 independently for all six BC Fire Centres: Cariboo,
Coastal, Kamloops, Northwest, Prince George, and Southeast. These are regional
boundaries, not city limits. The experiment runner does not generate HTML automatically.

Run from the repository root:

```powershell
python experiments/02_hotspot_persistence_bc/run_experiment.py
```

After all six regions finish, build the standalone report:

```powershell
python experiments/02_hotspot_persistence_bc/build_report.py
```

`RESULTS.html` embeds the regional maps and metric plots for offline viewing.

Use `--resume` to reuse finished regions with matching input/code hashes. Optional
`--regions Kamloops` runs one region; `--batch-size 2000` controls memory use.
Dependencies match Experiment 01: pandas, numpy, h3 4.5.0, shapely 2.x,
matplotlib, scipy, scikit-learn and pyarrow. Shared temporal functions are imported
from `01_hotspot_persistence_kamloops/run_persistence.py`.

- Same fixed boundary-derived resolution-7 grids; no detection-based grid selection.
- Same May-September 2018-2023 observations, unknown coverage and no added quality filters.
- Same daily tables and past-only local/neighbour 1-, 3-, 7-day features, censored history age.
- Same next-day recorded-activity proxy target; coverage-supported labels remain missing.
- Same May 8-September 30 target dates, with seven-day warmup and no cross-season history.
- Same fixed positive-count rules; no learned model or GNN.
- 2018-2021 diagnostics, select each region's rule on 2022 F1, evaluate on 2023.
- 2024 excluded; 2023 remains development evaluation, not an untouched final test.

The larger grids are processed in cell batches with complete seasonal histories.
Metrics use exact score histograms, validated against scikit-learn. Kamloops
metrics and sample features must reproduce Experiment 01 before proceeding.

`regions/<centre>/` contains data, feature Parquet files, predictions, checks,
metrics, manifests and figures (fixed grid, daily counts, sample cells, monthly
and August weekly maps for 2018/2023, persistence comparisons).
`results/` contains all-region metrics, selected-rule results and boundary audits.
`figures/` contains the cross-region comparison.

Each region keeps its original inside/on-boundary filter. Cached simplified
boundaries can leave gaps or overlaps; these are audited rather than silently
reassigned. Border H3 cells may occur in more than one grid, with each region
counting its own in-boundary detections. Regional cell-days are not pooled and
presented as unique BC-wide cell-days. Neighbours outside each region are excluded.

Source hashes, grid coverage, H3 assignments, daily/monthly totals, neighbour
sums, complete Cartesian tables, past-only histories and output row counts are
checked. Zero counts do not confirm no fire; prior-day records are assumed
available without modeling publication latency.
