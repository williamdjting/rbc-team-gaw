# Experiment 10: FIRMS x BC incident fusion for an ignition label

Builds a new-fire (ignition) label that removes the day-after-day repetition of
FIRMS detections. It does this by combining BC recorded ignitions with FIRMS
onsets, and quantifies how much of the current FIRMS target is repetition.
See `FINDINGS.md`.

```powershell
& C:/Users/sgurl/anaconda3/python.exe experiments/10_ignition_label_fusion/run_analysis.py
```

Inputs: `data/firms_with_cems_18_24.csv` (2024 dropped, weather ignored),
`data/bc_historical_incidents_2018_2023.csv` (timestamp and completeness audit),
and `experiments/09_firms_incident_diagnostics/results/incidents_in_scope.csv`
(run Experiment 09 first). Runtime is about 30 s.

**Frozen label (Week 3, Task 1):** `build_labels.py` → `results/labels_v2/`; see `LABEL_SPEC.md`.
It needs `data/bc_fire_perimeters_2018_2023.geojson` (`scripts/download_bc_fire_perimeters.py`).

```powershell
& C:/Users/sgurl/anaconda3/python.exe scripts/download_bc_fire_perimeters.py
& C:/Users/sgurl/anaconda3/python.exe experiments/10_ignition_label_fusion/build_labels.py
```

Exploratory outputs (v1, point + size attribution) in `results/`:

- `fused_ignition_events.csv`: prototype ignition events (tiers A/B/C), one row per event.
- `fused_label_mask.csv`: cell-days that are neither ignitions nor clean negatives.
- `incident_first_detection.csv`, `first_detection_by_size.csv`: per-incident FIRMS lag and visibility.
- `firms_cell_days_annotated.csv.gz`: every FIRMS cell-day with onset flags and incident attribution.
- `firms_onsets_k2_L7.csv`, `onset_sensitivity.csv`, `onset_growth_by_class.csv`.
- `repetition_breakdown.csv`, `top_incident_share.csv`, `label_tallies.csv`.
- `incident_completeness.csv`, `ignition_raw_hour_by_cause.csv`: source audits.
- `figures/*.png`.
