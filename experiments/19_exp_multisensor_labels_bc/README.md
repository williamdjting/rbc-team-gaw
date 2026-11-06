# Experiment 19: multi-satellite labels

The current label is a VIIRS S-NPP detection in an H3 res-7 cell on a UTC day. One satellite
misses burning cells on many days, which caps every forecast. This experiment builds candidate
labels from more satellites, and from filling one-day gaps, and measures how consistent each
label is.

```powershell
python ../../scripts/download_cwfis_hotspots.py   # once; writes datasets/cwfis_hotspots/
python analyze_labels.py
python build_report.py
```

- **Data:** the CWFIS hotspot archive (Natural Resources Canada, open): VIIRS on S-NPP and
  NOAA-20, MODIS and AVHRR.
  - Kept: BC, May–September 2018–2023, assigned to H3 res-7 cells.
  - NOAA-20 is complete from 2021, partial in 2019 (from July) and 2020, and absent in 2018.
- **Labels:**
  - The current label (FIRMS S-NPP).
  - CWFIS's own copy of S-NPP.
  - S-NPP plus NOAA-20, from FIRMS and CWFIS, or from CWFIS only.
  - S-NPP plus NOAA-20 plus MODIS.
  - Gap-filled versions of the above: a cell-day also counts as positive if the cell is
    positive the day before and the day after.

  The rule and the quiet test use the unfilled label, which is all a forecast can see.
- **Measures (BC-wide and per Fire Centre):**
  - Persistence.
  - The missed-day share.
  - The share of positives in quiet cells.
  - The previous-day rule's F1, on the exact cell and with the one-cell margin.
  - The share of positives inside final BCWS perimeters.
  - CWFIS vs FIRMS date alignment.

`results/` holds `label_metrics.csv`, `cwfis_firms_alignment.csv` and a manifest.
`RESULTS.html` is the standalone report (not published).
