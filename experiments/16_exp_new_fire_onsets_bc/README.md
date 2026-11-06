# Experiment 16: what are the "quiet" new-fire onsets?

Describes every new-fire onset in BC, May–September 2018–2023. An onset is an H3 res-7 cell
detected on day t with no detection in it or its ring-1 neighbours on days t-7 .. t-1 (the
"quiet positive" of Experiments 03–07 and 13–15 at a 1-day lead). Same-day onset cells that
touch are merged into one onset cluster. This is a descriptive analysis; no model is trained.

Each cluster is described by:
- BCWS match within 10 km: a new wildfire (started t-3 .. t+1, by cause), an older wildfire
  not yet declared out, another BCWS record type (nuisance, smoke chase, ...), or nothing on
  record;
- distance to the nearest FIRMS detection in the previous 7 and 30 days;
- how many other onset clusters started the same day within 50 km (lightning busts);
- how many of the next 7 days have a detection in the cluster or its neighbours;
- first-day detections, FRP, night share, low-confidence share and FIRMS type.

Future days are used only to describe onsets, never as model inputs. 2024 is not loaded.

Inputs: `../firms_summer_2018_2024/data/firms_h3_detections.csv` and the BCWS incident
download from `../13_exp_fire_context_bc/data/`.

## Run

```
python analyze_onsets.py
```

## Outputs (`results/`)

- `onset_cells.csv`, `onset_clusters.csv` — one row per onset cell / cluster with the
  descriptors above.
- `summary.json` — headline counts. 5,470 onset clusters (10,682 cells, about 11% of all
  detected cell-days). By BCWS match: 1,264 new wildfires, 1,720 older wildfires not recently
  detected, 25 other BCWS records, and 2,461 with no BCWS record.

These findings motivate Experiment 17 (history features and a non-wildfire flag).
