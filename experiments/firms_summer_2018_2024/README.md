# FIRMS summer experiment: 2018–2024

This experiment merges the new `DL_FIRE_SV-C2_830406.zip` archive with the existing `DL_FIRE_SV-C2_825835.zip` archive. It applies the same May–September and BC boundary rules to both sources.

The preprocessing output is written to `data/bc_firms_summer_2018_2024.csv`. Train, validation, and test splits are intentionally postponed until after the complete dataset has been reviewed.

The original archive is never modified. Processing is chunked so the script does not need to load the complete CSV into memory. Every run writes a manifest containing source hashes, filter rules, row counts, schema checks, and package versions.

## Scope

- Sources: NASA FIRMS VIIRS S-NPP Collection 2 archives.
- Spatial filter: point centre inside or on the cached BC legal boundary.
- Seasonal filter: May 1 through September 30 inclusive.
- Covered period: 2018–2024.
- Split: not applied yet; 2024 remains available for a later holdout decision.
- No confidence, type, FRP, brightness, or missing-value filters are applied.

FIRMS rows are hotspot observations, not unique fires or verified ignition records. The outputs support future hotspot-activity modeling; they do not establish fire-start dates.

## Run

From this experiment directory:

```powershell
python preprocess_dataset.py --config config.toml
```

Use `--dry-run` to validate paths, archive members, boundary geometry, and configuration without writing filtered CSVs.

## Output files

| File | Purpose |
| --- | --- |
| `data/bc_firms_summer_2018_2024.csv` | May–September BC observations for 2018–2024 |
| `data/year_counts.csv` | Counts before and after each filter by year |
| `data/quality_checks.json` | Missing values, duplicates, date and coordinate checks |
| `data/manifest.json` | Reproducibility metadata and source hashes |

## H3 transformation

`transform_to_h3.py` assigns every FIRMS row to an H3 cell and writes:

- `data/firms_h3_detections.csv`: original detection rows plus `h3_cell`.
- `data/firms_h3_cell_date.csv`: one row per H3 cell and acquisition date.
- `data/firms_h3_cell_summary.csv`: one row per H3 cell across the full period.

The default H3 resolution is configured in `config.toml`. This step creates spatial features only; it does not create future labels or train/validation/test splits. Install the `h3` package before running the transformation.

