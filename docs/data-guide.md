# Where to find the data

The local [data index](../data/README.md) explains every top-level data folder.
Individual files are listed in `data/file_catalog.csv`; a hash-verified relocation
record is in `data/file_relocation_manifest_2026_11_11.json`.

The newest upload is now:

`data/joined_hotspot_forecast_2018_2023/bc_hotspot_forecast_h3r7_sampled_features_2018_2023.parquet`

It contains 2,873,706 sampled cell-days, 78 columns, 2018-2023 and six Fire Centres.
All 63 frozen MLP inputs are present. It is a joined development sample, not raw
weather/lightning data or a 2024 holdout. See its local README and column dictionary.

Other main locations:

- `data/cancpld_lightning/`: downloaded official 2018-2024 lightning and BC daily grids.
- `data/snpp_lightning_h3r7/`: separate Experiment 25 S-NPP/lightning joins.
- `data/raw_archives/`: original uploaded ZIPs, with descriptive names.
- `data/firms_cems/`: earlier detection-level FIRMS/CEMS join, 2018-2024.
- `data/bcws_incidents/`: historical incident points, 2018-2023.
- `data/bcws_perimeters/`: final fire polygons, 2018-2023.
- `data/bcws_weather/`: earlier station weather outputs, 2018-2023.
- `data/ignition_dataset/res6/`: separate incident-based ignition task.
- `data/exploration/`: initial S-NPP EDA and separate 2024 S-NPP holdout copy.
- `data/region_exploration/`, `data/incident_h3_exploration/`, `data/checks/`:
  exploration and quality checks.
- `data/static/`: wildland-urban-interface buffers.

Data remain Git-ignored. Experiment-owned caches and frozen artifacts remain in
their experiment folders. Established generated filenames remain stable; moved
incident/CEMS paths were updated in their scripts, notebook and documentation.
