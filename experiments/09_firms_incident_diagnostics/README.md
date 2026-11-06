# Experiment 09: FIRMS clusters and BC incident locations

Compare observed spatial concentration without confusing thermal observations,
positive cell-days, occupied cells, and incident starts. Outputs are a standalone
`Result.html`, CSV diagnostics and PNG plots. The companion notebook is
`notebooks/04_firms_bc_incident_diagnostics.ipynb`.

```powershell
& C:/Users/sgurl/anaconda3/python.exe experiments/09_firms_incident_diagnostics/run_diagnostic.py
```

Inputs: the FIRMS+CEMS joined detection CSV (weather columns are not used), the
downloaded BC 2018-2023 incident CSV, BC legal boundary and cached Fire Centre
boundaries. Primary comparison uses 2018-2023 May-September, identical BC geometry,
H3 resolution 7, FIRMS type 0, and BC records classified as Fire with usable dates
matching FIRE_YEAR. 2024 rows in the FIRMS input are filtered out before analysis.
All maps use geographic point membership, not assignment by incident centre code.

## Diagnostics

- Scope and missing/invalid date audit by fire year and incident type.
- FIRMS type/confidence/satellite composition and filter sensitivity.
- 2023 BC and Kamloops maps: raw FIRMS intensity, unique FIRMS occupied cells,
  and BC incident cells at identical scale. Monthly Kamloops presence maps.
- Annual occupied-cell footprint, top-cell concentration and effective cell count.
  Effective count is inverse Simpson concentration: 1 / sum(cell_share squared).
- H3 resolution 5/6/7 spatial overlap: whole-season spatial overlap only, not event matching.
- Equal-size occupied-cell subsamples (up to 1,000, 50 seeds) compare nearest-neighbour
  distances after removing repeated observations. Resampling spread is descriptive;
  these are not independent-fire confidence intervals or a formal significance test.
- Incident proximity sensitivity: 1/3/5/10 km; days 0-7, 0-14, 0-30, -3 to +14,
  all inclusive. Every window uses the same May 4-August 31 incident cohort so
  the full matching interval lies inside May-September observation availability.
- Primary proximity check: within 5 km and days 0-14; counts split by year and
  recorded CURRENT_SIZE. Size is retrospective and source units are not assumed.
- Retrospective 2023 H3 cell-day components: spatial neighbours with 1/3/7-day
  link gaps; includes singletons. Exported primary component IDs use a 3-day gap.

## Interpretation limits

No nearby FIRMS record does not prove an incident is wrong or a satellite missed
it: timing/location uncertainty, source coverage, fire duration and other factors
can all contribute. A nearby detection can come from a different fire. Final size
is used only to describe visibility patterns, never as a forecasting input.

Connected components are provisional spatial-temporal groups, **not unique fires**.
Chains can merge fires; observation gaps can split one fire. Component IDs are
retrospective and must not be used as features available before ignition.

The source incident file is not presumed a complete census; substantial missing
dates/record-type exclusions can affect geography. Only valid-dated records enter
time-specific comparisons. Empty map cells do not establish verified fire absence.
Date matching uses UTC days, consistent with FIRMS; incident timezone semantics
remain a source-validation question. No labels are replaced by this diagnostic.

`reference_annual_totals.csv` transcribes 2018-2023 fire totals from the
[BC annual statistics page](https://www2.gov.bc.ca/gov/content/safety/wildfire-status/about-bcws/wildfire-statistics/wildfire-averages),
checked November 4, 2026 (page updated February 3, 2026). These are a completeness
cross-check, not interchangeable row-level ground truth. The page describes an
April-March reporting year, whereas the main diagnostic uses UTC calendar dates.
The separate `results/service_count_check.json` records a live read-only service
check: 739 records for FIRE_YEAR=2018 and 9,104 records for 2018-2023, matching
the local download. The main offline runner does not repeat network requests.

Official references: [BC incident service](https://delivery.maps.gov.bc.ca/arcgis/rest/services/whse/bcgw_pub_WHSE_LAND_AND_NATURAL_RESOURCE/MapServer/22),
[NASA FIRMS fields](https://firms.modaps.eosdis.nasa.gov/content/descriptions/FIRMS_VIIRS_Firehotspots.html),
[NASA detection limitations](https://firms2.modaps.eosdis.nasa.gov/alerts/notes.php).

```powershell
& C:/Users/sgurl/anaconda3/python.exe -m unittest discover -s experiments/09_firms_incident_diagnostics -p 'test_*.py' -v
```
