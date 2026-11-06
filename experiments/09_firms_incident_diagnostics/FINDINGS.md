# Findings: FIRMS clustering versus BC historical incidents

## Main result

FIRMS is more spatially clustered even after removing repeated observations.
It does **not** occupy fewer H3 cells: in the matched 2023 May-September comparison,
FIRMS vegetation detections occupy 9,325 resolution-7 cells; 2,090 BC incidents
occupy 1,999 cells. FIRMS maps represent observed fire activity over time and can
cover many neighbouring cells of the same fire; incident maps represent starts.

| 2023 comparison | FIRMS | BC incidents |
|---|---:|---:|
| Source records | 375,055 detections | 2,090 usable incident starts |
| Occupied H3 cells | 9,325 | 1,999 |
| Weight in top 10% of occupied cells | 32.4% | 13.9% |
| Mean nearest-cell distance, equal samples of 1,000 occupied cells | 6.56 km | 11.74 km |
| Kamloops equal samples of 333 cells | 3.32 km | 6.22 km |

Nearest-distance values average 50 fixed-seed subsamples. They describe spatial
dispersion after matching sample size, not statistical confidence about independent
fire events. Concentration also persists when FIRMS is reduced to unique cell-days:
the top 10% of occupied cells carry 31.0% of cell-day weight.

## Which incidents have nearby detections?

Across 5,908 eligible 2018-2023 incidents, 24.9% have at least one FIRMS vegetation
detection within 5 km during days 0-14 after the recorded start. In 2023 this is
30.0% of 1,949 eligible incidents. All windows use the same May 4-August 31 cohort.

The association with recorded size is strong: 9.5% of incidents in CURRENT_SIZE
(0,1] have nearby activity, versus 93.8% in the >1,000 bin. The small bin contains
4,334 incidents; the largest contains 241. Source size units are not assumed here.
This is consistent with size-dependent visibility, but does not establish its cause.
Date/location uncertainty, fire duration and source coverage remain alternatives.

These are proximity checks, not event assignments or satellite detection-efficiency
estimates. Nearby observations can belong to another fire. Unmatched records are
not relabelled negative or discarded as false incidents.

## How much do clusters dominate?

With adjacent H3 cell-days connected across up to three days, the ten largest
provisional 2023 components account for 42.4% of FIRMS observations. Counts vary
from 3,465 components with one-day links, to 2,107 with three-day links, to 1,326
with seven-day links. Thus cluster identity is sensitive to the rule. Components
can merge fires and observation gaps can split a fire; these are not unique-fire counts.

## Source issue to resolve before making labels

The complete downloaded incident file has only 739 records for FIRE_YEAR=2018,
including non-fire types. A direct service count confirms 739, and the service's
2018-2023 count confirms all 9,104 downloaded rows. This rules out simple local
pagination loss. BC's published total is 2,117 fires for 2018. Source completeness,
definitions and reporting scope must be reconciled before treating this file as
a complete ignition census. The discrepancy predates our spatial/seasonal filters.

The published annual statistics use April-March years. A source ignition date that
does not match FIRE_YEAR is not necessarily erroneous; this comparison excludes
such rows to preserve calendar-year scope. Among Fire records inside BC, only 139
have missing, unparseable or calendar-year-mismatched dates; the earlier larger
missing-date count included non-fire record types.

## Recommended label work

1. Resolve the historical incident inventory discrepancy, especially 2018.
2. Validate incident identities against official fire perimeters and dates, retaining
   ambiguous/unmatched links rather than forcing a nearest-incident assignment.
3. Use audited incident starts for a new-fire task, and FIRMS for an ongoing-activity
   task. Compare per-event and per-cell-day results separately.
4. Treat satellite component IDs as diagnostic grouping only; do not use future
   clustering or final fire size as information available at prediction time.

Artifacts: `Result.html`, `results/*.csv`, `results/figures/*.png`, and
`notebooks/04_firms_bc_incident_diagnostics.ipynb` from the repository root.

Sources: [BC annual totals](https://www2.gov.bc.ca/gov/content/safety/wildfire-status/about-bcws/wildfire-statistics/wildfire-averages),
[BC incident service](https://delivery.maps.gov.bc.ca/arcgis/rest/services/whse/bcgw_pub_WHSE_LAND_AND_NATURAL_RESOURCE/MapServer/22),
[NASA FIRMS attributes](https://firms.modaps.eosdis.nasa.gov/content/descriptions/FIRMS_VIIRS_Firehotspots.html),
[NASA detection limitations](https://firms2.modaps.eosdis.nasa.gov/alerts/notes.php).
