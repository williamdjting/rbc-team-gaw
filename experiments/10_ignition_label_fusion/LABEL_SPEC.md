# Ignition label specification (v2, frozen for Week 3)

Built by `build_labels.py`; outputs in `results/labels_v2/`. Parameters are recorded in
`label_manifest.json`. Change a parameter only by editing the constants at the top of
`build_labels.py` and rerunning (about 25 s).

## Sources

| Source | Role |
|---|---|
| BC historical incidents (`data/bc_historical_incidents_2018_2023.csv`) | Fire starts: local ignition date and point. `Fire` records inside BC only. |
| BC historical fire perimeters (`data/bc_fire_perimeters_2018_2023.geojson`, `scripts/download_bc_fire_perimeters.py`) | Final burned area for fires of about 1 ha and larger (92-99% coverage). Used for attribution footprints, plus 515 fires missing from the incident file (385 in 2018). |
| FIRMS VIIRS S-NPP type 0, H3 resolution 7 | Confirms starts, corrects late BC dates, locates perimeter-only fires, and drives the masks. Not a source of positives by default. |

BC `IGNITION_DATE` is Pacific local clock time stored as if it were UTC. Use the date part as-is.

## Label

**Positive event** (`ignition_events.csv`, `in_default_label == True`): one row per fire.

| Field | Rule |
|---|---|
| `date` | BC local ignition date. If FIRMS saw the fire inside its own perimeter earlier (up to 14 days), the earlier FIRMS date instead (`date_source = firms_earlier_than_bc`, 191 events). The original BC date is kept in `bc_date`. |
| `h3_cell` (res 7) | Incident point. For perimeter-only fires, the first FIRMS cell inside the perimeter, otherwise a point inside the polygon. |
| `tier` | **A**: FIRMS saw the start (within 5 km of the point from 3 days before to 5 days after the date, or inside the perimeter from 14 days before to 5 days after). **B**: FIRMS never saw it, mostly fires under 1 ha. |
| Optional | `size_ha` (final, retrospective), `cause`, `local_hour`, `source` (incident or perimeter_only). |

Tier **C** (FIRMS-only onsets that keep growing) is kept in the file with
`in_default_label == False`. Two rounds of spot-checks (56 maps) found most of them
are gas flares, landfills, mills, mines or fall slash-pile burning, not unrecorded
wildfires.

**Masks**: these cell-days are neither positive nor negative.

- `fire_footprints.csv.gz`: every res-7 cell inside a fire's final perimeter (or its point cell), from ignition to out date (ignition + 150 days if no out date is recorded). The ignition cell-day itself stays positive. Final perimeters over-mask early-fire days; this is deliberately conservative.
- `onset_mask.csv`: FIRMS onsets that are spread or flare-ups, early growth, near an active fire (within 10 km), static sources, late-season (September) FIRMS-only, or fading FIRMS-only.

**FIRMS onset** (used for matching and masks): a positive cell-day with no detection
within its H3 2-ring (about 5 km) during the previous 7 days.

**Static source cell**: a cell whose 1-ring has fire-unrelated detections in 2 or more
seasons, or any FIRMS type-2 detection. 428 cells. This is used only to classify
FIRMS-only onsets; real BC fire starts in these cells (28) are kept.

## Counts (May 8 - September 30)

| Year | Tier A | Tier B | Default total | Fires >= 1 ha | FIRMS positive cell-days (old label) |
|---|---:|---:|---:|---:|---:|
| 2018 | 386 | 567 | 953 | 623 | 27,012 |
| 2019 | 66 | 499 | 565 | 106 | 1,125 |
| 2020 | 60 | 395 | 455 | 84 | 828 |
| 2021 | 333 | 1,010 | 1,343 | 390 | 16,643 |
| 2022 | 160 | 1,270 | 1,430 | 289 | 3,186 |
| 2023 | 499 | 1,550 | 2,049 | 580 | 45,861 |

2018 small fires (< 1 ha) are still under-recorded. The incident file covers 30% of
2018 fires, and perimeters exist only for larger fires. Treat 2018 negatives as
unreliable for small-fire targets.

## Using this label (Task 2)

- Positive: cell-day equal to an event's (`h3_cell`, `date`). For a day-ahead
  forecast, features come from `date - 1` and earlier.
- Excluded: cell-days in `fire_footprints` active windows (except the ignition
  cell-day) and in `onset_mask`.
- Negative: everything else in the season grid.
- Growth, final size, perimeters and out dates are future information. Never use
  them as features.

## Validation summary

- Perimeter attribution leaves only 2.4-3.1% of FIRMS positive cell-days unexplained
  in the big years (point + size attribution left 15-40%). 2019 and 2020 stay higher
  (40-64%) because of small seasons, slash-pile burning and static sources.
- In the big years, 81-85% of FIRMS positive cell-days are day 8 or later of a fire,
  and only 1.5-2.5% are its first two days.
- Spot-check round 1 (36 tier C onsets, before the static and late-season rules):
  about 21 static/industrial, 6 slash-pile, 3 BC date late (now fixed by the
  14-day in-perimeter rule), and 3-6 plausible unrecorded fires.
- Spot-check round 2 (20 onsets, after the rules): mostly remaining flares and
  industrial points, plus same-day neighbour duplicates. Hence tier C is off by
  default. Maps are in `results/labels_v2/spotcheck/`; `tier_c_review.csv` has
  blank verdict columns for a manual pass.
