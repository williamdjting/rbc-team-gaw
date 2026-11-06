# Findings: fusing FIRMS and BC incidents into an ignition label

Scope: 2018-2023, FIRMS VIIRS S-NPP vegetation detections (type 0) inside BC,
H3 resolution 7; BC `Fire` records inside BC with usable dates (Experiment 09
audit). Targets start May 8 so every onset has a full 7-day lookback. 2024 excluded.

## 1. Most FIRMS positives are later days of fires that are already burning

Each FIRMS positive cell-day was attributed to a BC incident that was burning that
day and whose point lies within 1.5 x its final-size equivalent radius + 5 km.

| Year | FIRMS positive cell-days | Day 0-1 of a BC fire | Day 2-7 | Day 8+ | No BC fire | Top 5 fires' share | Top 20 fires' share |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2018 | 27,012 | 1.5% | 5.7% | 53.2% | 39.6% | 28% | 47% |
| 2021 | 16,643 | 3.4% | 10.8% | 69.6% | 16.1% | 32% | 58% |
| 2022 | 3,186 | 6.0% | 14.2% | 60.4% | 19.3% | 35% | 56% |
| 2023 | 45,861 | 1.9% | 11.7% | 71.8% | 14.6% | 37% | 57% |

In big years, 20 fires generate about half of all positives, and fewer than 7% of
positives are on a fire's first two days. A cell-day model is therefore mostly
learning "this fire is still burning", which is why it converges on persistence.

## 2. BC timestamps are local clock time, not UTC

The `IGNITION_DATE` epoch values encode Pacific local time as if they were UTC.
Raw hours bottom out at 01-07 and peak at 14-16 for both lightning and person
causes (the natural afternoon peak). The Lytton Creek fire (K71086, started about
16:40 PDT on June 30, 2021) is stored as 16:40 "UTC". So the date part is the
**local** ignition date. VIIRS S-NPP passes over BC at roughly 01-06 and 11-15
local, on the same UTC and local calendar date, so BC and FIRMS dates align
without shifting. Do **not** convert the BC time from UTC to Pacific.

## 3. BC records are incomplete before 2021

| Year | Local Fire records | Published BC total | Coverage |
|---|---:|---:|---:|
| 2018 | 637 | 2,117 | 30% |
| 2019 | 660 | 825 | 80% |
| 2020 | 571 | 670 | 85% |
| 2021 | 1,514 | 1,647 | 92% |
| 2022 | 1,703 | 1,801 | 95% |
| 2023 | 2,279 | 2,293 | 99% |

The 2018 records that are present skew toward large fires (104 of 637 exceed 100 ha).
Treat 2018 BC "no ignition" as unknown, not as negative.

## 4. FIRMS sees large fires quickly but misses most BC ignitions

These are BC ignitions with a FIRMS detection within 5 km during days -3 to +14
(6,206 incidents with a complete window).

| Final size (ha) | Incidents | Seen by FIRMS | Median first-detection lag |
|---|---:|---:|---:|
| <=0.01 | 2,574 | 10% | 2 d |
| 0.01-0.1 | 914 | 9% | 1 d |
| 0.1-1 | 1,167 | 14% | 1 d |
| 1-10 | 634 | 42% | 1 d |
| 10-100 | 373 | 75% | 1 d |
| 100-1000 | 301 | 88% | 2 d |
| >1000 | 243 | 94% | 2 d |

- 75% of BC ignitions are under 1 ha and are essentially invisible to VIIRS.
  BC is the only source for these.
- Lag: the mode is +1 day. Ignitions recorded after 17:00 local appear in FIRMS on
  the same day only 8% of the time, versus 40% for 05-11 starts. About 9% of
  matched fires show FIRMS activity 1-3 days **before** the BC date, which suggests
  late discovery or holdover fires.

## 5. FIRMS "onsets" remove the repetition, but only about 20% are BC ignitions

An onset is a positive cell-day with no detection within the 2-ring (about 5 km)
during the previous 7 days. Onsets are 7,532 of 94,655 in-season positive
cell-days (8%). Sensitivity: 6-27% across ring 0-3 and 3-14 day lookbacks
(`onset_sensitivity.csv`).

| Onset class | Count | Grows (>=3 cell-days in 14 d) | Median FRP |
|---|---:|---:|---:|
| Matches a BC ignition (5 km, -3..+5 d) | 1,542 | 60% | 8.9 MW |
| Spread or flare-up of a BC fire >5 days old | 2,611 | 71% | 8.1 MW |
| Near a recent BC fire | 421 | 95% | 19.6 MW |
| Unexplained (FIRMS only) | 2,958 | 38% | 4.1 MW |

Unexplained onsets that **fade** are low-FRP and peak in May and September
(92% of May ones fade). That is consistent with managed, slash or agricultural
burning, so mask them. Unexplained onsets that **grow** cluster in August 2018,
consistent with the missing 2018 BC records, so they are probable unrecorded fires.

## 6. Ignition labels are far more stable across years

| Year | FIRMS cell-days (current label) | BC ignitions | Fused ignition events |
|---|---:|---:|---:|
| 2018 | 27,012 | 568 | 1,134 |
| 2019 | 1,125 | 540 | 629 |
| 2020 | 828 | 440 | 452 |
| 2021 | 16,643 | 1,320 | 1,434 |
| 2022 | 3,186 | 1,401 | 1,435 |
| 2023 | 45,861 | 2,046 | 2,352 |

FIRMS positives vary 55x between years; fused ignition events vary 5x. The
current 2022 validation year (3,186 positives) looks nothing like 2023 (45,861).
With ignition labels, 2022 (1,435) and 2023 (2,352) are comparable, so thresholds
chosen on 2022 should transfer better.

## Prototype fused label (`results/fused_ignition_events.csv`)

| Tier | Definition | Events |
|---|---|---:|
| A | BC ignition with a FIRMS detection at 5 km, day -3..+5 | 1,272 |
| B | BC ignition never seen by FIRMS (mostly <1 ha) | 5,043 |
| C | Unexplained FIRMS onset that grows | 1,121 |
| Mask | Spread/flare-up onsets, near-recent-fire onsets, fading FIRMS-only onsets | 4,869 cell-days |

Event date is the BC local ignition date for A/B and the FIRMS onset date for C.
Tier C and the masks use future information (growth, final size) for **label
construction only**. Never use them as model inputs.

## Limits

Attribution uses point locations and final size, not daily perimeters. Large or
elongated fires can exceed the attribution radius, which pushes some spread into
"unexplained", and adjacent fires can be merged. The onset threshold (2-ring,
7 days) is a choice; see the sensitivity table. Tier C is a probable fire, not a
verified one. Validate against BC fire perimeters (`PROT_HISTORICAL_FIRE_POLYS_SP`)
before final use.
