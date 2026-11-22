# Label v1 vs label v2 (2023, full grid)

**Label v1 (current).** S-NPP or NOAA-20 detection in the cell. One-day gaps are filled.

**Label v2.**
- Label v1, plus 2–4-day gaps filled when the cell is inside a BCWS wildfire's final perimeter (1-ring buffer), or
  within 10 km of its ignition point.
- That fire must have ignited before the gap and not been declared out before the gap ended.
- Built by `build_label_v2.py` → `data/pair_v2_lightning_h3r7/`. Inputs are unchanged.
- 2023 positives go from 60,907 to 72,949 (+20%).

**Models.**
- **Main:** CatBoost on the 21 fire-history and lightning inputs.
- **Stacked:** main inputs plus the next-door spread score, built from fire history and fire-shape features only (no
  wind, rain or terrain), out-of-fold.
- Same pipeline for both labels (`run_label.py`): train 2018–2021, thresholds per Fire Centre on 2022, test 2023.

## Overall

| | Label v1: F1 | P | R | Label v2: F1 | P | R |
|---|---|---|---|---|---|---|
| Rule (fire in cell yesterday) | 0.626 | 0.649 | 0.605 | 0.635 | 0.725 | 0.564 |
| Main model | 0.625 | 0.658 | 0.594 | 0.644 | 0.661 | 0.628 |
| **Stacked model** | **0.627** | 0.640 | 0.614 | **0.645** | 0.677 | 0.617 |

Bootstrap over ISO weeks (2,000 resamples), 95% interval:

| Difference | Label v1 | Label v2 |
|---|---|---|
| Main − rule | −0.005 to +0.001 | −0.002 to +0.018 |
| **Stacked − rule** | −0.002 to +0.005 | **+0.000 to +0.020** |
| Stacked − main | +0.001 to +0.005 | −0.001 to +0.004 |

- **Model gain over the rule:** under label v1 it is +0.001 (no gain). Under label v2 it is **+0.011**, the first
  time the model beats the rule by more than noise (the interval just reaches 0).
- **Scored against label v1 instead,** the v2-trained models score lower (main 0.606, stacked 0.614). They flag the
  gap days, which label v1 counts as false alarms. The two labels define "fire" differently, so each model has to be
  judged on the label it was trained on.

## By fire type (caught / fires; flagged)

| Type | Label v1: rule | v1: stacked | Label v2: rule | v2: stacked |
|---|---|---|---|---|
| Already burning | 36,839 / 36,839 (flagged 56,746) | 35,824 (53,543) | 41,147 / 41,147 (flagged 56,746) | 40,060 (54,202) |
| Next-door spread | 0 / 12,188 | 1,565 (4,822) | 0 / 15,141 | **3,726 (9,081)** |
| Recent fire nearby | 0 / 6,548 | 29 (90) | 0 / 11,329 | **1,188 (3,132)** |
| New, quiet area | 0 / 5,332 | 0 | 0 / 5,332 | 0 |

- **Stopped fires (rule false alarms):** 19,907 under v1, 15,599 under v2. 4,300 "stops" become fires that kept
  burning.
- **Next-door and recent-nearby fires rise under v2** (15,141 and 11,329). Days 2–4 of a filled gap are cells whose
  own last detection was 2–4 days ago.
- **The model learns that such cells inside an active fire are still burning:** it catches 3,726 next-door and 1,188
  recent-nearby fires, against 1,565 and 29 under v1.
- **New fires in quiet areas are unchanged** (5,332) and still not caught.

## Reading
- **Label v2 removes label noise that no model could beat.** Under v1, a fire the satellites missed for 2–4 days
  counted as "stopped" then "new". Under v2 it is one fire.
- **With that noise removed, the models' knowledge pays off.** A big, active fire nearby means this cell is still
  burning. The gap between models and the rule widens from about 0 to about +0.01.
- **Report both labels.** Present label v2 as the more realistic definition of "fire present", with v1 numbers next to
  it for comparison with earlier weeks. Agree the label change with the mentors.
- **Caveat:** BCWS records are less complete in early training years (2018: 70% of short gaps filled, 2019: 10%), so
  the training label is less consistent than the 2023 test label.
