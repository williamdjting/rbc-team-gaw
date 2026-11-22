# Jump features (fire 3–24 km away): results (2023, label v1)

**Answer.**
- **The watch list improves; F1 gets slightly worse.**
- **New-fire ranking:** average precision on quiet cells rises from 0.004 to 0.029, about 7×.
- **Top-50 watch list:** 7.5–7.7% → 9.5–10.0%. Top 100: 11% → 15.6%.
- **Genuine new ignitions** on the top-50 list rise from 0.5% to 4.4–5.2%.
- **Total F1 drops a little:** main 0.625 → 0.619 and stacked 0.627 → 0.622, both significant. The 2022 thresholds trade
  continuing-fire catches for precision and transfer worse.
- **Recommendation:** use the jump features in the **watch-list ranking**, and keep the earlier stacked model for the
  yes/no forecast.

**Inputs added (from S-NPP or NOAA-20 detections up to t−1):**
- rings to the nearest cell burning yesterday, and to the nearest cell burning in the last 3 days (up to 10 rings,
  about 24 km);
- burning cells within 3, 5 and 8 rings, and the change in the 5- and 8-ring counts since the day before;
- FRP within 5 rings.

## Total F1 (thresholds per Fire Centre on 2022)

| Model | F1 | Precision | Recall |
|---|---|---|---|
| Rule | 0.626 | 0.649 | 0.605 |
| Main | 0.625 | 0.658 | 0.594 |
| Stacked | **0.627** | 0.640 | 0.614 |
| Main + jumps | 0.619 | 0.663 | 0.581 |
| Stacked + jumps | 0.622 | 0.651 | 0.596 |

Bootstrap, 95% interval:

| Difference | Interval |
|---|---|
| Main + jumps − main | −0.010 to −0.001 |
| Stacked + jumps − stacked | −0.010 to −0.001 |

With jumps, continuing-fire catches fall: 35,824 → 34,832 for the stacked model.

## Watch list: new fires in quiet areas (top N cells per Fire Centre per day)

**2023 new fires (5,332):**

| Group | Fires | Definition |
|---|---|---|
| Jump of an older BCWS fire | 3,697 | Inside an active BCWS fire's perimeter, more than 2.4 km from its recent detections |
| Genuine new ignition | 984 | BCWS fire started within ±3 days |
| Other burns | 651 | No BCWS wildfire |

**Catch rates:**

| Model | AP | Top 25 | **Top 50** | Top 100 | Top 50: ignitions | Top 50: jumps | Top 50: other |
|---|---|---|---|---|---|---|---|
| Main | 0.0041 | 5.2% | 7.5% | 11.1% | 0.5% | 8.6% | 12.0% |
| Stacked | 0.0042 | 4.8% | 7.7% | 10.7% | 0.5% | 8.8% | 12.1% |
| Main + jumps | 0.0285 | 5.8% | 9.5% | 15.6% | 4.4% | 11.8% | 4.2% |
| **Stacked + jumps** | **0.0292** | 5.9% | **10.0%** | 15.6% | **5.2%** | **12.3%** | 4.0% |

Top-50 gain (95%, week bootstrap): main +jumps −1.1 to +5.1 points, stacked +jumps −0.9 to +5.6 points. The weeks vary
a lot, so these intervals are wide.

**Reading.**
- **Jumps and ignitions:**
  - The model now sees big fires up to about 24 km away and moves listed cells toward them.
  - Jumps caught: 8.8% → 12.3%.
  - Genuine new ignitions: 0.5% → 5.2%. New ignitions cluster around active fires, from the same lightning storms
    and the same dry, windy days.
- **Other burns:** they drop from 12% to 4%. Recurring farmland and industrial sites get pushed off the list by cells
  near wildfires. A separate "recurring site" list, or a few reserved slots, would cover them.
- **At the yes/no threshold:** the jump models flag 147–165 quiet cells and catch 10–19 new fires. Still almost none.

## Most important inputs
- **Stacked + jumps:** days since lightning 52%, FRP within 5 rings 15%, neighbourhood active days 7%.
- **Main + jumps:** FRP within 5 rings 8%, distance to the nearest fire 3%.

## Caveats
- One seed, no tuning.
- **Weak threshold year.** 2022 was a quiet fire year and the thresholds come from it, which may be why the jump
  models' thresholds transfer poorly.
- **Approximate grouping.** The new-fire groups use BCWS points and perimeters and are approximate.

## Threshold tuning (`tune.py`; all choices on 2022, test 2023)

**The per-centre thresholds were the problem.** One pooled BC-wide threshold, also chosen on 2022, is better for
every model. 2022 was a quiet year with very few fires in some centres, so per-centre thresholds overfit.

| Model | Per-centre threshold | **One pooled threshold** |
|---|---|---|
| Main | 0.625 | 0.626 |
| **Stacked** | 0.627 | **0.631** (P 0.644, R 0.618; threshold 0.311) |
| Main + jumps | 0.619 | 0.628 |
| Stacked + jumps | 0.622 | 0.629 (P 0.666, R 0.596) |

**Bootstrap (ISO weeks, 2,000 resamples), 95% interval:**

| Comparison | Interval |
|---|---|
| **Stacked (pooled) − rule** | **+0.002 to +0.008** |
| Stacked + jumps (pooled) − stacked (pooled) | −0.005 to +0.002 (a tie) |

- **First significant gain over the rule on label v1:** the stacked model with a pooled threshold.
- **Stacked + jumps ties it on F1,** so one model could serve both the yes/no forecast and the watch list.

**Other strategies:**

| Strategy | F1 |
|---|---|
| Rule volume: flag as many cells per centre-day as the rule | 0.624–0.625 |
| Routed: old flags on burning cells, jump model elsewhere, per-centre | 0.624 |
| Routed, pooled | 0.629 |

None beat the pooled threshold.

## Watch-list curve (quiet cells, new fires, 2023)

| Cells per centre per day | Stacked: caught | Stacked + jumps: caught | Stacked + jumps: listed cells that burned | Lift vs random |
|---|---|---|---|---|
| 10 | 2.6% | 3.0% | 1.8% | 116× |
| 25 | 4.8% | 5.9% | 1.4% | 92× |
| 50 | 7.7% | 10.0% | 1.2% | 77× |
| 100 | 10.7% | 15.5% | 0.95% | 60× |
| 150 | 13.1% | 20.8% | 0.84% | 54× |
| 200 | 15.0% | 24.6% | 0.75% | 48× |

The jump features' gain grows with list size: +2.3 points at 50, +4.8 at 100, +9.6 at 200.
