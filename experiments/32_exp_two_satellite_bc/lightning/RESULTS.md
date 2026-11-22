# Holdover lightning × dryness: results (2023, label v1, one pooled threshold on 2022)

**Answer: no meaningful improvement.**
- **Total F1 gets worse:** 0.623 vs 0.629 for stacked + jumps (−0.009 to −0.004) and 0.631 for stacked.
- **Spread ranking is unchanged:** next-door AP 0.254 vs 0.254.
- **Watch list:**
  - small gains at longer lists (top 100: 15.6% → 16.8%; top 200: 24.6% → 26.3%);
  - top 50 is not significant (10.0% → 10.3%; −0.5 to +1.4 points);
  - **no gain for genuine new ignitions** (top 50: 5.2% → 5.0%).

**Inputs added** (known by t−1):
- **Lightning:** flashes in the cell's 0.1° box and the surrounding 3 × 3 boxes over 7 and 14 days, and days since
  lightning in the cell's box.
- **Dryness:** BCWS daily FWI codes at the nearest station (FFMC, DMC, DC, BUI, ISI, FWI).
- **Interactions:** 14-day lightning × DC, 7-day lightning × DMC.

## Total F1

| Model | F1 | Precision | Recall | Continuing caught | Next-door caught |
|---|---|---|---|---|---|
| Rule | 0.626 | 0.649 | 0.605 | 36,839 | 0 |
| **Stacked** | **0.631** | 0.644 | 0.618 | 36,034 | 1,589 |
| Stacked + jumps | 0.629 | 0.666 | 0.596 | 35,120 | 1,163 |
| Stacked + jumps + lightning/dryness | 0.623 | 0.672 | 0.582 | 34,437 | 957 |

## Watch list (5,332 new fires in quiet areas)

| Model | AP | Top 25 | Top 50 | Top 100 | Top 200 | Top 50 ignitions | Top 50 jumps | Top 50 other |
|---|---|---|---|---|---|---|---|---|
| Stacked | 0.004 | 4.8% | 7.7% | 10.7% | 15.0% | 0.5% | 8.8% | 12.1% |
| Stacked + jumps | 0.029 | 5.9% | 10.0% | 15.6% | 24.6% | 5.2% | 12.3% | 4.0% |
| + lightning/dryness | 0.030 | 6.6% | 10.3% | 16.8% | 26.3% | 5.0% | 12.5% | 5.7% |

Bootstrap, 95% interval:

| Difference | Interval |
|---|---|
| Top 50 | −0.5 to +1.4 points |
| Top 50, ignitions only | −1.2 to +0.5 points |

## Why
- **The model leans on the dryness codes, which don't transfer.**
  - FWI codes take 71% of the importance; the extra lightning features take 2%.
  - As with ERA5, station dryness acts as a per-region, per-period fingerprint of the training years.
  - Recall on continuing fires drops.
- **Lightning is still at 0.1° (~10 km).** Longer windows don't fix the location problem: the strike that starts a fire
  is one of many in the box.
- **Dryness is regional.** One station value covers many cells, so it says *when*, not *where*.

## Decision
Keep the earlier models:
- **Yes/no forecast:** stacked, one pooled threshold (F1 0.631).
- **Watch list:** stacked + jumps.

Lightning and dryness at this resolution don't add enough to justify the F1 cost.
