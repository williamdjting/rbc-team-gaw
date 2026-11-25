# Experiment 39 results: new-fire-only model (2023, quiet cells)

**Answer.**
- **The new-fire-only MLP with focal loss is the best watch-list model so far.** With 50 cells per Fire Centre per day
  it catches **12.2%** of new fires, against 10.0% for the best general model (+1.0 to +3.3 points, 95% interval).
- **It is better across every list size,** and it also wins on BC-wide top-k precision (+0.2 to +1.0 points at
  top 100 per day).
- **Imbalance-aware CatBoost helps a little but not significantly:** weighted cross-entropy 10.9%, 2 non-fires per
  fire 10.8%.
- **Total F1 is unchanged (0.631).** Even a dedicated new-fire model never becomes confident enough to flag a quiet
  cell in a yes/no forecast. New fires stay a ranking problem.

**Setup.**
- Quiet cells only: no detection in the cell or its neighbours for 7 days.
- Inputs: fire history, lightning and jump features (current data only).
- Train 2018–2021 (6,333 new fires), test 2023: 33.8M quiet cell-days, 5,332 new fires, base rate 0.016%.
- New-fire groups (2023): 3,697 jumps of older BCWS fires, 984 genuine ignitions, 651 other burns.

## Watch list: top N quiet cells per Fire Centre per day

| Model | AP | Top 25 | **Top 50** | Top 100 | Top 200 | Listed cells that burned (top 50) |
|---|---|---|---|---|---|---|
| Reference: stacked + jumps (general model) | 0.0292 | 5.9% | 10.0% | 15.6% | 24.6% | 1.2% |
| Q_pop: CatBoost, no imbalance handling | 0.0256 | 6.3% | 9.8% | 15.6% | 24.1% | 1.2% |
| Q_wce: CatBoost, weighted cross-entropy | 0.0288 | 6.6% | 10.9% | 17.8% | 27.3% | 1.3% |
| Q_1to2: CatBoost, 2 non-fires per fire | 0.0262 | 6.9% | 10.8% | 16.9% | 25.5% | 1.3% |
| **Q_focal: MLP, focal loss** | **0.0315** | **7.4%** | **12.2%** | **19.5%** | **28.8%** | **1.5%** |

Difference from the reference at top 50 (week bootstrap, 95%):

| Model | Interval |
|---|---|
| Q_focal | **+1.0 to +3.3 points** |
| Q_wce | −0.3 to +1.9 points |
| Q_1to2 | −0.3 to +1.9 points |
| Q_pop | −1.1 to +0.5 points |

**By new-fire group (top 50 per Fire Centre per day):**

| Model | Genuine ignitions | Jumps of older fires | Other burns |
|---|---|---|---|
| Reference | 5.2% | 12.3% | 4.0% |
| Q_focal | **6.4%** | **14.6%** | **6.9%** |

## The model's most confident predictions: what % are real new fires

BC-wide, per day (the top k quiet cells across all of BC each day):

| k per day | Reference | Q_wce | Q_focal | New fires covered (Q_focal) |
|---|---|---|---|---|
| 10 | 6.0% | 7.5% | 6.5% | 1.8% |
| 100 | 3.7% | 4.0% | **4.3%** | 11.8% |
| 1,000 | 1.4% | 1.5% | 1.6% | 42.8% |

Q_focal − reference at top 100 per day: **+0.2 to +1.0 points**.

Whole season (the k most confident quiet cell-days of 2023):

| k | Reference | Q_pop | Q_wce | Q_1to2 | Q_focal |
|---|---|---|---|---|---|
| 100 | 23% | 19% | 19% | 27% | 20% |
| 1,000 | 12.1% | 12.2% | 13.6% | 12.2% | 11.5% |
| 10,000 | 7.3% | 6.6% | 6.6% | 6.0% | 7.4% |
| 50,000 | 3.1% | 2.8% | 3.2% | 2.9% | 3.4% |

- **Against random:** a random quiet cell burns 0.016% of the time, so the top 1,000 are about 750× random.
- **Not enough for a yes/no flag:** even the most confident cells are mostly not fires.

## Total F1 if quiet cells are decided by the new model
- All four models: the F1-optimal threshold chosen on 2022 flags **no** quiet cells.
- Total F1 stays **0.631**, the same as the best stacked model.

## Reading
- **Sam's hypothesis partly holds.**
  - A model trained only on new fires, with an imbalance-aware loss, ranks new fires better than the general model.
  - Focal loss works best: +2.2 points at top 50, +3.9 at top 100, +4.2 at top 200.
  - It helps every group, including genuine ignitions (5.2% → 6.4%).
- **The gain is modest, and the ceiling is in the data.**
  - Even the best list puts only about 1 in 70 listed cells on a real fire.
  - The F1 threshold never selects a quiet cell.
  - With current inputs (fire history, lightning at 0.1°, nearby fires), new fires are a prioritisation problem, not
    a yes/no forecast.

## Caveats
- One seed per model. The MLP's validation AP moved between 0.348 and 0.365 across epochs, so its result has some
  seed noise.
- The 1:2 sample uses random non-fires. The paper samples them across FWI levels.
- The new-fire groups are approximate (BCWS points and perimeters).
