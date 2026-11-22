# Fuel (land cover) and afternoon station weather: results (2023, two-satellite label)

**Answer.**
- **Land cover alone does nothing.** Total F1 changes by −0.002 to +0.000, and the new-fire watch list gets slightly
  worse.
- **Adding afternoon station weather helps.** The general model goes from 0.625 to **0.629**, a gain of +0.001 to
  +0.008. It ties the stacked model from earlier today (0.629) as the best total F1.
- **Next-door spread:** day-before afternoon wind helps a little (AP +0.001 to +0.021). **The target-day afternoon
  wind helps a lot** (AP +0.028 to +0.062), but that is an upper bound: in practice it needs a weather forecast.
- **New fires in quiet areas:** still none caught at the threshold. The watch list moves from 7.5% to 7.9%, which is
  not significant.
- **Putting everything into one stacked model scores lower** (0.626). Its 2022 thresholds trade recall for precision
  and don't carry over to 2023.

**Data.**
- **Land cover:** ESA WorldCover 2021 shares per cell (`scripts/build_landcover_h3r7.py`).
- **Station weather:** BCWS hourly observations, 12:00–16:00 local time on the day before
  (`build_hourly_wind.py`):
  - mean wind and peak gust;
  - wind direction (as u/v);
  - minimum RH and maximum temperature;
  - maximum hourly ISI and FFMC.
- **Next-door wind alignment:** for next-door cells, how much that wind blows from the burning neighbours toward the
  cell.
- **Setup:** same splits as Exp 32. Week bootstrap with 1,000 resamples.

## Total F1

| Model | F1 | Precision | Recall |
|---|---|---|---|
| Rule: either satellite yesterday | 0.626 | 0.649 | 0.605 |
| G0: general model (Exp 32) | 0.625 | 0.658 | 0.594 |
| G_lc: + land cover | 0.624 | 0.659 | 0.592 |
| **G_lchw: + land cover + afternoon station weather** | **0.629** | 0.643 | 0.615 |
| ST32: Exp 32 stacked | 0.629 | 0.645 | 0.613 |
| ST_full: everything + next-door score | 0.626 | 0.682 | 0.578 |

| Difference (95%) | Interval |
|---|---|
| G_lc − G0 | −0.002 to +0.000 |
| G_lchw − G_lc | **+0.002 to +0.009** |
| G_lchw − G0 | **+0.001 to +0.008** |
| ST_full − ST32 | −0.007 to +0.003 |

F1 by Fire Centre:

| Model | Cariboo | Coastal | Kamloops | Northwest | Prince George | Southeast |
|---|---|---|---|---|---|---|
| G0 | 0.563 | 0.632 | 0.642 | 0.618 | 0.624 | 0.641 |
| G_lchw | 0.564 | 0.643 | 0.651 | 0.620 | 0.628 | 0.652 |

G_lchw is better than G0 in all six centres.

## By fire type: caught / flagged (2023)

| Type (fires) | Rule | G0 | **G_lchw** | ST32 |
|---|---|---|---|---|
| Already burning (36,839) | 36,839 / 56,746 | 35,524 / 52,714 | 35,702 / 52,450 | 35,757 / 53,267 |
| Next-door spread (12,188) | 0 | 621 / 2,175 | **1,543 / 4,987** (12.7%, 31% right) | 1,524 / 4,468 (12.5%, 34% right) |
| Recent fire nearby (6,548) | 0 | 49 / 126 | **216 / 793** | 39 / 100 |
| New, quiet area (5,332) | 0 | 0 | 0 | 0 |

## Next-door spread: AP on next-door cells (12,188 fires among 120,873 cell-days)

| Model | AP |
|---|---|
| General G0 | 0.204 |
| Fire shape (N_shape) | 0.267 |
| + land cover (N_lc) | 0.265 (−0.005 to +0.001) |
| + day-before afternoon station wind and weather (N_lchw) | **0.277** (+0.001 to +0.021 over N_lc) |
| Upper bound: target-day afternoon station weather (NU) | **0.322** (+0.028 to +0.062 over N_lchw) |

This is the first input that clearly matters for spread: **the afternoon wind and weather of the day the fire
spreads.** Yesterday's afternoon carries only a little of it. Using it operationally would need a forecast of
tomorrow afternoon's wind, RH and temperature, for example from Environment Canada's HRDPS/GEM models. That is worth
testing.

## New fires in quiet areas (5,332)

| Model | AP | Top 25 | Top 50 | Top 100 |
|---|---|---|---|---|
| G0 | 0.0041 | 5.2% | 7.5% | 11.1% |
| G_lc | 0.0032 | 4.6% | 6.7% | 9.8% |
| G_lchw | **0.0057** | 5.2% | **7.9%** | 11.1% |

Top 50 differences:

| Comparison | Interval |
|---|---|
| G_lc − G0 | −1.3 to −0.3 pts (worse) |
| G_lchw − G_lc | +0.6 to +2.0 pts |
| G_lchw − G0 | −0.4 to +1.4 pts (not significant) |

## Most important inputs (G_lchw, training data)

| Input | Importance |
|---|---|
| Burnable share | 17% |
| Afternoon max temperature | 17% |
| Afternoon min RH | 13% |
| Hourly FFMC | 9% |
| Days since lightning | 9% |
| Hourly ISI | 5% |
| Tree share | 4% |
| Gust | 3% |
| Wind u/v | 3% each |

Land cover only adds value alongside the weather. On its own it acts as a fixed per-cell fingerprint.

## Caveats
- **Land cover is from 2021.** WorldCover 2021 is one snapshot taken after the 2018 and 2021 fires, so recent burn
  scars show as shrub or grass.
- **Missing station weather.** 9% of cell-days have no station within 100 km.
- **No tuning.** One seed, default CatBoost settings.
- **No separate station-weather test.** There is no "station weather only" model, so the gain could partly need land
  cover. Exp 21's noon station weather had not helped in the older setup.
