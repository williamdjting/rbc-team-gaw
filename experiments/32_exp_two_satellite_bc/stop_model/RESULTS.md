# Stop model: can we predict which burning cells stop tomorrow? (2023, two-satellite label)

**Answer.**
- **The ranking improves a little.** Fire lifecycle and rain help sort burning cells into "keeps burning" and
  "stops". The AP of "keeps burning" rises from 0.823 to 0.841.
- **Total F1 does not improve.** Routed into the best general model it scores 0.630 vs 0.629 (−0.003 to +0.006).
- **Even the target day's observed rain and weather (upper bound)** only reaches 0.637 (−0.004 to +0.023).
- **About a third of "stops" aren't real stops.** 32% of stopped cells burn again within 7 days, so the fire was
  still alive but not detected (cloud, smoke, smouldering below detection, viewing angle). No input known the day
  before can predict a missed detection.

**Setup.**
- **Rows:** 2023 cells burning yesterday: 56,746 cell-days. 36,839 keep burning and **19,907 stop (35%)**.
- **Model:** a dedicated CatBoost trained on 57,941 burning-yesterday rows (2018–2021).
- **How it is used:** its decision replaces the general model's on these rows. Every other row keeps the general
  model with land cover + afternoon station weather (G_lchw). Thresholds per Fire Centre are chosen on 2022 to
  maximise total F1.

## Results on burning-yesterday cells, 2023

| Model | AP (keeps burning) | Stops dropped (of 19,907) | Still-burning wrongly dropped | Total F1 |
|---|---|---|---|---|
| Rule: keep all | — | 0 | 0 | 0.625 |
| General model G_lchw (best so far) | 0.823 | 3,159 (16%) | 1,137 | 0.629 |
| K0: dedicated, same inputs | 0.823 | 4,656 | 2,169 | 0.625 |
| K1: + fire lifecycle | 0.832 | 4,989 | 2,295 | 0.625 |
| K2: + rain (station precipitation, t−1 and 3-day) | **0.841** | 5,940 | 2,588 | 0.627 |
| K3: + BCWS incidents (active nearby, age, declared out) | 0.840 | 4,912 | 1,858 | **0.630** |
| KU: + target-day rain and weather (upper bound) | 0.886 | 7,838 (39%) | 2,639 | 0.637 |

The "Rule: keep all" row means flagging every burning-yesterday cell, with the general model used elsewhere.

**Fire lifecycle features:** days burning, FRP trend, fire size and growth, number of satellites seeing it, day/night
share, pixel size.

**Bootstrap (ISO weeks, 1,000 resamples), 95% interval**

| Comparison | Interval |
|---|---|
| AP: + lifecycle | **+0.005 to +0.016** |
| AP: + rain | **+0.003 to +0.015** |
| AP: + incidents | −0.003 to +0.001 |
| AP: target-day weather | **+0.030 to +0.069** |
| Total F1: K3 − G_lchw | −0.003 to +0.006 |
| Total F1: KU − G_lchw | −0.004 to +0.023 |

## Why F1 barely moves
- **Dropping a cell is a poor trade.** 65% of burning-yesterday cells keep burning, so dropping one only helps F1 when
  the model is quite sure it stops. Each extra stop dropped costs about 0.3–0.4 still-burning fires.
- **Many stops are detection gaps, not fires going out.**
  - Share of stopped cells with fire again within 2 / 3 / 5 / 7 days: **10% / 17% / 27% / 32%**.
  - The one-day gap fill already removes 1-day gaps.
  - These cells cannot be predicted from yesterday's inputs; they depend on whether the satellite sees the fire
    tomorrow.
- **The rest depends on the target day's weather.**
  - Rain matters most: precipitation is the top input at 12% importance.
  - Yesterday's rain helps the ranking, but the stopping rain usually falls on the day itself. That is why only the
    target-day upper bound moves F1.

## Most important inputs (K3)
- Precipitation yesterday: 12%
- Day share of detections: 9%
- Days since lightning: 9%
- FRP yesterday: 6%
- Neighbourhood active days: 6%
- Detections yesterday: 5%
- Age of nearest active BCWS incident: 5%

## Ideas this points to
- **Label:** fill gaps of 2–3 days (as for 1 day), or score with a 2-day tolerance. About 10–17% of today's "stops"
  would then count as still burning. That is a label decision, so check it with the mentors first.
- **Observation model:** tomorrow's VIIRS viewing angle and cloud cover would explain some missed detections.
- **Rain forecast:** a forecast for tomorrow. We decided not to pursue forecasts for now.
