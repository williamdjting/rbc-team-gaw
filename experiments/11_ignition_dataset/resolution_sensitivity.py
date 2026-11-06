"""Label-only counts at H3 resolutions 5, 6 and 7 (no dense tables or features).

Positives = unique (cell, date) of default-label events in the grid and target window.
Masked = footprint/onset cell-days that are not positive. Rows = cells x 146 target days.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import build_dataset as bd

HERE = Path(__file__).resolve().parent


def stats(res):
    grid = bd.load_grid(res)
    cells = set(grid["cell"])
    ev, fp, om, _ = bd.load_inputs(res)
    rows = []
    for year in bd.YEARS:
        start, end = pd.Timestamp(year, *bd.TARGET_START), pd.Timestamp(year, *bd.SEASON_END)
        days = (end - start).days + 1
        e = ev[ev["in_default_label"] & ev["date"].between(start, end) & ev["cell"].isin(cells)]
        pos = set(zip(e["cell"], e["date"]))
        f = fp[fp["cell"].isin(cells) & (fp["end_date"] >= start) & (fp["start_date"] <= end)]
        masked = set()
        for c, s, t in zip(f["cell"], f["start_date"].clip(lower=start), f["end_date"].clip(upper=end)):
            masked.update((c, d) for d in pd.date_range(s, t))
        o = om[om["cell"].isin(cells) & om["date"].between(start, end)]
        masked.update(zip(o["cell"], o["date"]))
        masked -= pos
        n = len(cells) * days
        rows.append(dict(resolution=res, year=year, cells=len(cells), rows=n, events=len(e),
                         positive_cell_days=len(pos), events_sharing_cell_day=len(e) - len(pos),
                         masked=len(masked), prevalence=len(pos) / (n - len(masked)),
                         masked_share=len(masked) / n))
    return pd.DataFrame(rows)


def main():
    t = pd.concat([stats(r) for r in (5, 6, 7)], ignore_index=True)
    t.to_csv(HERE / "results/resolution_sensitivity.csv", index=False)
    pd.set_option("display.width", 200)
    print(t)
    print(t.groupby("resolution")[["cells", "rows", "positive_cell_days", "events_sharing_cell_day", "masked"]].sum()
          .assign(prevalence=lambda x: x["positive_cell_days"] / (x["rows"] - x["masked"])))


if __name__ == "__main__":
    (HERE / "results").mkdir(exist_ok=True)
    main()
