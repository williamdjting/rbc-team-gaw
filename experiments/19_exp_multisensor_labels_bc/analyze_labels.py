"""Can more satellites and gap filling make the next-day detection label more consistent?

The current label (Experiments 02-07 and 13-18) is a FIRMS VIIRS S-NPP detection in an H3 res-7 cell on
a UTC day. One satellite misses burning cells on many days (cloud, smoke, overpass geometry),
which caps any forecast. This builds candidate labels from the CWFIS hotspot archive
(scripts/download_cwfis_hotspots.py) for the six Fire Centre grids:
- snpp_firms: the current label;
- snpp_cwfis: CWFIS's copy of VIIRS S-NPP (I-band), as an alignment check;
- viirs_pair: S-NPP (FIRMS) or NOAA-20 VIIRS (I-band);
- viirs_modis: viirs_pair or MODIS (Terra, Aqua);
- cwfis_viirs: CWFIS S-NPP or NOAA-20 (CWFIS drops some non-wildfire detections that FIRMS keeps);
- *_filled: plus one-day gap filling, a cell-day is positive if the cell is positive the day
  before and the day after (uses the future, so labels only). The rule and the quiet test use the
  unfilled label, which is all a forecast can see on the day.
AVHRR (2018-2020 only) and VIIRS M-band are left out so the sensor mix is stable.
For each label, year and Fire Centre (target days May 8 - September 30, as in Experiment 14):
positives; persistence P(positive | positive yesterday); the gap share (a cell positive within
3 days before and after, but not that day); the share of positives in quiet cells (nothing in the
cell or its neighbours for 7 days); the previous-day rule's F1 on the exact cell and with the
one-cell margin; and the share of positives inside (or within 2 km of) a final BCWS perimeter of
that year. NOAA-20 is complete from 2021, partial in 2019 (from July) and 2020; it is absent in 2018.
2024 is never loaded.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import scipy.sparse as sp
import shapely

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
ROOT = EXPERIMENTS.parent
RESULTS = HERE / "results"
GRIDS = EXPERIMENTS / "02_hotspot_persistence_bc/regions"
FIRMS = EXPERIMENTS / "firms_summer_2018_2024/data/firms_h3_detections.csv"
CWFIS = ROOT / "datasets/cwfis_hotspots/bc_hotspots_2018_2023.parquet"
PERIMETERS = EXPERIMENTS / "17_exp_history_nonwildfire_bc/data/bcws_perimeters_2018_2023.geojson"
YEARS = range(2018, 2024)
SEASON_DAYS, FIRST_TARGET_DAY = 153, 7  # May 1 - September 30; targets from May 8
GAP_DAYS, QUIET_DAYS, PERIMETER_MARGIN_KM = 3, 7, 2.0


def season_day(dates):
    return (dates - pd.to_datetime(dates.dt.year.astype(str) + "-05-01")).dt.days.to_numpy()


def load_sources():
    """(source, year) -> DataFrame of unique h3_cell, day."""
    firms = pd.read_csv(FIRMS, usecols=["acq_date", "h3_cell"], parse_dates=["acq_date"])
    firms = firms.loc[firms.acq_date.dt.year.between(2018, 2023)]
    cwfis = pd.read_parquet(CWFIS, columns=["rep_date", "satellite", "sensor", "h3_cell"])
    assert cwfis.rep_date.dt.year.between(2018, 2023).all() and firms.acq_date.dt.year.between(2018, 2023).all()
    cwfis["acq_date"] = cwfis.rep_date.dt.normalize()
    pick = {"snpp_firms": firms,
            "snpp_cwfis": cwfis.loc[cwfis.sensor.eq("VIIRS-I") & cwfis.satellite.eq("S-NPP")],
            "noaa20": cwfis.loc[cwfis.sensor.eq("VIIRS-I") & cwfis.satellite.eq("NOAA-20")],
            "modis": cwfis.loc[cwfis.sensor.eq("MODIS")]}
    out = {}
    for name, frame in pick.items():
        frame = frame.assign(day=season_day(frame.acq_date), year=frame.acq_date.dt.year)
        frame = frame.loc[(frame.day >= 0) & (frame.day < SEASON_DAYS)].drop_duplicates(["h3_cell", "day"])
        for year, part in frame.groupby("year"):
            out[name, int(year)] = part[["h3_cell", "day"]]
    return out


def neighbourhood(cells):
    position = {c: i for i, c in enumerate(cells)}
    rows, cols = [], []
    for i, cell in enumerate(cells):
        for other in h3.grid_disk(cell, 1):
            j = position.get(other)
            if j is not None:
                rows.append(i)
                cols.append(j)
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(cells),) * 2)


def local_km(lon, lat):
    return np.column_stack([np.asarray(lon) * 111.32 * np.cos(np.deg2rad(54.0)), np.asarray(lat) * 110.57])


def perimeter_cells(cells):
    """year -> boolean per cell: centre inside, or within 2 km of, a final BCWS perimeter of that year."""
    data = json.loads(PERIMETERS.read_text())
    years = np.array([f["properties"]["FIRE_YEAR"] for f in data["features"]])
    shapes = shapely.from_geojson([json.dumps(f["geometry"]) for f in data["features"]])
    shapes = shapely.transform(shapes, lambda xy: local_km(xy[:, 0], xy[:, 1]))
    lat_lon = np.array([h3.cell_to_latlng(c) for c in cells])
    points = shapely.points(local_km(lat_lon[:, 1], lat_lon[:, 0]))
    out = {}
    for year in YEARS:
        tree = shapely.STRtree(shapes[years == year])
        hits = tree.query(points, predicate="dwithin", distance=PERIMETER_MARGIN_KM)
        flag = np.zeros(len(cells), bool)
        flag[hits[0]] = True
        out[year] = flag
    return out


def grid(source, cells):
    a = np.zeros((len(cells), SEASON_DAYS), bool)
    if source is not None:
        index = cells.get_indexer(source.h3_cell)
        keep = index >= 0
        a[index[keep], source.day.to_numpy()[keep]] = True
    return a


def shift(a, days):
    """a[:, d - days] (False outside the season)."""
    out = np.zeros_like(a)
    if days == 0:
        return a.copy()
    if days > 0:
        out[:, days:] = a[:, :-days]
    else:
        out[:, :days] = a[:, -days:]
    return out


def any_within(a, days, direction):
    out = np.zeros_like(a)
    for k in range(1, days + 1):
        out |= shift(a, k * direction)
    return out


def f1(flagged, correct, positives, caught):
    precision, recall = correct / max(flagged, 1), caught / max(positives, 1)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def metrics(label, observed, disk1, inside, reference):
    """Counts for one label grid (cells x days); target days from FIRST_TARGET_DAY. The rule and
    the quiet test use `observed` (the label without gap filling), known on the day."""
    t = slice(FIRST_TARGET_DAY, SEASON_DAYS)
    truth, rule = label[:, t], shift(observed, 1)[:, t]
    near = (disk1 @ label.astype(np.float32))[:, t] > 0
    rule_near = (disk1 @ rule.astype(np.float32)) > 0
    before, after = any_within(label, GAP_DAYS, 1)[:, t], any_within(label, GAP_DAYS, -1)[:, t]
    recent = np.zeros_like(label)
    for k in range(1, QUIET_DAYS + 1):
        recent |= shift(observed, k)
    recent_disk = (disk1 @ recent.astype(np.float32))[:, t] > 0
    return {"positives": int(truth.sum()), "flagged": int(rule.sum()),
            "persist_num": int((truth & rule).sum()), "persist_den": int(rule.sum()),
            "gap_num": int((before & after & ~truth).sum()), "gap_den": int((before & after).sum()),
            "quiet_positives": int((truth & ~recent_disk).sum()),
            "exact_correct": int((rule & truth).sum()), "exact_caught": int((truth & rule).sum()),
            "margin_correct": int((rule & near).sum()), "margin_caught": int((truth & rule_near).sum()),
            "inside_perimeter": int((truth & inside[:, None]).sum()),
            "contains_reference": int((truth & reference[:, t]).sum()), "reference_positives": int(reference[:, t].sum())}


def finish(row):
    row["persistence"] = row["persist_num"] / max(row["persist_den"], 1)
    row["gap_share"] = row["gap_num"] / max(row["gap_den"], 1)
    row["quiet_share"] = row["quiet_positives"] / max(row["positives"], 1)
    row["rule_f1_exact"] = f1(row["flagged"], row["exact_correct"], row["positives"], row["exact_caught"])
    row["rule_f1_margin"] = f1(row["flagged"], row["margin_correct"], row["positives"], row["margin_caught"])
    row["inside_perimeter_share"] = row["inside_perimeter"] / max(row["positives"], 1)
    row["firms_snpp_recovered"] = row["contains_reference"] / max(row["reference_positives"], 1)
    row["positives_vs_firms"] = row["positives"] / max(row["reference_positives"], 1)
    return row


def main():
    tic = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    sources = load_sources()
    rows, alignment = [], []
    for folder in sorted(p for p in GRIDS.iterdir() if (p / "data/h3_grid.parquet").exists()):
        region = folder.name.replace("_", " ").title()
        cells = pd.Index(pd.read_parquet(folder / "data/h3_grid.parquet", columns=["h3_cell"]).h3_cell)
        disk1 = neighbourhood(list(cells))
        inside = perimeter_cells(cells)
        for year in YEARS:
            g = {name: grid(sources.get((name, year)), cells) for name in ("snpp_firms", "snpp_cwfis", "noaa20", "modis")}
            for lag in (-1, 0, 1):  # CWFIS dates should line up with FIRMS at lag 0
                a, b = g["snpp_firms"], shift(g["snpp_cwfis"], lag)
                alignment.append({"region": region, "year": year, "lag_days": lag, "both": int((a & b).sum()),
                                  "either": int((a | b).sum())})
            labels = {"snpp_firms": g["snpp_firms"], "snpp_cwfis": g["snpp_cwfis"],
                      "viirs_pair": g["snpp_firms"] | g["noaa20"],
                      "viirs_modis": g["snpp_firms"] | g["noaa20"] | g["modis"],
                      "cwfis_viirs": g["snpp_cwfis"] | g["noaa20"]}
            observed = dict(labels)
            for name in ("snpp_firms", "viirs_pair", "viirs_modis", "cwfis_viirs"):
                labels[f"{name}_filled"] = labels[name] | (shift(labels[name], 1) & shift(labels[name], -1))
                observed[f"{name}_filled"] = labels[name]
            for name, label in labels.items():
                rows.append({"region": region, "year": year, "label": name,
                             **metrics(label, observed[name], disk1, inside[year], g["snpp_firms"])})
        print(f"{region}: done ({time.time()-tic:.0f}s)", flush=True)

    table = pd.DataFrame(rows)
    counts = [c for c in table.columns if c not in ("region", "year", "label")]
    bc = table.groupby(["year", "label"])[counts].sum().reset_index().assign(region="BC-wide")
    table = pd.concat([table, bc], ignore_index=True)
    table = pd.DataFrame([finish(r) for r in table.to_dict("records")])
    table.to_csv(RESULTS / "label_metrics.csv", index=False)
    align = pd.DataFrame(alignment).groupby(["year", "lag_days"])[["both", "either"]].sum().reset_index()
    align["jaccard"] = align.both / align.either
    align.to_csv(RESULTS / "cwfis_firms_alignment.csv", index=False)
    (RESULTS / "manifest.json").write_text(json.dumps({
        "labels": sorted(table.label.unique()), "years": list(YEARS), "target_days": "May 8 - September 30",
        "gap_days": GAP_DAYS, "quiet_days": QUIET_DAYS, "perimeter_margin_km": PERIMETER_MARGIN_KM,
        "noaa20_coverage": "complete 2021-2023; partial 2019 (from July) and 2020; none in 2018",
        "holdout_loaded": False, "runtime_seconds": round(time.time() - tic)}, indent=2), encoding="utf-8")
    pd.set_option("display.width", 220)
    show = ["positives_vs_firms", "persistence", "gap_share", "quiet_share", "rule_f1_exact", "rule_f1_margin",
            "inside_perimeter_share", "firms_snpp_recovered"]
    print(align.pivot(index="year", columns="lag_days", values="jaccard").round(3).to_string())
    print(table.loc[table.region.eq("BC-wide")].set_index(["year", "label"])[show].round(3).to_string())


if __name__ == "__main__":
    main()
