"""What are the "quiet" fires? Describe every new-fire onset in BC, May-September 2018-2023.

An onset is a res-7 cell detected on day t with no detection in it or its ring-1 neighbours
on days t-7 .. t-1 (the "quiet positive" of Experiments 03-07 and 13-15 at a 1-day lead). Same-day
onset cells that touch are merged into one onset cluster. Each cluster is described by:
- BCWS match within 10 km (incident dates in Pacific time, FIRMS dates in UTC, so windows
  allow a day either way): a wildfire that started on t-3 .. t+1 (by cause); an older
  wildfire not yet declared out; another BCWS record type (nuisance, smoke chase,
  prescribed, ...); or nothing on record;
- distance to the nearest FIRMS detection in the previous 7 and 30 days;
- how many other onset clusters started the same day within 50 km (lightning busts);
- how many of the next 7 days have a detection in the cluster or its neighbours;
- first-day detections, FRP, night share, low-confidence share and FIRMS type.
Future days are used only to describe onsets, never as model inputs. 2024 is not loaded.
"""
from __future__ import annotations

import json
from pathlib import Path

import h3
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
DETECTIONS = EXPERIMENTS / "firms_summer_2018_2024/data/firms_h3_detections.csv"
BCWS = EXPERIMENTS / "13_exp_fire_context_bc/data/bcws_incidents_2018_2023.csv"
RESULTS = HERE / "results"
EARTH_KM = 6371.0
MATCH_KM, BUST_KM = 10.0, 50.0
ONE = pd.Timedelta(days=1)


def xyz(lat, lon):
    lat, lon = np.deg2rad(np.asarray(lat, float)), np.deg2rad(np.asarray(lon, float))
    return EARTH_KM * np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def load_detections():
    d = pd.read_csv(DETECTIONS, usecols=["latitude", "longitude", "acq_date", "frp", "daynight", "confidence", "type", "h3_cell"])
    d["date"] = pd.to_datetime(d.acq_date)
    d = d.loc[d.date.dt.year.between(2018, 2023)].reset_index(drop=True)
    assert d.date.dt.month.between(5, 9).all()
    return d


def load_bcws():
    b = pd.read_csv(BCWS)
    assert b.FIRE_YEAR.between(2018, 2023).all()
    for column in ("IGNITION_DATE", "FIRE_OUT_DATE"):
        b[column] = pd.to_datetime(b[column], utc=True).dt.tz_convert("America/Vancouver").dt.tz_localize(None).dt.normalize()
    b = b.loc[b.LATITUDE.notna() & b.IGNITION_DATE.notna() & (b.IGNITION_DATE >= "2017-01-01")].reset_index(drop=True)
    b.loc[b.FIRE_OUT_DATE < b.IGNITION_DATE, "FIRE_OUT_DATE"] = pd.NaT
    return b


def onsets(detections):
    """One row per onset cell-day."""
    by_day = {t: set(g) for t, g in detections.groupby("date").h3_cell}
    disk = {t: {n for c in cells for n in h3.grid_disk(c, 1)} for t, cells in by_day.items()}
    rows = []
    for t, cells in sorted(by_day.items()):
        if t.month == 5 and t.day < 8:
            continue  # a full 7-day history is needed
        recent = set().union(*[disk.get(t - k * ONE, set()) for k in range(1, 8)])
        rows += [{"date": t, "h3_cell": c} for c in cells - recent]
    return pd.DataFrame(rows), by_day, disk


def cluster(onset):
    """Merge touching onset cells on the same day."""
    onset = onset.sort_values(["date", "h3_cell"]).reset_index(drop=True)
    labels = np.zeros(len(onset), int)
    offset = 0
    for _, index in onset.groupby("date").groups.items():
        cells = onset.loc[index, "h3_cell"].tolist()
        position = {c: i for i, c in enumerate(cells)}
        pairs = [(i, position[n]) for i, c in enumerate(cells) for n in h3.grid_ring(c, 1) if n in position]
        rows, cols = (zip(*pairs) if pairs else ((), ()))
        graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(cells),) * 2)
        n, label = connected_components(graph, directed=False)
        labels[index] = label + offset
        offset += n
    onset["cluster"] = labels
    return onset


def stringify(value):
    """JSON keys must be strings (pandas gives tuples and intervals)."""
    if isinstance(value, dict):
        return {(" / ".join(map(str, k)) if isinstance(k, tuple) else str(k)): stringify(v) for k, v in value.items()}
    return value


def main():
    RESULTS.mkdir(exist_ok=True)
    detections = load_detections()
    bcws = load_bcws()
    onset, by_day, disk = onsets(detections)
    onset = cluster(onset)
    centre = np.array([h3.cell_to_latlng(c) for c in onset.h3_cell])
    onset["lat"], onset["lon"] = centre[:, 0], centre[:, 1]
    first_day = detections.assign(night=detections.daynight.eq("N"), low=detections.confidence.eq("l"),
                                  static=detections.type.ne(0)).groupby(["h3_cell", "date"]).agg(
        detections=("frp", "size"), frp_sum=("frp", "sum"), frp_max=("frp", "max"), night=("night", "mean"),
        low_confidence=("low", "mean"), non_vegetation=("static", "mean")).reset_index()
    onset = onset.merge(first_day, on=["h3_cell", "date"], how="left")

    clusters = onset.groupby("cluster").agg(date=("date", "first"), cells=("h3_cell", "size"), lat=("lat", "mean"),
                                            lon=("lon", "mean"), detections=("detections", "sum"), frp_sum=("frp_sum", "sum"),
                                            frp_max=("frp_max", "max"), night=("night", "mean"),
                                            low_confidence=("low_confidence", "mean"), non_vegetation=("non_vegetation", "mean"),
                                            members=("h3_cell", list)).reset_index()
    points = xyz(clusters.lat, clusters.lon)
    det_xyz = xyz(detections.latitude, detections.longitude)
    det_day = detections.date.to_numpy()
    bcws_xyz = xyz(bcws.LATITUDE, bcws.LONGITUDE)
    for name, days in (("nearest_detection_7d_km", 7), ("nearest_detection_30d_km", 30)):
        values = np.full(len(clusters), np.inf)
        for t, index in clusters.groupby("date").groups.items():
            prior = (det_day < np.datetime64(t)) & (det_day >= np.datetime64(t - days * ONE))
            if prior.any():
                values[index] = cKDTree(det_xyz[prior]).query(points[index], k=1)[0]
        clusters[name] = values
    same_day = np.zeros(len(clusters), int)
    for t, index in clusters.groupby("date").groups.items():
        tree = cKDTree(points[index])
        same_day[index] = tree.query_ball_point(points[index], r=BUST_KM, return_length=True) - 1
    clusters["other_onsets_same_day_50km"] = same_day
    clusters["active_days_next_7"] = [sum(any(n in disk.get(t + k * ONE, set()) for n in members) for k in range(1, 8))
                                      for t, members in zip(clusters.date, clusters.members)]

    # BCWS match within MATCH_KM, by priority.
    category, cause, start_offset = [], [], []
    tree = cKDTree(bcws_xyz)
    near = tree.query_ball_point(points, r=MATCH_KM)
    for t, candidates in zip(clusters.date, near):
        local = bcws.iloc[candidates]
        fire = local.loc[local.FIRE_TYPE.eq("Fire")]
        new = fire.loc[fire.IGNITION_DATE.between(t - 3 * ONE, t + ONE)]
        out = fire.FIRE_OUT_DATE.fillna(fire.IGNITION_DATE + pd.Timedelta(days=60))
        older = fire.loc[(fire.IGNITION_DATE < t - 3 * ONE) & (out >= t - ONE) & (fire.IGNITION_DATE >= t - pd.Timedelta(days=180))]
        other = local.loc[local.FIRE_TYPE.ne("Fire") & local.IGNITION_DATE.between(t - 3 * ONE, t + ONE)]
        if len(new):
            category.append("new BCWS wildfire")
            cause.append(new.FIRE_CAUSE.mode().iloc[0])
            start_offset.append(int((new.IGNITION_DATE.min() - t).days))
        elif len(older):
            category.append("older BCWS wildfire, not recently detected")
            cause.append(older.FIRE_CAUSE.mode().iloc[0])
            start_offset.append(int((older.IGNITION_DATE.max() - t).days))
        elif len(other):
            category.append("other BCWS record")
            cause.append(other.FIRE_TYPE.mode().iloc[0])
            start_offset.append(np.nan)
        else:
            category.append("no BCWS record")
            cause.append("")
            start_offset.append(np.nan)
    clusters["bcws_category"], clusters["bcws_cause_or_type"], clusters["bcws_start_minus_onset_days"] = category, cause, start_offset
    clusters["month"] = clusters.date.dt.month
    clusters["year"] = clusters.date.dt.year
    clusters.drop(columns="members").to_csv(RESULTS / "onset_clusters.csv", index=False)
    onset.drop(columns=["lat", "lon"]).to_csv(RESULTS / "onset_cells.csv", index=False)

    weighted = lambda column: clusters.groupby(column).agg(clusters=("cluster", "size"), cells=("cells", "sum"))
    summary = {
        "onset_cells": int(len(onset)), "onset_clusters": int(len(clusters)),
        "share_of_all_detected_cell_days": float(len(onset) / sum(len(v) for t, v in by_day.items()
                                                                 if not (t.month == 5 and t.day < 8))),
        "by_bcws_category": weighted("bcws_category").to_dict(),
        "by_bcws_cause_or_type": clusters.groupby(["bcws_category", "bcws_cause_or_type"]).cells.sum().to_dict(),
        "nearest_detection_7d_km": pd.cut(clusters.nearest_detection_7d_km, [0, 5, 10, 25, 50, 100, np.inf]).value_counts(sort=False).to_dict(),
        "nearest_detection_30d_km": pd.cut(clusters.nearest_detection_30d_km, [0, 5, 10, 25, 50, 100, np.inf]).value_counts(sort=False).to_dict(),
        "other_onsets_same_day_50km": pd.cut(clusters.other_onsets_same_day_50km, [-1, 0, 2, 5, 10, np.inf]).value_counts(sort=False).to_dict(),
        "active_days_next_7": clusters.active_days_next_7.value_counts().sort_index().to_dict(),
        "cluster_cells": pd.cut(clusters.cells, [0, 1, 2, 5, np.inf]).value_counts(sort=False).to_dict(),
        "month": weighted("month").to_dict(), "year": weighted("year").to_dict(),
        "first_day_median": clusters[["detections", "frp_sum", "frp_max", "night", "low_confidence"]].median().to_dict(),
        "non_vegetation_clusters": int((clusters.non_vegetation > 0).sum())}
    summary = stringify(summary)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
