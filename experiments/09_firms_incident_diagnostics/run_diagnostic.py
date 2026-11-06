"""Matched-scope spatial concentration, temporal clusters and incident proximity audit."""
from __future__ import annotations

import base64
import hashlib
from html import escape
import json
from pathlib import Path

import h3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm, LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely import intersects_xy, make_valid
from shapely.geometry import shape
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results"
FIG = OUT / "figures"
FIRMS = ROOT / "data/firms_with_cems_18_24.csv"
INCIDENTS = ROOT / "data/bc_historical_incidents_2018_2023.csv"
BOUNDARY = ROOT / "datasets/bc_legal_boundary.geojson"
CENTRES = ROOT / "data/region_exploration/fire_centres.geojson"
RADIUS_KM = 6371.0088
CMP = LinearSegmentedColormap.from_list("visible_oranges", plt.get_cmap("Oranges")(np.linspace(.35, 1, 256)))


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def sphere(lat, lon):
    lat, lon = np.deg2rad(np.asarray(lat)), np.deg2rad(np.asarray(lon))
    return np.column_stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])


def chord_km(chord):
    return 2 * RADIUS_KM * np.arcsin(np.clip(np.asarray(chord) / 2, 0, 1))


def concentration(counts):
    values = np.sort(np.asarray(counts, dtype=float))[::-1]
    if not len(values):
        return dict(occupied_cells=0, weight=0, top_1pct_share=np.nan, top_10pct_share=np.nan,
                    effective_cells=np.nan, effective_fraction=np.nan)
    shares = values / values.sum()
    return dict(occupied_cells=len(values), weight=int(values.sum()),
                top_1pct_share=shares[:max(1, int(np.ceil(.01*len(values))))].sum(),
                top_10pct_share=shares[:max(1, int(np.ceil(.1*len(values))))].sum(),
                effective_cells=1 / (shares**2).sum(),
                effective_fraction=1 / ((shares**2).sum()*len(values)))


def connected_labels(nodes, gap):
    """Retrospective components: same/adjacent H3 cells, up to gap calendar days.

    Transitive connections can span months and merge separate fires. These IDs
    must not be interpreted as official incidents or used as forecast features.
    """
    parent = np.arange(len(nodes))
    rank = np.zeros(len(nodes), dtype=int)
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        a, b = find(i), find(j)
        if a == b:
            return
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1
    seen, neighbours = {}, {}
    ordered = nodes.reset_index(drop=True).sort_values(["day", "h3_cell"])
    for i, day, cell in ordered[["day", "h3_cell"]].itertuples():
        if cell not in neighbours:
            neighbours[cell] = h3.grid_disk(cell, 1)
        for delta in range(gap + 1):
            for near in neighbours[cell]:
                j = seen.get((day - delta, near))
                if j is not None:
                    union(i, j)
        seen[(day, cell)] = i
    return np.unique([find(i) for i in range(len(parent))], return_inverse=True)[1]


def load():
    bc_json = json.loads(BOUNDARY.read_text())
    bc = make_valid(unary_union([shape(f["geometry"]) for f in bc_json["features"]]))
    centres = json.loads(CENTRES.read_text())
    kamloops = make_valid(shape(next(f["geometry"] for f in centres["features"]
                                     if f["properties"]["MOF_FIRE_CENTRE_NAME"] == "Kamloops Fire Centre")))
    columns = ["latitude", "longitude", "acq_date", "h3_cell", "type", "confidence", "satellite", "version", "frp"]
    pieces = []
    for part in pd.read_csv(FIRMS, usecols=columns, chunksize=150_000):
        part["date"] = pd.to_datetime(part.acq_date, errors="raise")
        # 2024 records are discarded before any diagnostic calculation.
        pieces.append(part.loc[part.date.dt.year.between(2018, 2023) & part.date.dt.month.between(5, 9)])
    f = pd.concat(pieces, ignore_index=True)
    f["year"] = f.date.dt.year
    f["day"] = (f.date - pd.Timestamp("1970-01-01")).dt.days
    valid = f.latitude.between(-90, 90) & f.longitude.between(-180, 180)
    assert valid.all(), "Invalid FIRMS coordinates"
    f = f.loc[intersects_xy(bc, f.longitude.to_numpy(), f.latitude.to_numpy())].copy()
    f["recomputed_h3"] = [h3.latlng_to_cell(lat, lon, 7) for lat, lon in zip(f.latitude, f.longitude)]
    assert f.h3_cell.eq(f.recomputed_h3).all(), "FIRMS H3 assignments disagree with coordinates"
    f.drop(columns="recomputed_h3", inplace=True)
    source_counts = f.groupby(["year", "type", "confidence", "satellite", "version"], dropna=False).size().rename("detections").reset_index()
    source_counts.to_csv(OUT / "firms_source_composition.csv", index=False)
    sensitivity = []
    for name, subset in [("all_types", f), ("vegetation_type0", f.loc[f.type.eq(0)]),
                         ("vegetation_nominal_high", f.loc[f.type.eq(0) & f.confidence.str.lower().isin(["n", "h"])])]:
        for year, g in subset.groupby("year"):
            sensitivity.append(dict(filter=name, year=year, detections=len(g), cells=g.h3_cell.nunique(),
                                    cell_days=len(g.drop_duplicates(["h3_cell", "date"]))))
    pd.DataFrame(sensitivity).to_csv(OUT / "firms_filter_sensitivity.csv", index=False)
    f = f.loc[f.type.eq(0)].reset_index(drop=True)
    b = pd.read_csv(INCIDENTS)
    assert b.OBJECTID.is_unique and b.FIRE_YEAR.between(2018, 2023).all()
    annual = pd.read_csv(HERE / "reference_annual_totals.csv").set_index("year")
    annual["local_all_record_types"] = b.groupby("FIRE_YEAR").size()
    annual["local_fire_records"] = b.loc[b.FIRE_TYPE.eq("Fire")].groupby("FIRE_YEAR").size()
    annual.reset_index().to_csv(OUT / "annual_source_counts.csv", index=False)
    b["date"] = pd.to_datetime(b.IGNITION_DATE, unit="ms", utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    b["date_status"] = np.select([b.IGNITION_DATE.isna(), b.date.isna(), b.date.dt.year.ne(b.FIRE_YEAR)],
                                   ["missing", "invalid", "year_mismatch"], default="usable")
    b["coord_valid"] = b.point_latitude.between(-90, 90) & b.point_longitude.between(-180, 180)
    b["inside_bc"] = False
    b.loc[b.coord_valid, "inside_bc"] = intersects_xy(bc, b.loc[b.coord_valid, "point_longitude"].to_numpy(), b.loc[b.coord_valid, "point_latitude"].to_numpy())
    b.groupby(["FIRE_YEAR", "FIRE_TYPE", "date_status", "inside_bc"], dropna=False).size().rename("records").reset_index().to_csv(OUT / "incident_quality.csv", index=False)
    b["event_key"] = b.FIRE_YEAR.astype(str) + ":" + b.FIRE_NUMBER.astype(str)
    fire_records = b.loc[b.FIRE_TYPE.eq("Fire") & b.inside_bc].copy()
    assert not fire_records.event_key.duplicated().any()
    b = fire_records.loc[fire_records.date_status.eq("usable") & fire_records.date.dt.month.between(5, 9)].copy()
    b["year"] = b.date.dt.year
    b["day"] = (b.date - pd.Timestamp("1970-01-01")).dt.days
    b["h3_cell"] = [h3.latlng_to_cell(lat, lon, 7) for lat, lon in zip(b.point_latitude, b.point_longitude)]
    b["support_eligible"] = ((b.date >= pd.to_datetime(b.year.astype(str) + "-05-04")) &
                             (b.date + pd.Timedelta(days=30) <= pd.to_datetime(b.year.astype(str) + "-09-30")))
    quality = pd.DataFrame([
        ["BC incident source rows", len(pd.read_csv(INCIDENTS))],
        ["BC Fire records inside boundary", len(fire_records)],
        ["BC Fire records with unusable/year-mismatched date", int(fire_records.date_status.ne("usable").sum())],
        ["BC usable May-Sep Fire records", len(b)],
        ["BC records eligible for every support window", int(b.support_eligible.sum())],
        ["FIRMS vegetation detections May-Sep 2018-2023", len(f)],
    ], columns=["stage", "rows"])
    quality.to_csv(OUT / "scope.csv", index=False)
    b.to_csv(OUT / "incidents_in_scope.csv", index=False)
    return f, b, bc, kamloops


def spatial_stats(f, b, kamloops):
    rows, scale, samples = [], [], []
    nodes = f.groupby(["year", "date", "day", "h3_cell"]).size().rename("detections").reset_index()
    for year in range(2018, 2024):
        fy, by, ny = f.loc[f.year.eq(year)], b.loc[b.year.eq(year)], nodes.loc[nodes.year.eq(year)]
        for source, counts in [("FIRMS detections", fy.groupby("h3_cell").size()),
                               ("FIRMS cell-days", ny.groupby("h3_cell").size()),
                               ("BC incidents", by.groupby("h3_cell").size())]:
            rows.append(dict(year=year, source=source, **concentration(counts)))
        for resolution in [5, 6, 7]:
            fc = fy.h3_cell.map(lambda x: h3.cell_to_parent(x, resolution))
            bcells = by.h3_cell.map(lambda x: h3.cell_to_parent(x, resolution))
            fs, bs = set(fc), set(bcells)
            scale.append(dict(year=year, resolution=resolution, firms_cells=len(fs), incident_cells=len(bs),
                              shared_cells=len(fs & bs), incident_cell_overlap=len(fs & bs)/len(bs) if bs else np.nan,
                              jaccard=len(fs & bs)/len(fs | bs) if fs | bs else np.nan))
    pd.DataFrame(rows).to_csv(OUT / "spatial_concentration.csv", index=False)
    pd.DataFrame(scale).to_csv(OUT / "spatial_overlap_by_resolution.csv", index=False)
    # Match sample size to avoid explaining concentration just by different counts.
    for scope in ["BC", "Kamloops"]:
        fsub, bsub = f.loc[f.year.eq(2023)], b.loc[b.year.eq(2023)]
        if scope == "Kamloops":
            fsub = fsub.loc[intersects_xy(kamloops, fsub.longitude.to_numpy(), fsub.latitude.to_numpy())]
            bsub = bsub.loc[intersects_xy(kamloops, bsub.point_longitude.to_numpy(), bsub.point_latitude.to_numpy())]
        cells = {"FIRMS occupied cells": sorted(fsub.h3_cell.unique()), "BC incident cells": sorted(bsub.h3_cell.unique())}
        n = min(1000, *(len(c) for c in cells.values()))
        if n < 2:
            continue
        for source, cell_ids in cells.items():
            coords = np.array([h3.cell_to_latlng(c) for c in cell_ids])
            xyz = sphere(coords[:, 0], coords[:, 1])
            for seed in range(50):
                selected = xyz[np.random.default_rng(seed).choice(len(xyz), n, replace=False)]
                distances = chord_km(cKDTree(selected).query(selected, k=2)[0][:, 1])
                samples.append(dict(scope=scope, source=source, seed=seed, cells_sampled=n,
                                    mean_nearest_km=distances.mean(), median_nearest_km=np.median(distances)))
    pd.DataFrame(samples).to_csv(OUT / "equal_size_spatial_samples.csv", index=False)
    return nodes


def incident_support(f, b):
    rows = []
    windows = [(0, 7), (0, 14), (0, 30), (-3, 14)]
    for year in range(2018, 2024):
        fy = f.loc[f.year.eq(year)].reset_index(drop=True)
        incidents = b.loc[b.year.eq(year) & b.support_eligible]
        xyz = sphere(fy.latitude, fy.longitude)
        tree = cKDTree(xyz)
        day = fy.day.to_numpy()
        print(f"Nearby detections: {year}, {len(incidents):,} eligible incidents", flush=True)
        for incident in incidents.itertuples():
            location = sphere([incident.point_latitude], [incident.point_longitude])[0]
            nearby = np.asarray(tree.query_ball_point(location, 2*np.sin(10/(2*RADIUS_KM))), dtype=int)
            distances = chord_km(np.linalg.norm(xyz[nearby] - location, axis=1))
            lags = day[nearby] - incident.day
            for radius in [1, 3, 5, 10]:
                for start, end in windows:
                    chosen = (distances <= radius + 1e-9) & (lags >= start) & (lags <= end)
                    index = nearby[chosen]
                    rows.append(dict(event_key=incident.event_key, year=year, date=incident.date,
                                      h3_cell=incident.h3_cell, latitude=incident.point_latitude,
                                      longitude=incident.point_longitude, recorded_size=incident.CURRENT_SIZE,
                                      radius_km=radius, start_day=start, end_day=end,
                                      supported=bool(len(index)), detections=len(index),
                                      observed_days=len(np.unique(day[index])),
                                      first_lag=int(lags[chosen].min()) if len(index) else np.nan,
                                      nearest_km=float(distances[chosen].min()) if len(index) else np.nan))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "incident_proximity_detail.csv", index=False)
    result.groupby(["radius_km", "start_day", "end_day"]).supported.agg(["size", "sum", "mean"]).reset_index().to_csv(OUT / "support_sensitivity.csv", index=False)
    primary = result.loc[result.radius_km.eq(5) & result.start_day.eq(0) & result.end_day.eq(14)].copy()
    primary["size_bin"] = pd.cut(primary.recorded_size.where(primary.recorded_size.gt(0)),
        [0, 1, 10, 100, 1000, np.inf], labels=["0-1", "1-10", "10-100", "100-1000", ">1000"]).astype("object").fillna("Unknown/zero")
    primary.groupby("size_bin", observed=True).supported.agg(["size", "sum", "mean"]).reset_index().to_csv(OUT / "support_by_recorded_size.csv", index=False)
    primary.groupby("year").supported.agg(["size", "sum", "mean"]).reset_index().to_csv(OUT / "support_by_year.csv", index=False)
    return primary


def clusters(nodes):
    year_nodes = nodes.loc[nodes.year.eq(2023)].reset_index(drop=True)
    rows = []
    for gap in [1, 3, 7]:
        print(f"Retrospective H3 components: gap <= {gap} days", flush=True)
        labels = connected_labels(year_nodes, gap)
        temp = year_nodes.assign(cluster=labels)
        counts = temp.groupby("cluster").agg(cell_days=("day", "size"), detections=("detections", "sum"),
            cells=("h3_cell", "nunique"), first_day=("date", "min"), last_day=("date", "max"),
            observed_days=("date", "nunique")).sort_values("detections", ascending=False)
        counts["span_days"] = (counts.last_day - counts.first_day).dt.days + 1
        counts.to_csv(OUT / f"clusters_gap{gap}_2023.csv")
        rows.append(dict(gap_days=gap, components=len(counts), singleton_components=int(counts.cell_days.eq(1).sum()),
                         top10_detection_share=counts.detections.head(10).sum()/counts.detections.sum(),
                         largest_detection_share=counts.detections.max()/counts.detections.sum(),
                         median_span_days=counts.span_days.median(), max_span_days=counts.span_days.max()))
        if gap == 3:
            temp.to_csv(OUT / "cluster_nodes_gap3_2023.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "cluster_sensitivity.csv", index=False)


def polygons(geometry):
    if geometry.geom_type == "Polygon":
        yield geometry
    else:
        for g in getattr(geometry, "geoms", []):
            yield from polygons(g)


def draw_cells(ax, geometry, counts, title, binary=False):
    for poly in polygons(geometry):
        ax.plot(*poly.exterior.xy, color="#8b969b", linewidth=.45)
    if len(counts):
        vertices = [[(lon, lat) for lat, lon in h3.cell_to_boundary(cell)] for cell in counts.index]
        if binary:
            ax.add_collection(PolyCollection(vertices, facecolors="#d55e00", edgecolors="none"))
        else:
            norm = LogNorm(vmin=1, vmax=max(2, counts.max()))
            collection = PolyCollection(vertices, array=counts.to_numpy(float), cmap=CMP, norm=norm, edgecolors="none")
            ax.add_collection(collection)
            plt.colorbar(collection, ax=ax, shrink=.7, label="Detections / cell (log)")
    x0, y0, x1, y1 = geometry.bounds
    ax.set(xlim=(x0-.05, x1+.05), ylim=(y0-.05, y1+.05), title=title)
    ax.set_aspect(1/np.cos(np.deg2rad(geometry.centroid.y)))
    ax.tick_params(labelsize=7)


def save(fig, filename):
    fig.savefig(FIG / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def charts(f, b, nodes, bc, kamloops, support):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(16, 12), layout="constrained")
    for row, (name, geometry) in enumerate([("BC", bc), ("Kamloops", kamloops)]):
        fy, by = f.loc[f.year.eq(2023)], b.loc[b.year.eq(2023)]
        fy = fy.loc[intersects_xy(geometry, fy.longitude.to_numpy(), fy.latitude.to_numpy())]
        by = by.loc[intersects_xy(geometry, by.point_longitude.to_numpy(), by.point_latitude.to_numpy())]
        fc, ic = fy.groupby("h3_cell").size(), by.groupby("h3_cell").size()
        draw_cells(axes[row, 0], geometry, fc, f"{name}: FIRMS detections\n{len(fy):,} observations", False)
        draw_cells(axes[row, 1], geometry, fc, f"{name}: FIRMS occupied cells\n{len(fc):,} cells, repetition removed", True)
        draw_cells(axes[row, 2], geometry, ic, f"{name}: BC incident cells\n{len(by):,} incidents / {len(ic):,} cells", True)
    fig.suptitle("Same scope: May-September 2023, H3 resolution 7\nOrange presence maps use the same cell size and map extent", fontsize=15)
    save(fig, "01_matched_spatial_maps.png")

    fig, axes = plt.subplots(2, 4, figsize=(17, 9), layout="constrained")
    fy = f.loc[f.year.eq(2023) & intersects_xy(kamloops, f.longitude.to_numpy(), f.latitude.to_numpy())]
    by = b.loc[b.year.eq(2023) & intersects_xy(kamloops, b.point_longitude.to_numpy(), b.point_latitude.to_numpy())]
    for col, month in enumerate([6, 7, 8, 9]):
        fc = fy.loc[fy.date.dt.month.eq(month)].groupby("h3_cell").size()
        ic = by.loc[by.date.dt.month.eq(month)].groupby("h3_cell").size()
        draw_cells(axes[0, col], kamloops, fc, f"FIRMS month {month}\n{len(fc):,} occupied cells", True)
        draw_cells(axes[1, col], kamloops, ic, f"BC month {month}\n{int(ic.sum()):,} incident starts", True)
    fig.suptitle("Kamloops 2023: ongoing observed activity versus recorded starts\nBoth rows show cell presence, not detection intensity", fontsize=14)
    save(fig, "02_monthly_comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    sources = [("FIRMS detections", f.loc[f.year.eq(2023)].groupby("h3_cell").size()),
               ("FIRMS cell-days", nodes.loc[nodes.year.eq(2023)].groupby("h3_cell").size()),
               ("BC incidents", b.loc[b.year.eq(2023)].groupby("h3_cell").size())]
    for name, counts in sources:
        values = np.sort(counts.to_numpy())[::-1]
        axes[0].plot(np.r_[0, np.arange(1, len(values)+1)/len(values)]*100,
                     np.r_[0, values.cumsum()/values.sum()]*100, label=name)
    axes[0].plot([0, 100], [0, 100], "--", color="gray", label="Equal weight per occupied cell")
    axes[0].set(xlabel="Top fraction of each source's occupied cells (%)", ylabel="Cumulative source weight (%)", title="2023 spatial concentration")
    axes[0].legend(fontsize=8)
    samples = pd.read_csv(OUT / "equal_size_spatial_samples.csv")
    names, values = [], []
    for (scope, source), g in samples.groupby(["scope", "source"], sort=False):
        names.append(f"{scope}\n{source.split()[0]}\nn={int(g.cells_sampled.iloc[0])}")
        values.append(g.mean_nearest_km)
    axes[1].boxplot(values, tick_labels=names)
    axes[1].set(ylabel="Mean nearest occupied-cell distance (km)", title="Equal-size samples, 50 repeats")
    save(fig, "03_concentration.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    sensitivity = pd.read_csv(OUT / "support_sensitivity.csv")
    sensitivity["window"] = sensitivity.start_day.astype(str) + " to " + sensitivity.end_day.astype(str)
    pivot = sensitivity.pivot(index="radius_km", columns="window", values="mean")
    axes[0].imshow(pivot, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    axes[0].set(xticks=range(len(pivot.columns)), xticklabels=pivot.columns,
                yticks=range(len(pivot)), yticklabels=pivot.index, xlabel="Days relative to recorded start (inclusive)",
                ylabel="Search radius (km)", title="Incidents with nearby FIRMS detections")
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            axes[0].text(j, i, f"{value:.1%}", ha="center", va="center", color="white" if value>.55 else "black")
    sizes = pd.read_csv(OUT / "support_by_recorded_size.csv").set_index("size_bin").reindex(["0-1", "1-10", "10-100", "100-1000", ">1000", "Unknown/zero"]).dropna()
    axes[1].bar(sizes.index, sizes["mean"]*100, color="#168b82")
    for i, (_, r) in enumerate(sizes.iterrows()):
        axes[1].text(i, r["mean"]*100+1, f"n={int(r['size'])}", ha="center", fontsize=8)
    axes[1].set(ylim=(0, 110), xlabel="Recorded CURRENT_SIZE (source units)", ylabel="Incidents with nearby detections (%)", title="Primary check: 5 km, days 0-14")
    axes[1].tick_params(axis="x", rotation=25)
    save(fig, "04_incident_support.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    subset = support.loc[support.year.eq(2023)]
    for state, color, label in [(False, "#cf5663", "No nearby record"), (True, "#168b82", "Nearby FIRMS record")]:
        g = subset.loc[subset.supported.eq(state)]
        for poly in polygons(bc):
            axes[0].plot(*poly.exterior.xy, color="gray", linewidth=.35)
        axes[0].scatter(g.longitude, g.latitude, s=6, c=color, label=f"{label} (n={len(g)})", alpha=.7)
    axes[0].set_aspect(1/np.cos(np.deg2rad(bc.centroid.y)))
    axes[0].set_title("2023 eligible incidents: 5 km / days 0-14")
    axes[0].legend(fontsize=7)
    year = pd.read_csv(OUT / "support_by_year.csv")
    axes[1].bar(year.year, year["mean"]*100, color="#168b82")
    for r in year.itertuples():
        axes[1].text(r.year, r.mean*100+1, f"n={r.size}", ha="center", fontsize=8)
    axes[1].set(ylim=(0, 100), ylabel="Incidents with nearby FIRMS activity (%)", title="Same support rule across years")
    save(fig, "05_support_geography.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    cluster = pd.read_csv(OUT / "clusters_gap3_2023.csv")
    values = cluster.detections.to_numpy()
    axes[0].plot(np.arange(1, len(values)+1), values.cumsum()/values.sum()*100, color="#d55e00")
    axes[0].set(xscale="log", xlabel="Number of largest provisional components", ylabel="Share of FIRMS observations (%)", title="2023 cluster concentration (3-day links)")
    sens = pd.read_csv(OUT / "cluster_sensitivity.csv")
    axes[1].bar(sens.gap_days.astype(str), sens.components, color="#168b82")
    axes[1].set(xlabel="Maximum day gap for adjacent-cell links", ylabel="Connected components", title="Cluster count depends on the rule")
    for i, r in enumerate(sens.itertuples()):
        axes[1].text(i, r.components, str(r.components), ha="center", va="bottom")
    axes[1].margins(y=.15)
    save(fig, "06_cluster_sensitivity.png")

    # Each occupied cell shows its dominant retrospective component across the season.
    cn = pd.read_csv(OUT / "cluster_nodes_gap3_2023.csv")
    dominant = cn.groupby(["h3_cell", "cluster"]).detections.sum().reset_index().sort_values("detections", ascending=False).drop_duplicates("h3_cell")
    palette = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(13, 8), layout="constrained")
    for ax, (name, geometry) in zip(axes, [("BC", bc), ("Kamloops", kamloops)]):
        draw_cells(ax, geometry, pd.Series(dtype=float), f"{name}: provisional components + incident points", True)
        scope_cells = set(f.loc[f.year.eq(2023) & intersects_xy(geometry, f.longitude.to_numpy(), f.latitude.to_numpy()), "h3_cell"])
        scoped = dominant.loc[dominant.h3_cell.isin(scope_cells)]
        top = list(scoped.groupby("cluster").detections.sum().nlargest(10).index)
        for cluster_id, group in scoped.groupby("cluster"):
            color = palette(top.index(cluster_id)) if cluster_id in top else "#c3c9cc"
            vertices = [[(lon, lat) for lat, lon in h3.cell_to_boundary(cell)] for cell in group.h3_cell]
            ax.add_collection(PolyCollection(vertices, facecolors=color, edgecolors="none", zorder=2))
        points = b.loc[b.year.eq(2023)]
        points = points.loc[intersects_xy(geometry, points.point_longitude.to_numpy(), points.point_latitude.to_numpy())]
        ax.scatter(points.point_longitude, points.point_latitude, marker="x", s=9, c="#1b2933", linewidths=.5, zorder=3)
    fig.suptitle("2023: ten largest components per panel are coloured; grey = other components\nColours reset per panel; crosses = BC recorded starts; components are not confirmed fires", fontsize=12)
    save(fig, "07_cluster_map.png")

    annual = pd.read_csv(OUT / "annual_source_counts.csv")
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    x = np.arange(len(annual))
    ax.bar(x-.2, annual.published_total_fires, width=.4, color="#506a7b", label="BC published annual total")
    ax.bar(x+.2, annual.local_fire_records, width=.4, color="#e88a34", label="Local source records classified Fire")
    ax.set(xticks=x, xticklabels=annual.year, ylabel="Fire records", title="Source completeness check before seasonal/date filtering")
    ax.legend()
    save(fig, "08_source_completeness.png")


def report():
    def table(name, limit=None):
        frame = pd.read_csv(OUT / name)
        if limit:
            frame = frame.head(limit)
        return '<div class="scroll">' + frame.to_html(index=False, float_format=lambda v:f"{v:,.3f}", border=0) + '</div>'
    def image(name):
        data = base64.b64encode((FIG / name).read_bytes()).decode()
        return f'<img src="data:image/png;base64,{data}" alt="{escape(name)}">'
    sections = ["<h1>FIRMS clusters versus BC incident locations</h1>",
        "<p>2018-2023 May-September; matched BC boundary; H3 resolution 7. Main maps: 2023. 2024 excluded.</p>",
        "<h2>1. Comparable scope</h2>", table("scope.csv"),
        image("08_source_completeness.png"), table("annual_source_counts.csv"),
        '<p>Coverage flag: the full local 2018 file contains 739 records of all types, while BC reports 2,117 fires. This gap exists before any seasonal/date filters. Published totals and the incident layer may differ in definition or revision; this file is not a verified complete census. <a href="https://www2.gov.bc.ca/gov/content/safety/wildfire-status/about-bcws/wildfire-statistics/wildfire-averages">BC annual totals</a> use an April-March reporting period. Calendar-year mismatch is excluded here, not automatically classified as bad data.</p>',
        "<p>FIRMS: presumed vegetation thermal detections (type 0). BC: Fire records with usable year-consistent dates. Missing records are not verified fire absence.</p>",
        image("01_matched_spatial_maps.png"), image("02_monthly_comparison.png"),
        "<h2>2. Repetition versus occupied footprint</h2>", image("03_concentration.png"), table("spatial_concentration.csv"),
        "<p>Top-cell shares are fractions of each source's own occupied cells. Effective cells = 1 / sum of squared cell shares. Equal-size sampling removes detection repetition and equalizes cell count; boxplots show sampling variation, not confidence intervals on independent fires.</p>",
        "<details><summary>H3 scale sensitivity and source quality</summary>", table("spatial_overlap_by_resolution.csv"), table("firms_filter_sensitivity.csv"), table("incident_quality.csv"), "</details>",
        "<h2>3. Which incident locations have nearby FIRMS activity?</h2>", image("04_incident_support.png"), image("05_support_geography.png"),
        table("support_by_recorded_size.csv"), table("support_by_year.csv"),
        "<p>Proximity is not a verified event match: another fire can supply the nearby detection. All sensitivity windows use the same May 4-August 31 cohort, with observations available through September 30. Endpoints are inclusive. Recorded size is retrospective, used only for diagnosis. No nearby detection does not prove a false incident or a satellite miss.</p>",
        "<h2>4. Provisional satellite clusters</h2>", image("06_cluster_sensitivity.png"), image("07_cluster_map.png"), table("cluster_sensitivity.csv"), table("clusters_gap3_2023.csv", 15),
        "<p>Nodes are positive H3 cell-days. Links join the same or adjacent cells across up to 1, 3 or 7 days. Connected chains can merge separate fires or split one fire. These are retrospective diagnostic groups, not ignition labels or confirmed incidents.</p>",
        "<h2>Interpretation and next label work</h2><ul><li>Compare occupied-cell maps before interpreting raw detection intensity as spatial coverage.</li><li>Audit dates, record types and observation availability before treating BC and FIRMS disagreements as label errors.</li><li>Use official perimeters and incident IDs to validate event grouping; retain ambiguous matches.</li><li>Keep recorded new starts and ongoing hotspot activity as separate targets. Never infer daily burn extent from a final perimeter.</li></ul>",
        '<p>Sources: <a href="https://firms.modaps.eosdis.nasa.gov/content/descriptions/FIRMS_VIIRS_Firehotspots.html">NASA FIRMS fields</a>; <a href="https://firms2.modaps.eosdis.nasa.gov/alerts/notes.php">NASA detection limitations</a>; <a href="https://delivery.maps.gov.bc.ca/arcgis/rest/services/whse/bcgw_pub_WHSE_LAND_AND_NATURAL_RESOURCE/MapServer/22">BC historical incidents</a>.</p>']
    style = "body{font:15px system-ui;color:#233844;background:#f5f7f8;margin:0}main{max-width:1250px;margin:auto;padding:28px}img{width:100%;background:white;margin:16px 0}h2{margin-top:36px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-size:12px}td,th{padding:8px;border-bottom:1px solid #dbe4e7;text-align:right}td:first-child,th:first-child{text-align:left}p{color:#536671}summary{cursor:pointer;font-weight:bold}"
    (HERE / "Result.html").write_text('<!doctype html><meta charset="utf-8"><title>FIRMS and BC incident diagnostic</title><style>'+style+'</style><main>'+''.join(sections)+'</main>', encoding="utf-8")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    print("Loading and checking common scope", flush=True)
    f, b, bc, kamloops = load()
    nodes = spatial_stats(f, b, kamloops)
    primary = incident_support(f, b)
    clusters(nodes)
    print("Building matched maps and diagnostic plots", flush=True)
    charts(f, b, nodes, bc, kamloops, primary)
    report()
    (OUT / "manifest.json").write_text(json.dumps(dict(
        source_hashes={str(p.relative_to(ROOT)):digest(p) for p in [FIRMS, INCIDENTS, BOUNDARY, CENTRES, Path(__file__), HERE / "reference_annual_totals.csv"]},
        years=list(range(2018,2024)), months=[5,6,7,8,9], h3_resolution=7,
        firms_filter="type == 0; no confidence exclusion in primary analysis",
        incidents_filter="Fire; inside BC; valid date matching FIRE_YEAR; May-Sep",
        support_cohort="May 4-August 31, same cohort for all windows",
        support_primary="within 5 km, 0-14 calendar days inclusive; proximity, not identity",
        clustering="same/adjacent H3 cell-days; temporal gaps 1,3,7; diagnostic only",
        holdout_evaluated=False), indent=2), encoding="utf-8")
    print(f"Wrote {HERE / 'Result.html'}", flush=True)


if __name__ == "__main__":
    main()
