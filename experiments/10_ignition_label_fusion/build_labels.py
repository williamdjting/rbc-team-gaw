"""Frozen ignition-label builder (v2): BC incidents + BC fire perimeters + FIRMS onsets.

Label definition (frozen; see LABEL_SPEC.md):
  Fires       BC `Fire` incidents inside BC with usable dates, plus perimeter-only fires
              (perimeter exists, incident record missing; mostly 2018). Ignition date is
              the BC local date (incident IGNITION_DATE date part, else perimeter FIRE_DATE).
  Footprint   Buffered final perimeter (PERIM_BUFFER_KM) when available, otherwise a disc
              of radius 1.5 x size-equivalent radius + POINT_BUFFER_KM around the point.
  Onset       FIRMS type-0 positive cell-day (H3 res 7) with no detection in its 2-ring
              (about 5 km) during the previous 7 days.
  Tier A      Fire with FIRMS activity in its footprint (or within 5 km of its point)
              from 3 days before to 5 days after the ignition date.
  Tier B      Fire FIRMS never saw in that window (mostly < 1 ha).
  Tier C      Onset not explained by any fire start, active footprint or nearby active fire,
              that keeps burning (>= 3 cell-days in its 2-ring within 14 days).
  Mask        Cells inside an active fire footprint during [ignition, out] (except the
              ignition cell-day itself), and onsets classed as early growth, spread,
              near-active-fire or fading FIRMS-only. Masked rows are neither positive nor
              negative for the ignition task.
Growth, final perimeters and out dates are future information: label construction only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from shapely import contains_xy, make_valid
from shapely.geometry import Point, shape
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_analysis as ra  # noqa: E402

ROOT = HERE.parents[1]
OUT = HERE / "results" / "labels_v2"
SPOT = OUT / "spotcheck"
PERIMETERS = ROOT / "data/bc_fire_perimeters_2018_2023.geojson"
BOUNDARY = ROOT / "datasets/bc_legal_boundary.geojson"
KM_LAT = 110.574
KM_LON_EQ = 111.320

ONSET_RING, ONSET_LOOKBACK = 2, 7
PERIM_BUFFER_KM = 2.0       # VIIRS pixel (375 m) + H3 cell-centre offset (<=1.2 km) + mapping error
POINT_BUFFER_KM = 5.0
START_MATCH_KM = 5.0
START_WINDOW = (-3, 5)      # FIRMS day minus ignition day (point match)
EARLY_IN_FOOTPRINT = -14    # FIRMS inside a fire's own perimeter up to 14 d before the BC date = same fire
STATIC_MIN_YEARS = 2        # unattributed detections in >= this many seasons => static heat source
LATE_SEASON_MONTH = 9       # FIRMS-only onsets from September on overlap slash-pile burning season
EARLY_GROWTH_DAYS = 5
NEAR_ACTIVE_KM = 10.0
GROW_MIN_CELL_DAYS = 3
OUT_DAY_DEFAULT = 150       # active window when no out date is recorded


def polygonal(geom):
    """Keep only polygon parts (make_valid can return GeometryCollections)."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    parts = [g for g in getattr(geom, "geoms", []) if g.geom_type in ("Polygon", "MultiPolygon")]
    return unary_union(parts) if parts else None


def local_km(geom, lat0):
    scale = np.array([np.cos(np.deg2rad(lat0)) * KM_LON_EQ, KM_LAT])
    return shapely.transform(geom, lambda xy: xy * scale)


def to_km(lon, lat, lat0):
    return np.asarray(lon) * np.cos(np.deg2rad(lat0)) * KM_LON_EQ, np.asarray(lat) * KM_LAT


def load_fires():
    bc = make_valid(unary_union([shape(f["geometry"]) for f in json.loads(BOUNDARY.read_text())["features"]]))
    raw = pd.read_csv(ra.RAW_INCIDENTS)
    raw = raw[raw["FIRE_TYPE"] == "Fire"].copy()
    ign = pd.to_datetime(raw["IGNITION_DATE_UTC"], errors="coerce", utc=True).dt.tz_localize(None)
    out = pd.to_datetime(raw["FIRE_OUT_DATE_UTC"], errors="coerce", utc=True).dt.tz_localize(None)
    ok = ign.notna() & raw["point_latitude"].notna() & (ign.dt.year.between(raw["FIRE_YEAR"] - 1, raw["FIRE_YEAR"] + 1))
    raw, ign, out = raw[ok], ign[ok], out[ok]
    inside = contains_xy(bc, raw["point_longitude"].to_numpy(), raw["point_latitude"].to_numpy())
    raw, ign, out = raw[inside], ign[inside], out[inside]
    inc = pd.DataFrame(dict(
        fire_id=raw["FIRE_YEAR"].astype(str) + ":" + raw["FIRE_NUMBER"], source="incident",
        ign_day=ra.epoch_day(ign.dt.normalize()), local_hour=ign.dt.hour,
        out_day=np.where(out.notna(), ra.epoch_day(out.dt.normalize().fillna(pd.Timestamp("1970-01-01"))), -1),
        lat=raw["point_latitude"].to_numpy(), lon=raw["point_longitude"].to_numpy(),
        size_ha=raw["CURRENT_SIZE"].fillna(0).clip(lower=0).to_numpy(), cause=raw["FIRE_CAUSE"].to_numpy()))

    g = json.loads(PERIMETERS.read_text())["features"]
    per = pd.DataFrame([f["properties"] for f in g])
    per["geometry"] = [polygonal(make_valid(shape(f["geometry"]))) if f["geometry"] else None for f in g]
    per = per[per["geometry"].notna()]
    per["fire_id"] = per["FIRE_YEAR"].astype(str) + ":" + per["FIRE_NUMBER"]
    geom = dict(zip(per["fire_id"], per["geometry"]))
    inc["geometry"] = inc["fire_id"].map(geom)

    only = per[~per["fire_id"].isin(inc["fire_id"])].copy()
    fdate = pd.to_datetime(only["FIRE_DATE_RAW_ISO"], errors="coerce", utc=True).dt.tz_localize(None)
    only = only[fdate.notna() & (fdate.dt.year == only["FIRE_YEAR"])]
    fdate = fdate[only.index]
    rep = [p.representative_point() for p in only["geometry"]]
    po = pd.DataFrame(dict(
        fire_id=only["fire_id"], source="perimeter_only", ign_day=ra.epoch_day(fdate), local_hour=np.nan,
        out_day=-1, lat=[p.y for p in rep], lon=[p.x for p in rep],
        size_ha=only["FIRE_SIZE_HECTARES"].fillna(0).to_numpy(), cause=only["FIRE_CAUSE"].to_numpy(),
        geometry=only["geometry"].to_numpy()))
    fires = pd.concat([inc, po], ignore_index=True)
    fires["out_day"] = np.where(fires["out_day"] < 0, fires["ign_day"] + OUT_DAY_DEFAULT, fires["out_day"])
    fires["out_day"] = np.maximum(fires["out_day"], fires["ign_day"])
    fires["has_perimeter"] = fires["geometry"].notna()
    fires["year"] = pd.to_datetime(fires["ign_day"], unit="D").dt.year
    return fires


def match(fires, cd):
    """Fire-centric matching of FIRMS cell-days. Returns long table of (cell-day, fire, relation)."""
    cd = cd.sort_values("day").reset_index(drop=True)
    days = cd["day"].to_numpy()
    lon, lat = cd["clon"].to_numpy(), cd["clat"].to_numpy()
    rows = []
    for f in fires.itertuples(index=False):
        lo = np.searchsorted(days, f.ign_day + (EARLY_IN_FOOTPRINT if f.has_perimeter else START_WINDOW[0]))
        hi = np.searchsorted(days, f.out_day, side="right")
        if hi <= lo:
            continue
        sl = np.arange(lo, hi)
        r_eq = np.sqrt(f.size_ha / 100 / np.pi)
        reach = (r_eq * 1.5 + POINT_BUFFER_KM) if not f.has_perimeter else PERIM_BUFFER_KM
        outer = NEAR_ACTIVE_KM + (reach if not f.has_perimeter else 0)
        if f.has_perimeter:
            minx, miny, maxx, maxy = f.geometry.bounds
        else:
            minx = maxx = f.lon
            miny = maxy = f.lat
        pad_lat = (outer + 1) / KM_LAT
        pad_lon = pad_lat / max(np.cos(np.deg2rad(f.lat)), 0.3)
        m = (lat[sl] >= miny - pad_lat) & (lat[sl] <= maxy + pad_lat) & (lon[sl] >= minx - pad_lon) & (lon[sl] <= maxx + pad_lon)
        sl = sl[m]
        if not len(sl):
            continue
        x, y = to_km(lon[sl], lat[sl], f.lat)
        px, py = to_km(f.lon, f.lat, f.lat)
        d_point = np.hypot(x - px, y - py)
        if f.has_perimeter:
            gk = local_km(f.geometry, f.lat)
            d_foot = np.array([gk.distance(Point(a, b)) for a, b in zip(x, y)]) if len(sl) < 400 else \
                gk.distance(np.array([Point(a, b) for a, b in zip(x, y)], dtype=object))
            d_foot = np.asarray(d_foot, float)
            in_foot = d_foot <= PERIM_BUFFER_KM
            near = d_foot <= PERIM_BUFFER_KM + NEAR_ACTIVE_KM
        else:
            in_foot = d_point <= reach
            near = d_point <= reach + NEAR_ACTIVE_KM
        lag = days[sl] - f.ign_day
        start = (lag <= START_WINDOW[1]) & (((lag >= START_WINDOW[0]) & (d_point <= START_MATCH_KM)) |
                                            ((lag >= EARLY_IN_FOOTPRINT) & in_foot))
        keep = start | (near & (lag >= START_WINDOW[0]))
        if keep.any():
            rows.append(pd.DataFrame(dict(cd_idx=sl[keep], fire_id=f.fire_id, lag=lag[keep], in_foot=in_foot[keep],
                                          start=start[keep], d_point=d_point[keep], size_ha=f.size_ha)))
    return cd, pd.concat(rows, ignore_index=True)


def static_cells(cd, links):
    """Cells whose 1-ring has fire-unrelated detections in >= STATIC_MIN_YEARS seasons, or FIRMS type 2.

    Catches gas flares, landfills, smelters and mines that FIRMS labels as vegetation (type 0).
    """
    unrelated = cd[~cd.index.isin(links["cd_idx"])]
    years = unrelated.groupby("h3_cell")["year"].unique()
    ring_years = {}
    for cell, ys in years.items():
        for nb in h3.grid_disk(cell, 1):
            ring_years.setdefault(nb, set()).update(ys)
    static = {c for c, ys in ring_years.items() if len(ys) >= STATIC_MIN_YEARS}
    t2 = pd.read_csv(ra.FIRMS, usecols=["type", "h3_cell", "acq_date"])
    t2 = t2[(t2["type"] == 2) & (pd.to_datetime(t2["acq_date"]).dt.year.isin(ra.YEARS))]
    for cell in t2["h3_cell"].unique():
        static.update(h3.grid_disk(cell, 1))
    return static


def classify(cd, links):
    s = links.groupby("cd_idx").agg(start=("start", "any"))
    act = links[links["in_foot"] & (links["lag"] >= START_WINDOW[0])]
    s = s.join(act.groupby("cd_idx").agg(in_active=("in_foot", "any"), max_age=("lag", "max")), how="left")
    cd = cd.join(s)
    cd["start"] = cd["start"].eq(True)
    cd["in_active"] = cd["in_active"].eq(True)
    cd["near_active"] = cd.index.isin(links["cd_idx"]) & ~cd["in_active"] & ~cd["start"]
    onset = cd[f"onset_k{ONSET_RING}_L{ONSET_LOOKBACK}"] & cd["in_season"]
    cd["onset_class"] = np.where(~onset, None, np.select(
        [cd["start"], cd["in_active"] & (cd["max_age"] <= EARLY_GROWTH_DAYS), cd["in_active"], cd["near_active"]],
        ["fire_start", "early_growth", "spread_or_flareup", "near_active_fire"], "unexplained"))
    static = static_cells(cd, links)
    cd["static_source"] = cd["h3_cell"].isin(static)
    month = pd.to_datetime(cd["day"], unit="D").dt.month
    un = cd["onset_class"] == "unexplained"
    cd.loc[un & cd["static_source"], "onset_class"] = "static_source"
    cd.loc[un & ~cd["static_source"] & (month >= LATE_SEASON_MONTH), "onset_class"] = "late_season_firms_only"
    o = cd[onset].copy()
    o["followup_cell_days_14d"] = ra.followup(cd, o)
    cd.loc[o.index, "followup_cell_days_14d"] = o["followup_cell_days_14d"]
    cd["grows"] = cd["followup_cell_days_14d"] >= GROW_MIN_CELL_DAYS
    # Attribution of every cell-day (repetition view).
    first = links.loc[links["in_foot"]].sort_values(["cd_idx", "size_ha"], ascending=[True, False]).drop_duplicates("cd_idx")
    cd["fire_id"] = cd.index.map(first.set_index("cd_idx")["fire_id"])
    cd["fire_age"] = cd.index.map(first.set_index("cd_idx")["lag"])
    return cd


def events(fires, cd, links):
    season = fires[ra.season_mask(fires["ign_day"]) & fires["year"].isin(ra.YEARS)].copy()
    st = links[links["start"]].merge(cd[["day", "h3_cell", "clat", "clon"]], left_on="cd_idx", right_index=True)
    first = st.sort_values(["fire_id", "lag", "d_point"]).drop_duplicates("fire_id").set_index("fire_id")
    season["firms_first_lag"] = season["fire_id"].map(first["lag"])
    season["tier"] = np.where(season["firms_first_lag"].notna(), "A", "B")
    # Location: incident point; perimeter-only fires use their first FIRMS start cell, else the polygon point.
    season["h3_cell"] = [h3.latlng_to_cell(a, b, 7) for a, b in zip(season["lat"], season["lon"])]
    season["location_source"] = np.where(season["source"] == "incident", "incident_point", "perimeter_point")
    po = (season["source"] == "perimeter_only") & season["firms_first_lag"].notna()
    season.loc[po, "h3_cell"] = season.loc[po, "fire_id"].map(first["h3_cell"])
    season.loc[po, "lat"] = season.loc[po, "fire_id"].map(first["clat"])
    season.loc[po, "lon"] = season.loc[po, "fire_id"].map(first["clon"])
    season.loc[po, "location_source"] = "first_firms_cell"
    # FIRMS cannot see a fire before it exists: if it saw this fire inside its own perimeter
    # before the BC date, the earlier day is the better ignition estimate.
    early = season["firms_first_lag"] < 0
    season["bc_date"] = pd.to_datetime(season["ign_day"], unit="D").dt.date
    season["day"] = season["ign_day"] + np.where(early, season["firms_first_lag"], 0).astype(int)
    season["date_source"] = np.where(early, "firms_earlier_than_bc", "bc")
    ev = season[["fire_id", "source", "tier", "day", "bc_date", "date_source", "h3_cell", "lat", "lon",
                 "location_source", "cause", "size_ha", "has_perimeter", "local_hour", "firms_first_lag",
                 "year"]].rename(columns={"fire_id": "event_id"})
    c = cd[(cd["onset_class"] == "unexplained") & cd["grows"]]
    tc = pd.DataFrame(dict(event_id=[f"FIRMS:{a}:{d}" for a, d in zip(c["h3_cell"], c["day"])], source="firms_only",
                           tier="C", day=c["day"], h3_cell=c["h3_cell"], lat=c["clat"], lon=c["clon"],
                           bc_date=None, date_source="firms_onset",
                           location_source="firms_onset_cell", cause="Unknown", size_ha=np.nan, has_perimeter=False,
                           local_hour=np.nan, firms_first_lag=0.0, year=c["year"]))
    ev = pd.concat([ev, tc], ignore_index=True)
    ev["date"] = pd.to_datetime(ev["day"], unit="D").dt.date
    # Spot-checks showed tier C is dominated by industrial sources and burn piles: off by default.
    ev["in_default_label"] = ev["tier"].isin(["A", "B"])
    return ev


def footprints(fires):
    rows = []
    for f in fires.itertuples(index=False):
        if f.has_perimeter:
            cells = set(h3.geo_to_cells(f.geometry.__geo_interface__, 7))
        else:
            cells = set()
        cells.add(h3.latlng_to_cell(f.lat, f.lon, 7))
        rows.append(pd.DataFrame(dict(fire_id=f.fire_id, h3_cell=sorted(cells), start_day=f.ign_day, end_day=f.out_day)))
    fp = pd.concat(rows, ignore_index=True)
    fp["start_date"] = pd.to_datetime(fp["start_day"], unit="D").dt.date
    fp["end_date"] = pd.to_datetime(fp["end_day"], unit="D").dt.date
    return fp


def spotcheck(cd, fires, n=40, seed=7):
    SPOT.mkdir(parents=True, exist_ok=True)
    c = cd[(cd["onset_class"] == "unexplained") & cd["grows"]]
    per_year = max(1, n // c["year"].nunique())
    sample = pd.concat([g.sample(min(len(g), per_year), random_state=seed) for _, g in c.groupby("year")])
    sample = sample.sort_values(["year", "day"]).head(n)
    for old in SPOT.glob("tier_c_sheet_*.png"):
        old.unlink()
    review = []
    for sheet, start in enumerate(range(0, len(sample), 10)):
        chunk = sample.iloc[start:start + 10]
        fig, axes = plt.subplots(2, 5, figsize=(22, 9))
        for ax, (_, o) in zip(axes.ravel(), chunk.iterrows()):
            win = cd[(cd["day"].between(o["day"] - 10, o["day"] + 20)) &
                     (np.abs(cd["clat"] - o["clat"]) < 0.25) & (np.abs(cd["clon"] - o["clon"]) < 0.4)]
            sc = ax.scatter(win["clon"], win["clat"], c=win["day"] - o["day"], cmap="coolwarm", vmin=-10, vmax=20,
                            s=8 + 3 * np.sqrt(win["n_det"]))
            ax.scatter([o["clon"]], [o["clat"]], marker="*", s=260, facecolor="none", edgecolor="k", linewidth=1.5)
            near = fires[(np.abs(fires["lat"] - o["clat"]) < 0.6) & (np.abs(fires["lon"] - o["clon"]) < 0.9) &
                         (fires["ign_day"].between(o["day"] - 120, o["day"] + 30))]
            for f in near.itertuples():
                if f.has_perimeter:
                    for poly in getattr(f.geometry, "geoms", [f.geometry]):
                        if poly.geom_type == "Polygon":
                            xx, yy = poly.exterior.xy
                            ax.plot(xx, yy, color="green", lw=0.8)
                ax.plot(f.lon, f.lat, "g^", ms=5)
                ax.annotate(f"{f.fire_id.split(':')[1]} d{int(f.ign_day - o['day'])} {f.size_ha:.0f}ha",
                            (f.lon, f.lat), fontsize=6, color="darkgreen")
            ax.set_xlim(o["clon"] - 0.4, o["clon"] + 0.4); ax.set_ylim(o["clat"] - 0.25, o["clat"] + 0.25)
            k = len(review)
            ax.set_title(f"#{k} {pd.to_datetime(o['day'], unit='D').date()} follow={int(o['followup_cell_days_14d'])} "
                         f"frp={o['frp_max']:.0f}", fontsize=8)
            ax.tick_params(labelsize=6)
            review.append(dict(review_id=k, h3_cell=o["h3_cell"], date=pd.to_datetime(o["day"], unit="D").date(),
                               lat=o["clat"], lon=o["clon"], n_det=o["n_det"], frp_max=o["frp_max"],
                               followup_cell_days_14d=o["followup_cell_days_14d"], sheet=sheet,
                               verdict="", notes=""))
        for ax in axes.ravel()[len(chunk):]:
            ax.axis("off")
        fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.6, label="day relative to onset")
        fig.suptitle("Tier C spot-check: star = onset cell; dots = FIRMS cell-days (colour = day); "
                     "green = BC perimeters/points (label: fire, ignition day vs onset, size)", fontsize=10)
        fig.savefig(SPOT / f"tier_c_sheet_{sheet}.png", dpi=110); plt.close(fig)
    pd.DataFrame(review).to_csv(SPOT / "tier_c_review.csv", index=False)


def compare_v1(ev, cd):
    v1 = pd.read_csv(HERE / "results/fused_ignition_events.csv")
    v1t = pd.crosstab(v1["year"], v1["tier"].str[0]).add_prefix("v1_")
    v2t = pd.crosstab(ev["year"], ev["tier"]).add_prefix("v2_")
    v2s = pd.crosstab(ev["year"], ev["source"]).add_prefix("v2_src_")
    t = v1t.join(v2t).join(v2s)
    t["v1_total"] = v1t.sum(axis=1)
    t["v2_total"] = v2t.sum(axis=1)
    on = cd[cd["onset_class"].notna()]
    oc = pd.crosstab(on["year"], on["onset_class"] + np.where(on["onset_class"] == "unexplained",
                                                              np.where(on["grows"], "_grows", "_fades"), ""))
    s = cd[cd["in_season"]].copy()
    s["bucket"] = np.select([s["fire_age"].isna(), s["fire_age"] <= 1, s["fire_age"] <= 7],
                            ["no fire attributable", "day 0-1", "day 2-7"], "day 8+")
    rep = pd.crosstab(s["year"], s["bucket"], normalize="index")
    return t, oc, rep


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    firms = ra.load_firms()
    cd = ra.cell_days(firms)
    cd = ra.onset_flags(cd, ks=(ONSET_RING,), lookbacks=(ONSET_LOOKBACK,))
    fires = load_fires()
    cd, links = match(fires, cd)
    cd = classify(cd, links)
    ev = events(fires, cd, links)
    fp = footprints(fires)
    onset_mask = cd[cd["onset_class"].isin(["early_growth", "spread_or_flareup", "near_active_fire",
                                            "static_source", "late_season_firms_only"]) |
                    ((cd["onset_class"] == "unexplained") & ~cd["grows"])]
    mask = pd.DataFrame(dict(h3_cell=onset_mask["h3_cell"], date=pd.to_datetime(onset_mask["day"], unit="D").dt.date,
                             reason=np.where(onset_mask["onset_class"] == "unexplained", "fading_firms_only",
                                             onset_mask["onset_class"])))

    ev.drop(columns="day").to_csv(OUT / "ignition_events.csv", index=False)
    fp.drop(columns=["start_day", "end_day"]).to_csv(OUT / "fire_footprints.csv.gz", index=False, compression="gzip")
    mask.to_csv(OUT / "onset_mask.csv", index=False)
    cd.drop(columns=[c for c in cd.columns if c.startswith("gap_")]).to_csv(
        OUT / "firms_cell_days_v2.csv.gz", index=False, compression="gzip")
    fires.drop(columns="geometry").to_csv(OUT / "fires.csv", index=False)
    t, oc, rep = compare_v1(ev, cd)
    t.to_csv(OUT / "events_v1_vs_v2.csv"); oc.to_csv(OUT / "onset_classes_v2.csv"); rep.to_csv(OUT / "repetition_v2.csv")
    by_size = ev[ev["source"] != "firms_only"].groupby(
        pd.cut(ev["size_ha"], ra.SIZE_BINS, labels=ra.SIZE_LABELS), observed=False)["tier"].apply(lambda s: (s == "A").mean())
    by_size.to_csv(OUT / "tier_a_share_by_size.csv")
    spotcheck(cd, fires)
    spec = dict(onset_ring=ONSET_RING, onset_lookback_days=ONSET_LOOKBACK, perimeter_buffer_km=PERIM_BUFFER_KM,
                point_buffer_km=POINT_BUFFER_KM, start_match_km=START_MATCH_KM, start_window_days=START_WINDOW,
                early_growth_days=EARLY_GROWTH_DAYS, near_active_km=NEAR_ACTIVE_KM,
                grow_min_cell_days=GROW_MIN_CELL_DAYS, out_day_default=OUT_DAY_DEFAULT,
                fires=len(fires), fires_with_perimeter=int(fires["has_perimeter"].sum()),
                perimeter_only_fires=int((fires["source"] == "perimeter_only").sum()),
                events=ev["tier"].value_counts().to_dict(), mask_rows=mask["reason"].value_counts().to_dict(),
                early_in_footprint_days=EARLY_IN_FOOTPRINT, static_min_years=STATIC_MIN_YEARS,
                late_season_month=LATE_SEASON_MONTH, static_cells=int(cd["static_source"].groupby(cd["h3_cell"]).first().sum()),
                firms_cell_days_in_static_cells=int((cd["static_source"] & cd["in_season"]).sum()),
                date_source=ev["date_source"].value_counts().to_dict(),
                footprint_rows=len(fp))
    (OUT / "label_manifest.json").write_text(json.dumps(spec, indent=2, default=str))
    pd.set_option("display.width", 200, "display.max_columns", 30)
    for name, obj in [("events v1 vs v2", t), ("onset classes v2", oc), ("repetition v2", rep),
                      ("tier A share by size", by_size), ("manifest", pd.Series(spec))]:
        print(f"\n=== {name}\n{obj}")


if __name__ == "__main__":
    main()
