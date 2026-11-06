"""FIRMS x BC incident fusion diagnostics for building an ignition (new-fire) label.

Questions:
  1. How soon after a BC recorded ignition does FIRMS first see the fire, by size/cause?
  2. How many FIRMS positive cell-days are genuinely "new" (onsets) versus repetition
     of fires that were already burning nearby?
  3. Which FIRMS onsets coincide with a BC ignition, which are spread/spotting from a
     known active fire, and which are unexplained?
  4. How many positives would each candidate ignition label have per year?

Inputs: FIRMS joined CSV (weather ignored) and Experiment 09's audited in-scope
BC incidents (Fire records, usable dates, inside BC). 2024 is excluded.

BC IGNITION_DATE epochs encode Pacific local clock time as if it were UTC (see
FINDINGS.md); the date part is therefore the local date. VIIRS S-NPP overpasses
over BC fall on the same UTC and local calendar date, so dates align directly.
"""
from __future__ import annotations

import bisect
import json
from pathlib import Path

import h3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results"
FIG = OUT / "figures"
FIRMS = ROOT / "data/firms_with_cems_18_24.csv"
RAW_INCIDENTS = ROOT / "data/bc_historical_incidents_2018_2023.csv"
INCIDENTS = ROOT / "experiments/09_firms_incident_diagnostics/results/incidents_in_scope.csv"
EARTH_KM = 6371.0088
YEARS = range(2018, 2024)
SEASON_START = (5, 8)   # first target day with a full 7-day lookback, as in Experiments 02-07
SIZE_BINS = [-0.001, 0.01, 0.1, 1, 10, 100, 1000, np.inf]
SIZE_LABELS = ["<=0.01", "0.01-0.1", "0.1-1", "1-10", "10-100", "100-1000", ">1000"]
PUBLISHED = {2018: 2117, 2019: 825, 2020: 670, 2021: 1647, 2022: 1801, 2023: 2293}


def xyz(lat, lon):
    lat, lon = np.deg2rad(np.asarray(lat, float)), np.deg2rad(np.asarray(lon, float))
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def km_to_chord(km):
    return 2 * np.sin(np.asarray(km) / (2 * EARTH_KM))


def chord_to_km(chord):
    return 2 * EARTH_KM * np.arcsin(np.clip(np.asarray(chord) / 2, 0, 1))


def epoch_day(dates):
    return (pd.to_datetime(dates).values.astype("datetime64[D]").astype(np.int64))


def season_mask(day):
    d = pd.to_datetime(np.asarray(day), unit="D")
    return ((d.month > SEASON_START[0]) | ((d.month == SEASON_START[0]) & (d.day >= SEASON_START[1]))) & (d.month <= 9)


def load_firms():
    cols = ["latitude", "longitude", "acq_date", "acq_time", "type", "confidence", "frp", "daynight", "h3_cell"]
    d = pd.read_csv(FIRMS, usecols=cols)
    d = d[(d["type"] == 0)].copy()
    d["date"] = pd.to_datetime(d["acq_date"])
    d = d[d["date"].dt.year.isin(YEARS)].copy()
    d["day"] = epoch_day(d["date"])
    d["year"] = d["date"].dt.year
    return d.reset_index(drop=True)


def load_incidents():
    i = pd.read_csv(INCIDENTS)
    i = i[(i["FIRE_TYPE"] == "Fire") & (i["date_status"] == "usable") & i["inside_bc"]].copy()
    i["ign_day"] = epoch_day(i["date"])
    out = pd.to_datetime(i["FIRE_OUT_DATE_UTC"], errors="coerce", utc=True).dt.tz_localize(None)
    i["out_day"] = epoch_day(out.dt.normalize().fillna(pd.Timestamp("2100-01-01")))
    i.loc[out.isna(), "out_day"] = i.loc[out.isna(), "ign_day"] + 180
    i["out_day"] = np.maximum(i["out_day"], i["ign_day"])
    i["local_hour"] = pd.to_datetime(i["IGNITION_DATE_UTC"], utc=True).dt.hour
    i["size_ha"] = i["CURRENT_SIZE"].fillna(0).clip(lower=0)
    i["size_bin"] = pd.cut(i["size_ha"], SIZE_BINS, labels=SIZE_LABELS)
    i["lat"], i["lon"] = i["point_latitude"], i["point_longitude"]
    # Final-perimeter equivalent radius (retrospective, diagnostic attribution only).
    i["radius_km"] = np.sqrt(i["size_ha"] / 100 / np.pi)
    return i.reset_index(drop=True)


# ---------------------------------------------------------------- 0. timestamp audit
def timestamp_audit():
    b = pd.read_csv(RAW_INCIDENTS)
    b = b[b["FIRE_TYPE"] == "Fire"]
    hour = pd.to_datetime(b["IGNITION_DATE_UTC"], errors="coerce", utc=True).dt.hour
    tab = pd.crosstab(hour, b["FIRE_CAUSE"]).reset_index().rename(columns={"IGNITION_DATE_UTC": "raw_hour"})
    tab.to_csv(OUT / "ignition_raw_hour_by_cause.csv", index=False)
    comp = (b[b["FIRE_YEAR"].isin(YEARS)].groupby("FIRE_YEAR").size().rename("local_fire_records").to_frame())
    comp["published_total"] = pd.Series(PUBLISHED)
    comp["coverage"] = comp["local_fire_records"] / comp["published_total"]
    sizes = pd.cut(b["CURRENT_SIZE"].fillna(0), SIZE_BINS, labels=SIZE_LABELS).astype(str)
    size_tab = pd.crosstab(b["FIRE_YEAR"], sizes)[SIZE_LABELS]
    size_tab.columns = [f"size_{c}" for c in SIZE_LABELS]
    comp = comp.join(size_tab)
    comp.to_csv(OUT / "incident_completeness.csv")
    return tab, comp


# ---------------------------------------------------------------- 1. BC -> first FIRMS
def first_detection(inc, firms, radius_km=(2, 5), pre=7, post=14):
    tree = cKDTree(xyz(firms["latitude"], firms["longitude"]))
    fday = firms["day"].to_numpy()
    pts = xyz(inc["lat"], inc["lon"])
    rows = []
    rmax = max(radius_km)
    hits = tree.query_ball_point(pts, km_to_chord(rmax))
    for k, idx in enumerate(hits):
        rec = {}
        idx = np.asarray(idx, dtype=int)
        d0 = inc.at[k, "ign_day"]
        if len(idx):
            dist = chord_to_km(np.linalg.norm(tree.data[idx] - pts[k], axis=1))
            days = fday[idx] - d0
        else:
            dist = days = np.array([])
        for r in radius_km:
            m = dist <= r
            win = days[m & (days >= -3) & (days <= post)]
            rec[f"first_lag_{r}km"] = win.min() if len(win) else np.nan
            rec[f"prior_activity_{r}km"] = bool(np.any(m & (days >= -pre) & (days <= -4)))
            rec[f"n_det_0_{post}_{r}km"] = int(np.sum(m & (days >= 0) & (days <= post)))
        rows.append(rec)
    res = pd.concat([inc[["event_key", "year", "ign_day", "size_ha", "size_bin", "FIRE_CAUSE", "local_hour",
                          "lat", "lon", "h3_cell"]], pd.DataFrame(rows)], axis=1)
    # Cohort with a complete FIRMS window on both sides of the ignition date.
    dt = pd.to_datetime(res["ign_day"], unit="D")
    res["window_complete"] = ((dt.dt.month > 5) | ((dt.dt.month == 5) & (dt.dt.day >= 8))) & (
        (dt.dt.month < 9) | ((dt.dt.month == 9) & (dt.dt.day <= 16)))
    return res


# ---------------------------------------------------------------- 2. FIRMS cell-days and onsets
def cell_days(firms):
    g = firms.groupby(["h3_cell", "day"]).agg(n_det=("frp", "size"), frp_sum=("frp", "sum"),
                                              frp_max=("frp", "max")).reset_index()
    ll = np.array([h3.cell_to_latlng(c) for c in g["h3_cell"].unique()])
    centres = pd.DataFrame({"h3_cell": g["h3_cell"].unique(), "clat": ll[:, 0], "clon": ll[:, 1]})
    g = g.merge(centres, on="h3_cell")
    g["year"] = pd.to_datetime(g["day"], unit="D").dt.year
    g["in_season"] = season_mask(g["day"])
    return g


def onset_flags(cd, ks=(0, 1, 2, 3), lookbacks=(3, 7, 14)):
    """Onset(k, L): no detection in the k-ring of the cell during the previous L days."""
    days_by_cell = {c: np.sort(v.to_numpy()) for c, v in cd.groupby("h3_cell")["day"]}
    rings = {}
    out = {}
    for k in ks:
        last_gap = np.empty(len(cd))
        for j, (cell, day) in enumerate(zip(cd["h3_cell"].to_numpy(), cd["day"].to_numpy())):
            key = (cell, k)
            if key not in rings:
                rings[key] = [n for n in h3.grid_disk(cell, k) if n in days_by_cell]
            gap = np.inf
            for n in rings[key]:
                arr = days_by_cell[n]
                p = bisect.bisect_left(arr, day) - 1
                if p >= 0:
                    gap = min(gap, day - arr[p])
            last_gap[j] = gap
        cd[f"gap_k{k}"] = last_gap
        for L in lookbacks:
            out[(k, L)] = last_gap > L
            cd[f"onset_k{k}_L{L}"] = last_gap > L
    return cd


# ---------------------------------------------------------------- 3. attribute FIRMS cell-days to incidents
def attribute(cd, inc, buffer_km=5.0, pre_days=3):
    """For each FIRMS cell-day, find BC incidents that could explain it.

    new_match: an incident whose recorded ignition is within [day-5, day+pre_days] and
               whose point is within buffer_km of the cell centre (a start).
    active:    an incident burning that day (ign-pre_days <= day <= out) whose point is
               within its equivalent final radius * 1.5 + buffer_km (ongoing fire).
    Uses final size, so attribution is retrospective and diagnostic only.
    """
    itree = cKDTree(xyz(inc["lat"], inc["lon"]))
    reach = inc["radius_km"].to_numpy() * 1.5 + buffer_km
    ign, out = inc["ign_day"].to_numpy(), inc["out_day"].to_numpy()
    cpts = xyz(cd["clat"], cd["clon"])
    cand = itree.query_ball_point(cpts, km_to_chord(reach.max()))
    days = cd["day"].to_numpy()
    new_idx = np.full(len(cd), -1)
    new_lag = np.full(len(cd), np.nan)
    act_idx = np.full(len(cd), -1)
    act_age = np.full(len(cd), np.nan)
    act_dist = np.full(len(cd), np.nan)
    for j, idx in enumerate(cand):
        if not idx:
            continue
        idx = np.asarray(idx)
        dist = chord_to_km(np.linalg.norm(itree.data[idx] - cpts[j], axis=1))
        lag = days[j] - ign[idx]                      # FIRMS day minus BC ignition day
        m_new = (dist <= buffer_km) & (lag >= -pre_days) & (lag <= 5)
        if m_new.any():
            c = idx[m_new][np.argmin(np.abs(lag[m_new]) + dist[m_new] / 100)]
            new_idx[j], new_lag[j] = c, days[j] - ign[c]
        m_act = (dist <= reach[idx]) & (lag >= -pre_days) & (days[j] <= out[idx])
        if m_act.any():
            # Prefer the largest burning fire, then the nearest point.
            sub = idx[m_act]
            c = sub[np.lexsort((dist[m_act], -inc["size_ha"].to_numpy()[sub]))[0]]
            act_idx[j], act_age[j] = c, days[j] - ign[c]
            act_dist[j] = dist[m_act][np.where(sub == c)[0][0]]
    cd["new_incident"] = np.where(new_idx >= 0, inc["event_key"].to_numpy()[np.maximum(new_idx, 0)], None)
    cd["new_lag"] = new_lag
    cd["active_incident"] = np.where(act_idx >= 0, inc["event_key"].to_numpy()[np.maximum(act_idx, 0)], None)
    cd["active_age_days"] = act_age
    cd["active_dist_km"] = act_dist
    cd["active_size_ha"] = np.where(act_idx >= 0, inc["size_ha"].to_numpy()[np.maximum(act_idx, 0)], np.nan)
    return cd


def followup(cd, onsets, k=2, horizon=14):
    """FIRMS cell-days in the onset's k-ring over the next `horizon` days (retrospective)."""
    by_cell = {c: np.sort(v.to_numpy()) for c, v in cd.groupby("h3_cell")["day"]}
    counts = []
    for cell, day in zip(onsets["h3_cell"].to_numpy(), onsets["day"].to_numpy()):
        n = 0
        for nb in h3.grid_disk(cell, k):
            arr = by_cell.get(nb)
            if arr is not None:
                n += bisect.bisect_right(arr, day + horizon) - bisect.bisect_right(arr, day)
        counts.append(n)
    return np.asarray(counts)


def classify_onsets(cd, col="onset_k2_L7"):
    o = cd[cd[col] & cd["in_season"]].copy()
    o["onset_class"] = np.select(
        [o["new_incident"].notna(),
         o["active_incident"].notna() & (o["active_age_days"] > 5),
         o["active_incident"].notna()],
        ["matches_bc_ignition", "spread_or_flareup_of_active_fire", "near_recent_bc_fire"],
        "unexplained_firms_only")
    o["followup_cell_days_14d"] = followup(cd, o)
    o["grows"] = o["followup_cell_days_14d"] >= 3
    o["month"] = pd.to_datetime(o["day"], unit="D").dt.month
    return o


def fused_events(inc, fd, onsets):
    """Prototype ignition-event table combining both sources.

    Tier A: BC ignition with a FIRMS onset/first detection nearby (5 km, day -3..+5).
    Tier B: BC ignition FIRMS never saw (mostly <1 ha; real but satellite-invisible).
    Tier C: FIRMS-only onset, not near any recorded start or active BC fire, that keeps
            burning (>=3 cell-days in its 2-ring within 14 days): probable unrecorded fire.
    Mask:   FIRMS-only onsets that fade (likely managed/agricultural burns or noise) and all
            spread/flare-up onsets of active fires. These are neither ignitions nor negatives.
    The 'grows' criterion and attribution use future information: label construction only,
    never model inputs.
    """
    season = inc[season_mask(inc["ign_day"])].merge(
        fd[["event_key", "first_lag_5km"]], on="event_key", how="left")
    seen_by_onset = set(onsets.loc[onsets["onset_class"] == "matches_bc_ignition", "new_incident"])
    lag = season["first_lag_5km"]
    season["tier"] = np.where(season["event_key"].isin(seen_by_onset) | lag.between(-3, 5), "A_bc_firms", "B_bc_only")
    bc = pd.DataFrame(dict(event_id=season["event_key"], tier=season["tier"], h3_cell=season["h3_cell"],
                           date=pd.to_datetime(season["ign_day"], unit="D").dt.date, lat=season["lat"],
                           lon=season["lon"], cause=season["FIRE_CAUSE"], final_size_ha=season["size_ha"],
                           local_hour=season["local_hour"], firms_first_lag_days=lag))
    u = onsets[onsets["onset_class"] == "unexplained_firms_only"]
    keep = u[u["grows"]]
    fo = pd.DataFrame(dict(event_id=[f"FIRMS:{c}:{d}" for c, d in zip(keep["h3_cell"], keep["day"])],
                           tier="C_firms_only_growing", h3_cell=keep["h3_cell"],
                           date=pd.to_datetime(keep["day"], unit="D").dt.date, lat=keep["clat"], lon=keep["clon"],
                           cause="unknown", final_size_ha=np.nan, local_hour=np.nan, firms_first_lag_days=0.0))
    masked = onsets[(onsets["onset_class"] != "matches_bc_ignition") & ~onsets.index.isin(keep.index)]
    mask = pd.DataFrame(dict(h3_cell=masked["h3_cell"], date=pd.to_datetime(masked["day"], unit="D").dt.date,
                             reason=np.where(masked["onset_class"] == "unexplained_firms_only",
                                             "firms_only_fading_onset", masked["onset_class"])))
    ev = pd.concat([bc, fo], ignore_index=True)
    ev["year"] = pd.to_datetime(ev["date"]).dt.year
    return ev, mask


# ---------------------------------------------------------------- 4. label tallies
def label_tallies(cd, inc, fd):
    season_inc = inc[season_mask(inc["ign_day"])].copy()
    fd = fd.set_index("event_key")
    season_inc["firms_seen"] = season_inc["event_key"].map(fd["first_lag_5km"]).notna()
    rows = []
    for y in YEARS:
        c = cd[(cd["year"] == y) & cd["in_season"]]
        si = season_inc[season_inc["year"] == y]
        on = c[c["onset_k2_L7"]]
        unexplained = on[on["new_incident"].isna() & on["active_incident"].isna()]
        rows.append(dict(
            year=y,
            L0_firms_positive_cell_days=len(c),
            L0_detections=int(c["n_det"].sum()),
            L1_bc_ignitions=len(si),
            L1_bc_ignition_cell_days=si[["h3_cell", "ign_day"]].drop_duplicates().shape[0],
            L1a_bc_ignitions_firms_confirmed=int(si["firms_seen"].sum()),
            L1b_bc_ignitions_size_ge_1ha=int((si["size_ha"] >= 1).sum()),
            L2_firms_onsets_k2_L7=len(on),
            L2_firms_onsets_k1_L7=int(c["onset_k1_L7"].sum()),
            L2_firms_onsets_k0_L7=int(c["onset_k0_L7"].sum()),
            L3_fused_bc_plus_unexplained_firms=len(si) + len(unexplained),
            unexplained_firms_onsets=len(unexplained),
        ))
    t = pd.DataFrame(rows)
    t["L0_per_L1"] = t["L0_firms_positive_cell_days"] / t["L1_bc_ignitions"]
    return t


def repetition_breakdown(cd):
    s = cd[cd["in_season"]].copy()
    s["bucket"] = np.select(
        [s["new_incident"].notna() & (s["new_lag"] <= 1),
         s["active_incident"].notna() & (s["active_age_days"] <= 1),
         s["active_incident"].notna() & (s["active_age_days"] <= 7),
         s["active_incident"].notna()],
        ["day 0-1 of a BC fire", "day 0-1 of a BC fire", "day 2-7 of a BC fire", "day 8+ of a BC fire"],
        "no BC fire attributable")
    t = pd.crosstab(s["year"], s["bucket"])
    t = t.div(t.sum(axis=1), axis=0)
    per = (s[s["active_incident"].notna()].groupby(["year", "active_incident"]).size()
           .rename("cell_days").reset_index().sort_values(["year", "cell_days"], ascending=[True, False]))
    top = []
    for y, g in per.groupby("year"):
        tot = (s["year"] == y).sum()
        top.append(dict(year=y, firms_cell_days=tot, attributed_incidents=len(g),
                        share_top1_incident=g["cell_days"].iloc[:1].sum() / tot,
                        share_top5_incidents=g["cell_days"].iloc[:5].sum() / tot,
                        share_top20_incidents=g["cell_days"].iloc[:20].sum() / tot))
    return t, pd.DataFrame(top), per


def plots(fd, onsets, tallies, rep):
    FIG.mkdir(parents=True, exist_ok=True)
    c = fd[fd["window_complete"]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    seen = c.groupby("size_bin", observed=False)["first_lag_5km"].apply(lambda s: s.notna().mean())
    axes[0].bar(seen.index.astype(str), seen.values, color="#d9480f")
    axes[0].set_title("BC ignitions seen by FIRMS (5 km, day -3..+14)")
    axes[0].set_xlabel("recorded final size (ha)"); axes[0].set_ylabel("share"); axes[0].tick_params(axis="x", rotation=30)
    lag = c["first_lag_5km"].dropna().clip(-3, 14)
    axes[1].hist(lag, bins=np.arange(-3.5, 15.5), color="#1c7ed6")
    axes[1].set_title("First FIRMS day minus BC ignition date"); axes[1].set_xlabel("days")
    oc = onsets.groupby(["year", "onset_class"]).size().unstack(fill_value=0)
    oc.plot(kind="bar", stacked=True, ax=axes[2], colormap="tab10")
    axes[2].set_title("FIRMS onsets (k=2 ring, 7-day quiet)"); axes[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "bc_firms_matching.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    t = tallies.set_index("year")
    t[["L0_firms_positive_cell_days", "L2_firms_onsets_k2_L7", "L1_bc_ignitions",
       "L3_fused_bc_plus_unexplained_firms"]].plot(ax=axes[0], marker="o", logy=True)
    axes[0].set_title("Positives per season by label definition (log)")
    rep.plot(kind="bar", stacked=True, ax=axes[1], colormap="viridis")
    axes[1].set_title("What FIRMS positive cell-days are (attribution)"); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "label_counts_and_repetition.png", dpi=130); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hours, comp = timestamp_audit()
    firms = load_firms()
    inc = load_incidents()
    fd = first_detection(inc, firms)
    fd.to_csv(OUT / "incident_first_detection.csv", index=False)
    c = fd[fd["window_complete"]]
    by_size = c.groupby("size_bin", observed=False).agg(
        incidents=("event_key", "size"),
        seen_2km=("first_lag_2km", lambda s: s.notna().mean()),
        seen_5km=("first_lag_5km", lambda s: s.notna().mean()),
        median_lag_5km=("first_lag_5km", "median"),
        share_lag_le1=("first_lag_5km", lambda s: (s <= 1).sum() / max(s.notna().sum(), 1)),
        share_firms_before_bc=("first_lag_5km", lambda s: (s < 0).sum() / max(s.notna().sum(), 1)),
        prior_activity_5km=("prior_activity_5km", "mean"))
    by_size.to_csv(OUT / "first_detection_by_size.csv")
    by_cause = c.groupby("FIRE_CAUSE").agg(
        incidents=("event_key", "size"), seen_5km=("first_lag_5km", lambda s: s.notna().mean()),
        median_lag_5km=("first_lag_5km", "median"),
        median_size_ha=("size_ha", "median"))
    by_cause.to_csv(OUT / "first_detection_by_cause.csv")
    by_year = c.groupby("year").agg(incidents=("event_key", "size"),
                                    seen_5km=("first_lag_5km", lambda s: s.notna().mean()))
    by_year.to_csv(OUT / "first_detection_by_year.csv")
    lag_hist = c["first_lag_5km"].value_counts().sort_index().rename("incidents")
    lag_hist.to_csv(OUT / "first_detection_lag_hist.csv")
    hour_lag = c[c["first_lag_5km"].notna()].groupby(pd.cut(c["local_hour"], [-1, 5, 11, 13, 17, 23]), observed=False)[
        "first_lag_5km"].agg(["size", "median", lambda s: (s <= 0).mean()]).rename(columns={"<lambda_0>": "share_same_day"})
    hour_lag.to_csv(OUT / "first_detection_lag_by_local_hour.csv")

    cd = cell_days(firms)
    cd = onset_flags(cd)
    cd = attribute(cd, inc)
    cd.drop(columns=[c for c in cd.columns if c.startswith("gap_")]).to_csv(OUT / "firms_cell_days_annotated.csv.gz",
                                                                            index=False, compression="gzip")
    sens = []
    s = cd[cd["in_season"]]
    for k in (0, 1, 2, 3):
        for L in (3, 7, 14):
            col = f"onset_k{k}_L{L}"
            on = s[s[col]]
            sens.append(dict(ring_k=k, lookback_days=L, onsets=len(on), share_of_positive_cell_days=len(on) / len(s),
                             share_matching_bc_ignition=on["new_incident"].notna().mean()))
    pd.DataFrame(sens).to_csv(OUT / "onset_sensitivity.csv", index=False)

    onsets = classify_onsets(cd)
    oc = pd.crosstab(onsets["year"], onsets["onset_class"], margins=True)
    oc.to_csv(OUT / "onset_classes_by_year.csv")
    onsets.to_csv(OUT / "firms_onsets_k2_L7.csv", index=False)
    growth = onsets.groupby("onset_class").agg(onsets=("day", "size"), share_grows=("grows", "mean"),
                                               median_followup=("followup_cell_days_14d", "median"),
                                               median_frp_max=("frp_max", "median"))
    growth.to_csv(OUT / "onset_growth_by_class.csv")
    un = onsets[onsets["onset_class"] == "unexplained_firms_only"]
    un_month = pd.crosstab(un["month"], un["grows"]).rename(columns={False: "fades", True: "grows"})
    un_month.to_csv(OUT / "unexplained_onsets_by_month.csv")
    events, mask = fused_events(inc, fd, onsets)
    events.to_csv(OUT / "fused_ignition_events.csv", index=False)
    mask.to_csv(OUT / "fused_label_mask.csv", index=False)
    ev_tab = pd.crosstab(events["year"], events["tier"], margins=True)
    ev_tab.to_csv(OUT / "fused_events_by_year.csv")

    # Recall of BC ignitions by FIRMS onsets: does the onset flag find the BC starts it sees?
    seen = fd[fd["window_complete"] & fd["first_lag_5km"].notna()]
    matched_keys = set(onsets["new_incident"].dropna())
    seen_onset = seen["event_key"].isin(matched_keys)
    bc_onset = seen.assign(has_onset=seen_onset).groupby("size_bin", observed=False)["has_onset"].mean()
    bc_onset.to_csv(OUT / "bc_seen_ignitions_with_firms_onset.csv")

    tallies = label_tallies(cd, inc, fd)
    tallies.to_csv(OUT / "label_tallies.csv", index=False)
    rep, top, per = repetition_breakdown(cd)
    rep.to_csv(OUT / "repetition_breakdown.csv")
    top.to_csv(OUT / "top_incident_share.csv", index=False)
    per.groupby("year").head(10).merge(
        inc[["event_key", "FIRE_NUMBER", "INCIDENT_NAME", "size_ha", "FIRE_CAUSE"]],
        left_on="active_incident", right_on="event_key").drop(columns="event_key").to_csv(
        OUT / "top_incidents_by_cell_days.csv", index=False)
    plots(fd, onsets, tallies, rep)

    summary = dict(
        firms_detections=len(firms), firms_cell_days_in_season=int(cd["in_season"].sum()),
        incidents=len(inc), incidents_window_complete=int(fd["window_complete"].sum()),
        incidents_seen_5km=float(c["first_lag_5km"].notna().mean()),
        lag_median=float(c["first_lag_5km"].median()),
        onset_classes=onsets["onset_class"].value_counts().to_dict(),
        bc_seen_with_onset=float(seen_onset.mean()),
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    pd.set_option("display.width", 200, "display.max_columns", 30)
    for name, obj in [("completeness", comp), ("by_size", by_size), ("by_cause", by_cause), ("by_year", by_year),
                      ("lag_hist", lag_hist), ("hour_lag", hour_lag), ("onset_sensitivity", pd.DataFrame(sens)),
                      ("onset_classes", oc), ("onset_growth", growth), ("unexplained_by_month", un_month),
                      ("fused_events", ev_tab), ("mask_reasons", mask["reason"].value_counts()),
                      ("bc_seen_with_onset_by_size", bc_onset), ("tallies", tallies.T),
                      ("repetition", rep), ("top_share", top), ("summary", pd.Series(summary))]:
        print(f"\n=== {name}\n{obj}")


if __name__ == "__main__":
    main()
