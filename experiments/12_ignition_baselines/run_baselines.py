"""Week 3, Task 3: non-learned baselines for next-day ignition (H3 res 6).

Fit on train years, select climatology settings on 2022 (valid), report 2022 and 2023.
Masked rows are dropped everywhere. Scores:
  seasonal              region x day-of-year rate (smoothed)
  climatology           spatially smoothed cell rate (shrunk to region) x seasonal profile
  climatology_refit     same settings refit on 2018-2022, scored on 2023 only
  wx_<index>            one fire-weather index alone (BCWS stations, IDW, previous day)
  human_<proxy>         WUI human-interface buffer coverage / -distance
  recent_ignitions      BC ignitions nearby in previous days (lightning-bust persistence)
  clim_x_fwi            climatology x observed/expected multiplier by FWI bin (fit on train)
  clim_x_fwi_x_recent   ... x observed/expected multiplier by recent-ignition bin
Metrics: average precision (pooled and mean of regions), ROC AUC, and recall under fixed
alert budgets (top-K cells per Fire Centre per day; ties broken randomly with a fixed seed).
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments/11_ignition_dataset"))
import build_dataset as bd  # noqa: E402

DATA = ROOT / "data/ignition_dataset/res6"
OUT = HERE / "results"
YEARS = list(range(2018, 2024))
TRAIN_SETS = {"2019-2021": [2019, 2020, 2021], "2018-2021": [2018, 2019, 2020, 2021]}
TARGETS = {"all": "y_ignition", "ge1ha": "y_ignition_ge1ha", "lightning": "y_n_lightning", "person": "y_n_person"}
BUDGETS = [5, 20, 100]
WX = ["fwi", "ffmc", "dmc", "dc", "isi", "bui", "temp_max", "rh_min", "precip_mm", "wind_max"]
WX_SIGN = {"rh_min": -1, "precip_mm": -1}
SEED = 0


def load():
    cols = ["h3_cell", "date", "region", "year", "masked", "f_doy", "f_bc_ign_ring1_1d", "f_bc_ign_ring1_7d",
            "f_bc_ign_ring2_7d"] + list(TARGETS.values())
    grid = pd.read_parquet(DATA / "grid.parquet")
    cell_index = {c: i for i, c in enumerate(grid["cell"])}
    human = pd.read_parquet(DATA / "static_human.parquet").set_index("h3_cell")
    parts = []
    for y in YEARS:
        d = pd.read_parquet(DATA / f"ignition_{y}.parquet", columns=cols)
        w = pd.read_parquet(DATA / f"weather_{y}.parquet")
        assert (w["h3_cell"].astype(str).to_numpy() == d["h3_cell"].astype(str).to_numpy()).all()
        assert (w["date"].to_numpy() == d["date"].to_numpy()).all()
        for v in WX:
            d[f"w_{v}"] = w[f"w_{v}"].to_numpy()
        d = d[~d["masked"]].drop(columns="masked")
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["cell_idx"] = d["h3_cell"].astype(str).map(cell_index).astype(np.int32)
    d["region"] = d["region"].astype(str)
    for t in ("lightning", "person"):
        d[TARGETS[t]] = (d[TARGETS[t]] > 0).astype(np.int8)
    d["s_hib_frac"] = d["h3_cell"].astype(str).map(human["s_hib_frac"]).to_numpy()
    d["s_hib_dist_km"] = d["h3_cell"].astype(str).map(human["s_hib_dist_km"]).to_numpy()
    d = d.drop(columns="h3_cell")
    return d, grid


# ------------------------------------------------------------------ climatology
def disks(grid, kmax):
    cells = grid["cell"].tolist()
    return [bd.ring_index(cells, k) for k in range(1, kmax + 1)]


def disk_sum(v, rings, k):
    out = v.astype(float).copy()
    padded = np.append(v.astype(float), 0.0)
    for r in rings[:k]:
        out += padded[r].sum(axis=1)
    return out


def fit_climatology(d, grid, rings, years, k, m, window):
    tr = d[d["year"].isin(years)]
    n = len(grid)
    counts = np.bincount(tr["cell_idx"], weights=tr["y_ignition"], minlength=n)
    expo = np.bincount(tr["cell_idx"], minlength=n).astype(float)
    regions = grid["region"].to_numpy()
    reg_rate = tr.groupby("region")["y_ignition"].mean()
    r0 = np.array([reg_rate[r] for r in regions])
    cd, ed = disk_sum(counts, rings, k), disk_sum(expo, rings, k)
    cell_rate = (cd + m) / (ed + m / r0)
    # Seasonal profile per region, smoothed over +/- window days, normalised to mean 1.
    doy = np.arange(1, 367)
    prof = {}
    for r, g in tr.groupby("region"):
        c = np.bincount(g["f_doy"], weights=g["y_ignition"], minlength=367)
        e = np.bincount(g["f_doy"], minlength=367).astype(float)
        ker = np.ones(2 * window + 1)
        cs, es = np.convolve(c, ker, "same"), np.convolve(e, ker, "same")
        rate = np.where(es > 0, cs / np.maximum(es, 1), 0)
        prof[r] = rate / (c.sum() / e.sum())
    return dict(cell_rate=cell_rate, prof=prof, doy=doy)


def score_climatology(fit, d, seasonal_only=False):
    regions = d["region"].to_numpy()
    s = np.empty(len(d))
    for r in np.unique(regions):
        m = regions == r
        s[m] = fit["prof"][r][d["f_doy"].to_numpy()[m]]
    if not seasonal_only:
        s = s * fit["cell_rate"][d["cell_idx"].to_numpy()]
    return s


# ------------------------------------------------------------------ observed / expected multipliers
def oe_bins(values, y, expected, edges):
    b = np.digitize(np.nan_to_num(values, nan=-1e9), edges)
    b[np.isnan(values)] = len(edges) + 1
    obs = np.bincount(b, weights=y, minlength=len(edges) + 2)
    exp = np.bincount(b, weights=expected, minlength=len(edges) + 2)
    mult = (obs + 1) / (exp + 1)          # +1 smoothing toward 1
    return mult


def apply_bins(values, edges, mult):
    b = np.digitize(np.nan_to_num(values, nan=-1e9), edges)
    b[np.isnan(values)] = len(edges) + 1
    return mult[b]


def recent_score(d):
    return (10 * d["f_bc_ign_ring1_1d"] + 3 * d["f_bc_ign_ring1_7d"] + d["f_bc_ign_ring2_7d"]).to_numpy(float)


# ------------------------------------------------------------------ metrics
def budget_recall(score, y, region, date, budgets, seed=SEED):
    rng = np.random.default_rng(seed)
    jitter = rng.random(len(score))
    key = pd.factorize(pd.Series(region).astype(str) + pd.Series(date).astype(str))[0]
    order = np.lexsort((-jitter, -np.nan_to_num(score, nan=-np.inf), key))
    k_sorted = key[order]
    start = np.r_[0, np.flatnonzero(np.diff(k_sorted)) + 1]
    rank = np.arange(len(order)) - np.repeat(start, np.diff(np.r_[start, len(order)]))
    ys = y[order]
    tot = y.sum()
    out = {}
    for b in budgets:
        sel = rank < b
        out[f"recall@{b}"] = ys[sel].sum() / tot if tot else np.nan
        out[f"precision@{b}"] = ys[sel].mean()
    return out


def metrics(score, d, target):
    y = d[target].to_numpy()
    s = np.nan_to_num(score, nan=np.nanmin(score) if np.isfinite(np.nanmin(score)) else 0)
    res = dict(positives=int(y.sum()), prevalence=y.mean(), ap=average_precision_score(y, s),
               auc=roc_auc_score(y, s))
    reg = []
    for r, idx in d.groupby("region").indices.items():
        if y[idx].sum():
            reg.append(average_precision_score(y[idx], s[idx]))
    res["ap_region_mean"] = float(np.mean(reg))
    res["ap_lift"] = res["ap"] / res["prevalence"]
    res.update(budget_recall(s, y, d["region"].to_numpy(), d["date"].to_numpy(), BUDGETS))
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d, grid = load()
    print(f"rows (unmasked): {len(d):,}", flush=True)
    rings = disks(grid, 4)
    valid, test = d[d["year"] == 2022], d[d["year"] == 2023]

    # 1. Tune climatology on 2022.
    rows = []
    for (tname, years), k, m, w in itertools.product(TRAIN_SETS.items(), [0, 1, 2, 3, 4], [0.5, 2, 8], [7, 15, 30]):
        fit = fit_climatology(d, grid, rings, years, k, m, w)
        s = score_climatology(fit, valid)
        rows.append(dict(train=tname, k=k, m=m, window=w, ap=average_precision_score(valid["y_ignition"], s)))
    tune = pd.DataFrame(rows).sort_values("ap", ascending=False)
    tune.to_csv(OUT / "climatology_tuning_2022.csv", index=False)
    best = tune.iloc[0]
    years = TRAIN_SETS[best["train"]]
    print("best climatology:", best.to_dict(), flush=True)
    fit = fit_climatology(d, grid, rings, years, int(best["k"]), best["m"], int(best["window"]))
    fit_refit = fit_climatology(d, grid, rings, [y for y in range(2018, 2023)], int(best["k"]), best["m"], int(best["window"]))

    # 2. Observed/expected multipliers fit on train years.
    tr = d[d["year"].isin(years)]
    clim_tr = score_climatology(fit, tr)
    exp_scale = tr["y_ignition"].sum() / clim_tr.sum()
    fwi_edges = np.nanquantile(tr["w_fwi"], np.linspace(0.1, 0.9, 9))
    fwi_mult = oe_bins(tr["w_fwi"].to_numpy(), tr["y_ignition"].to_numpy(), clim_tr * exp_scale, fwi_edges)
    e2 = clim_tr * exp_scale * apply_bins(tr["w_fwi"].to_numpy(), fwi_edges, fwi_mult)
    rec_edges = np.array([0.5, 1.5, 3.5, 6.5, 12.5, 25.5])
    rec_mult = oe_bins(recent_score(tr), tr["y_ignition"].to_numpy(), e2, rec_edges)
    pd.DataFrame({"fwi_bin_upper": list(fwi_edges) + [np.inf, "missing"], "fwi_multiplier": fwi_mult}).to_csv(
        OUT / "fwi_multipliers.csv", index=False)
    pd.DataFrame({"recent_bin_upper": list(rec_edges) + [np.inf, "missing"], "recent_multiplier": rec_mult}).to_csv(
        OUT / "recent_multipliers.csv", index=False)

    def scores(part, year):
        clim = score_climatology(fit, part)
        fwi_m = apply_bins(part["w_fwi"].to_numpy(), fwi_edges, fwi_mult)
        s = {"seasonal": score_climatology(fit, part, seasonal_only=True), "climatology": clim}
        for v in WX:
            s[f"wx_{v}"] = WX_SIGN.get(v, 1) * part[f"w_{v}"].to_numpy(float)
        s["human_hib_frac"] = part["s_hib_frac"].to_numpy(float)
        s["human_neg_dist"] = -part["s_hib_dist_km"].to_numpy(float)
        s["recent_ignitions"] = recent_score(part)
        s["clim_x_fwi"] = clim * fwi_m
        s["clim_x_fwi_x_recent"] = clim * fwi_m * apply_bins(recent_score(part), rec_edges, rec_mult)
        if year == 2023:
            s["climatology_refit"] = score_climatology(fit_refit, part)
            s["clim_refit_x_fwi_x_recent"] = s["climatology_refit"] * fwi_m * apply_bins(recent_score(part), rec_edges, rec_mult)
        return s

    res, region_rows = [], []
    for year, part in [(2022, valid), (2023, test)]:
        for name, s in scores(part, year).items():
            for tname, tcol in TARGETS.items():
                if tname != "all" and not (name in ("climatology", "clim_x_fwi_x_recent", "human_hib_frac",
                                                    "human_neg_dist", "wx_fwi", "recent_ignitions", "seasonal")):
                    continue
                r = metrics(s, part, tcol)
                r.update(year=year, baseline=name, target=tname)
                res.append(r)
            if name in ("climatology", "clim_x_fwi_x_recent", "wx_fwi", "human_hib_frac"):
                y = part["y_ignition"].to_numpy()
                for reg, idx in part.groupby("region").indices.items():
                    region_rows.append(dict(year=year, baseline=name, region=reg, positives=int(y[idx].sum()),
                                            prevalence=y[idx].mean(),
                                            ap=average_precision_score(y[idx], np.nan_to_num(s[idx], nan=-1e9))))
        print(f"{year} done", flush=True)
    m = pd.DataFrame(res)
    m.to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(region_rows).to_csv(OUT / "metrics_by_region.csv", index=False)
    (OUT / "selected.json").write_text(json.dumps(dict(
        climatology=best.to_dict(), train_years=years, fwi_edges=list(fwi_edges), recent_edges=list(rec_edges),
        budgets_cells_per_centre_per_day=BUDGETS, rows_unmasked=len(d)), indent=2, default=float))
    plots(m)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    show = m[m["target"] == "all"][["year", "baseline", "ap", "ap_region_mean", "auc", "ap_lift",
                                    "recall@5", "recall@20", "recall@100"]]
    print(show.sort_values(["year", "ap"], ascending=[True, False]).to_string(index=False))
    print(m[m["target"] != "all"][["year", "baseline", "target", "positives", "ap", "ap_lift", "auc", "recall@20"]]
          .sort_values(["year", "target", "ap"], ascending=[True, True, False]).to_string(index=False))


def plots(m):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    a = m[m["target"] == "all"]
    order = a[a["year"] == 2022].sort_values("ap")["baseline"].tolist()
    order += [b for b in a["baseline"].unique() if b not in order]
    for ax, metric in zip(axes, ["ap", "recall@20"]):
        p = a.pivot(index="baseline", columns="year", values=metric).reindex(order)
        p.plot(kind="barh", ax=ax, color=["#74a9cf", "#0570b0"])
        ax.set_title({"ap": "Average precision (pooled)", "recall@20": "Recall with 20 alerts / centre / day"}[metric])
    fig.tight_layout()
    fig.savefig(OUT / "baselines.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
