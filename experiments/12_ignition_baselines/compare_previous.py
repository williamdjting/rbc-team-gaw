"""Score Experiments 02-07 (FIRMS next-day hotspot models) on the new ignition target, 2023.

Old res-7 predictions are aggregated to res-6 cells by the maximum over child cells (and over
regional grids for border cells), then evaluated on exactly the unmasked 2023 rows used for the
Experiment 12 baselines, with the same metrics. Persistence (Experiment 02) is represented by
the FIRMS history features in the dataset (yesterday's detections in the cell, then its ring).
Targets: all ignitions, satellite-visible ignitions (tier A), >= 1 ha, lightning, person-caused.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h3
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
import run_baselines as rb  # noqa: E402

OUT = HERE / "results"
REGIONS = ["cariboo", "coastal", "kamloops", "northwest", "prince_george", "southeast"]
OLD = {
    "03 logistic + neighbours": ("03", "logistic_regression__cell_plus_neighbours"),
    "03 boosting + neighbours": ("03", "gradient_boosting__cell_plus_neighbours"),
    "04 GNN": ("04", "gnn"),
    "04 MLP": ("04", "mlp"),
    "05 CatBoost full data + neighbours": ("05", "catboost_unweighted__cell_plus_neighbours"),
    "06 CatBoost sampled + neighbours": ("06", None),
    "07 TabPFN hybrid + neighbours": ("07", "tabpfn_hybrid__cell_plus_neighbours"),
}
FILES = {"05": EXP / "05_catboost_full_data/results/catboost_predictions_2023.parquet",
         "06": EXP / "06_catboost_undersampled/results/catboost_predictions_2023.parquet",
         "07": EXP / "07_tabpfn/results/tabpfn_predictions_2023.parquet"}
TARGETS = {"all": "y_ignition", "satellite_visible_tierA": "y_tier_a", "ge1ha": "y_ignition_ge1ha",
           "lightning": "y_n_lightning", "person": "y_n_person"}
BUDGETS = [5, 20, 100]


def to_res6(df, col, parent):
    s = pd.DataFrame({"cell": df["h3_cell"].map(parent).to_numpy(), "date": df["target_date"].to_numpy(),
                      "score": df[col].to_numpy(np.float32)})
    return s.groupby(["cell", "date"], sort=False)["score"].max()


def load_old(parent):
    out = {}
    for name, (exp, col) in OLD.items():
        if exp in ("03", "04"):
            folder = EXP / ("03_exp_learned_baselines_bc" if exp == "03" else "04_exp_gnn_bc") / "results/predictions"
            parts = [to_res6(pd.read_parquet(folder / f"{r}_2023.parquet", columns=["target_date", "h3_cell", col]),
                             col, parent) for r in REGIONS]
            s = pd.concat(parts).groupby(level=[0, 1]).max()
        else:
            if col is None:
                import pyarrow.parquet as pq
                names = pq.ParquetFile(FILES[exp]).schema.names
                col = next(c for c in names if c.startswith("catboost") and c.endswith("cell_plus_neighbours"))
                OLD[name] = (exp, col)
            df = pd.read_parquet(FILES[exp], columns=["target_date", "h3_cell", col])
            s = to_res6(df, col, parent).groupby(level=[0, 1]).max()
        out[name] = s
        print(f"loaded {name} ({col}): {len(s):,} res-6 cell-days", flush=True)
    return out


def fast_metrics(score, part, target):
    y = (part[target].to_numpy() > 0).astype(np.int8)
    s = np.nan_to_num(score, nan=-1e9)
    res = dict(positives=int(y.sum()), prevalence=y.mean(), ap=average_precision_score(y, s), auc=roc_auc_score(y, s))
    res["lift"] = res["ap"] / res["prevalence"]
    rng = np.random.default_rng(rb.SEED)
    key = part["region_code"].to_numpy() * 1000 + part["f_doy"].to_numpy()
    order = np.lexsort((rng.random(len(s)), -s, key))
    ks = key[order]
    start = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
    rank = np.arange(len(order)) - np.repeat(start, np.diff(np.r_[start, len(order)]))
    for b in BUDGETS:
        res[f"recall@{b}"] = y[order][rank < b].sum() / y.sum()
    return res


def main():
    d, grid = rb.load()
    sel = json.loads((OUT / "selected.json").read_text())
    c = sel["climatology"]
    rings = rb.disks(grid, 4)
    years = rb.TRAIN_SETS[c["train"]]
    fit = rb.fit_climatology(d, grid, rings, years, int(c["k"]), c["m"], int(c["window"]))
    tr = d[d["year"].isin(years)]
    clim_tr = rb.score_climatology(fit, tr)
    scale = tr["y_ignition"].sum() / clim_tr.sum()
    fwi_edges = np.array(sel["fwi_edges"])
    fwi_mult = rb.oe_bins(tr["w_fwi"].to_numpy(), tr["y_ignition"].to_numpy(), clim_tr * scale, fwi_edges)
    e2 = clim_tr * scale * rb.apply_bins(tr["w_fwi"].to_numpy(), fwi_edges, fwi_mult)
    rec_edges = np.array(sel["recent_edges"])
    rec_mult = rb.oe_bins(rb.recent_score(tr), tr["y_ignition"].to_numpy(), e2, rec_edges)

    part = d[d["year"] == 2023].copy()
    del d, tr
    tier = pd.read_parquet(rb.DATA / "ignition_2023.parquet", columns=["masked", "info_n_tier_a", "f_firms_cell_1d",
                                                                         "f_firms_cell_7d", "f_firms_ring1_1d",
                                                                         "f_firms_ring1_7d"])
    tier = tier[~tier["masked"]].reset_index(drop=True)
    part = part.reset_index(drop=True)
    part["y_tier_a"] = tier["info_n_tier_a"].to_numpy()
    part["region_code"] = pd.factorize(part["region"])[0]
    cells = grid["cell"].to_numpy()
    part["cell"] = cells[part["cell_idx"].to_numpy()]

    clim = rb.score_climatology(fit, part)
    scores = {
        "NEW clim x FWI x recent ignitions": clim * rb.apply_bins(part["w_fwi"].to_numpy(), fwi_edges, fwi_mult)
        * rb.apply_bins(rb.recent_score(part), rec_edges, rec_mult),
        "NEW climatology": clim,
        "NEW human access (HIB share)": part["s_hib_frac"].to_numpy(float),
        "02 persistence (FIRMS history)": (1000 * tier["f_firms_cell_1d"] + 100 * tier["f_firms_ring1_1d"]
                                          + 10 * tier["f_firms_cell_7d"] + tier["f_firms_ring1_7d"]).to_numpy(float),
    }
    parent = {}
    for r in REGIONS:
        g = pd.read_parquet(EXP / f"02_hotspot_persistence_bc/regions/{r}/data/h3_grid.parquet", columns=["h3_cell"])
        parent.update({c7: h3.cell_to_parent(c7, 6) for c7 in g["h3_cell"]})
    old = load_old(parent)
    idx = pd.MultiIndex.from_arrays([part["cell"].to_numpy(), part["date"].to_numpy()])
    for name, s in old.items():
        scores[name] = s.reindex(idx).to_numpy(float)
        print(f"{name}: {np.isnan(scores[name]).mean():.2%} rows without an old score", flush=True)

    rows = []
    for name, s in scores.items():
        for tname, tcol in TARGETS.items():
            r = fast_metrics(s, part, tcol)
            r.update(model=name, target=tname)
            rows.append(r)
    m = pd.DataFrame(rows)
    m.to_csv(OUT / "compare_previous_2023.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    for t in TARGETS:
        print(f"\n=== target: {t}")
        print(m[m["target"] == t].sort_values("ap", ascending=False)[
            ["model", "positives", "ap", "lift", "auc", "recall@5", "recall@20", "recall@100"]].to_string(index=False))


if __name__ == "__main__":
    main()
