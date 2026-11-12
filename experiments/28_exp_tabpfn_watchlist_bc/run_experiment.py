"""Experiment 28: TabPFN as a quiet-cell watch-list ranking (all six Fire Centres).

Experiment 07 used TabPFN only on active cells; quiet cells were left to CatBoost because
TabPFN is slow. Here TabPFN ranks every quiet cell, with the inputs of Experiment 25
(S-NPP history + CanCPLD lightning). On quiet cells 13 of those 21 inputs are zero by
definition, so the informative inputs are days since detection, the history-censored flag
and six lightning features. Many quiet cell-days share identical inputs, so each year's
unique rows are scored once (about 0.2-0.3M per year instead of 34M).

- Context: 2018-2021 quiet rows from Experiment 25's training samples: 1,000 quiet positives
  and 2,000 quiet negatives (balanced so positives are visible; scores are used only to rank).
- Model: TabPFN v2 weights (TabPFN package 9.2.0, as Experiment 07). Newer weights are
  licence-gated and need an interactive login. n_estimators = 1, as Experiment 07. CPU.
- Scoring: Experiment 27's watch list, metrics and week-block bootstrap; compared with
  Experiment 25/26 MLPs (same information), Experiment 18's simple ranking and, for 2023,
  Experiment 13's CatBoost.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
ROOT = EXPERIMENTS.parent
spec = importlib.util.spec_from_file_location("experiment_27", EXPERIMENTS / "27_exp_quiet_watchlist_bc/run_experiment.py")
e27 = sys.modules["experiment_27"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e27)

RESULTS = HERE / "results"
SCORES = RESULTS / "scores"
TRAINING = ROOT / "data/snpp_lightning_h3r7/training_samples"
FEATURES = ["days_since_detection", "history_left_censored", "ltg_cell_1d", "ltg_3x3_1d", "ltg_3x3_3d",
            "ltg_9x9_1d", "ltg_9x9_3d", "ltg_days_since_3x3"]
CONTEXT_POSITIVES, CONTEXT_NEGATIVES, SEED = 1000, 2000, 28
BATCH = 20_000
MODELS_COMPARED = ["tabpfn", "mlp_lightning", "mlp_no_lightning", "catboost_13", "nearby_activity", "burned_before"]


def context():
    frames = [pd.read_parquet(f, columns=FEATURES + ["quiet_stratum", "target_snpp_filled"])
              for f in sorted(TRAINING.glob("*/snpp_lightning_*.parquet"))]
    d = pd.concat(frames, ignore_index=True)
    d = d.loc[d.quiet_stratum > 0]
    pos, neg = d.loc[d.target_snpp_filled], d.loc[~d.target_snpp_filled]
    ctx = pd.concat([pos.sample(CONTEXT_POSITIVES, random_state=SEED), neg.sample(CONTEXT_NEGATIVES, random_state=SEED)])
    stats = {"quiet_rows_available": int(len(d)), "quiet_positives_available": int(len(pos)),
             "context_positives": CONTEXT_POSITIVES, "context_negatives": CONTEXT_NEGATIVES}
    return ctx[FEATURES].to_numpy(np.float32), ctx.target_snpp_filled.to_numpy().astype(int), stats


def score_years(model):
    """Scores every quiet cell-day once per unique input row; saves cell-major per-centre arrays."""
    timing = []
    for year in e27.YEARS:
        paths = {r: e27.EXP25 / str(year) / f"{e27.slug(r)}_snpp_lightning_full_grid.parquet" for r in e27.REGIONS}
        if all((SCORES / f"tabpfn_{e27.slug(r)}_{year}.npy").exists() for r in e27.REGIONS):
            continue
        tables = {r: pq.read_table(p, columns=FEATURES + ["quiet_stratum"]).to_pandas() for r, p in paths.items()}
        quiet_rows = pd.concat([t.loc[t.quiet_stratum > 0, FEATURES] for t in tables.values()], ignore_index=True)
        unique = quiet_rows.drop_duplicates().reset_index(drop=True)
        cache = SCORES / f"unique_scores_{year}.parquet"
        if cache.exists():
            lookup = pd.read_parquet(cache)
            assert len(lookup) == len(unique)
        else:
            tic = time.time()
            probs = np.concatenate([model.predict_proba(unique.iloc[s:s + BATCH].to_numpy(np.float32))[:, 1]
                                    for s in range(0, len(unique), BATCH)]).astype(np.float32)
            seconds = time.time() - tic
            timing.append({"year": year, "quiet_cell_days": int(len(quiet_rows)), "unique_rows": int(len(unique)),
                           "seconds": round(seconds), "rows_per_second": round(len(unique) / seconds)})
            print(f"{year}: scored {len(unique):,} unique rows for {len(quiet_rows):,} quiet cell-days in {seconds / 60:.1f} min",
                  flush=True)
            lookup = unique.assign(score=probs)
            lookup.to_parquet(cache, index=False)
        for region, t in tables.items():
            merged = t.merge(lookup, on=FEATURES, how="left")
            assert len(merged) == len(t)
            score = merged.score.to_numpy(np.float32).copy()
            score[t.quiet_stratum.to_numpy() == 0] = np.nan  # only quiet cells are ranked
            assert np.isfinite(score[t.quiet_stratum.to_numpy() > 0]).all()
            np.save(SCORES / f"tabpfn_{e27.slug(region)}_{year}.npy", score)
    return timing


def evaluate():
    rows, onset_rows, daily = [], [], {}
    detections = e27.detection_arrays()
    for region_index, region in enumerate(e27.REGIONS):
        for year in e27.YEARS:
            cells, quiet, truth, scores = e27.load_exp25(region, year)
            tab = np.load(SCORES / f"tabpfn_{e27.slug(region)}_{year}.npy").reshape(len(cells), e27.TARGET_DAYS)
            scores["tabpfn"] = np.nan_to_num(tab, nan=-1.0)
            if year == 2023:
                e13 = e27.load_exp13(region, cells)
                truth = e13.pop("truth") > 0
                scores["catboost_13"] = e13["catboost_13"]
            scores.update(e27.simple_rankings(list(cells), detections, year))
            quiet_fire = truth & quiet
            future = e27.forward_window(truth.astype(np.float32), e27.WATCH_DAYS) > 0
            tiebreak = np.random.default_rng([e27.SEED, year, region_index]).random(truth.shape)  # as Experiment 27
            if year == 2023:
                o_row, o_col, o_group = e27.onset_groups(region, cells, year)
                keep = quiet_fire[o_row, o_col]
                o_row, o_col, o_group = o_row[keep], o_col[keep], o_group[keep]
            for name in MODELS_COMPARED:
                if name not in scores:
                    continue
                for n, listed in e27.ranked_lists(scores[name], quiet, tiebreak).items():
                    recent = e27.window(listed.astype(np.float32), e27.WATCH_DAYS) > 0
                    caught7 = recent & quiet_fire
                    rows.append({"region": region, "year": year, "model": name, "list_size": n,
                                 "quiet_fires": int(quiet_fire.sum()), "listed_cell_days": int(listed.sum()),
                                 "caught_next_day": float((listed & quiet_fire).sum() / max(quiet_fire.sum(), 1)),
                                 "caught_within_7d": float(caught7.sum() / max(quiet_fire.sum(), 1)),
                                 "hit_rate_next_day": float((listed & truth).sum() / max(listed.sum(), 1)),
                                 "hit_rate_within_7d": float((listed & future).sum() / max(listed.sum(), 1))})
                    daily[region, year, name, n] = np.stack([quiet_fire.sum(axis=0), caught7.sum(axis=0)], axis=1)
                    if year == 2023:
                        caught = recent[o_row, o_col]
                        for group in np.unique(o_group):
                            mask = o_group == group
                            onset_rows.append({"region": region, "model": name, "list_size": n, "onset_group": group,
                                               "quiet_onset_cells": int(mask.sum()), "caught_within_7d": int(caught[mask].sum())})
            top = {r["model"]: r["caught_within_7d"] for r in rows
                   if r["region"] == region and r["year"] == year and r["list_size"] == 50}
            print(f"{region} {year} top-50 caught in 7d: " + "; ".join(f"{k} {v:.3f}" for k, v in top.items()), flush=True)
    return pd.DataFrame(rows), pd.DataFrame(onset_rows), daily


def summarise(metrics, daily):
    m = metrics.assign(caught=metrics.caught_within_7d * metrics.quiet_fires, caught1=metrics.caught_next_day * metrics.quiet_fires,
                       hits=metrics.hit_rate_next_day * metrics.listed_cell_days)
    pooled = m.groupby(["year", "model", "list_size"])[["quiet_fires", "caught", "caught1", "hits", "listed_cell_days"]].sum()
    pooled["caught_within_7d"] = pooled.caught / pooled.quiet_fires
    pooled["caught_next_day"] = pooled.caught1 / pooled.quiet_fires
    pooled["hit_rate_next_day"] = pooled.hits / pooled.listed_cell_days
    pooled = pooled.reset_index()
    wins = []
    for r in pooled.itertuples():
        per = metrics.loc[metrics.year.eq(r.year) & metrics.list_size.eq(r.list_size)].set_index(["region", "model"]).caught_within_7d
        wins.append({b: int(sum(per[g, r.model] > per[g, b] for g in e27.REGIONS)) if (e27.REGIONS[0], b) in per.index else None
                     for b in ("mlp_lightning", "nearby_activity", "catboost_13")})
    for b in ("mlp_lightning", "nearby_activity", "catboost_13"):
        pooled[f"centres_beating_{b}"] = [w[b] for w in wins]
    pooled = pooled.drop(columns=["caught", "caught1", "hits"])

    comparisons = [("tabpfn", "mlp_lightning"), ("tabpfn", "mlp_no_lightning"), ("tabpfn", "nearby_activity"),
                   ("tabpfn", "catboost_13")]
    rng = np.random.default_rng(SEED)
    boot = []
    for year in e27.YEARS:
        dates = pd.date_range(f"{year}-05-08", f"{year}-09-30")
        week = dates.isocalendar().week.to_numpy()
        weeks = np.unique(week)
        members = [np.flatnonzero(week == w) for w in weeks]
        draws = [np.concatenate([members[i] for i in rng.integers(0, len(weeks), len(weeks))]) for _ in range(e27.BOOTSTRAP)]
        for n in e27.SIZES:
            for a, b in comparisons:
                if (e27.REGIONS[0], year, b, n) not in daily:
                    continue
                da = sum(daily[g, year, a, n] for g in e27.REGIONS)
                db = sum(daily[g, year, b, n] for g in e27.REGIONS)
                diff = lambda idx: da[idx, 1].sum() / da[idx, 0].sum() - db[idx, 1].sum() / db[idx, 0].sum()
                values = np.array([diff(d) for d in draws])
                boot.append({"year": year, "list_size": n, "model": a, "baseline": b,
                             "difference": float(diff(np.arange(len(dates)))), "ci_low": float(np.quantile(values, .025)),
                             "ci_high": float(np.quantile(values, .975)), "share_above_zero": float((values > 0).mean())})
    return pooled, pd.DataFrame(boot)


def main():
    started = time.time()
    SCORES.mkdir(parents=True, exist_ok=True)
    from tabpfn import TabPFNClassifier, __version__ as tabpfn_version
    from tabpfn.constants import ModelVersion
    X, y, stats = context()
    model = TabPFNClassifier.create_default_for_version(ModelVersion.V2, n_estimators=1, device="cpu",
                                                        random_state=SEED, ignore_pretraining_limits=True)
    model.fit(X, y)
    assert list(model.classes_) == [0, 1]
    timing = score_years(model)
    metrics, onsets, daily = evaluate()
    metrics.to_csv(RESULTS / "watch_metrics.csv", index=False)
    onsets.to_csv(RESULTS / "watch_onsets_2023.csv", index=False)
    pooled, boot = summarise(metrics, daily)
    pooled.to_csv(RESULTS / "watch_summary_bc.csv", index=False)
    boot.to_csv(RESULTS / "bootstrap_bc.csv", index=False)
    (RESULTS / "manifest.json").write_text(json.dumps({
        "tabpfn_package": tabpfn_version, "weights": "TabPFN v2 (ungated); newer weights need licence login",
        "n_estimators": 1, "device": "cpu", "features": FEATURES, "context": stats, "seed": SEED,
        "scoring": "every quiet cell-day, once per unique input row", "timing": timing,
        "watch_list": "Experiment 27 definitions, tie-break seeds and bootstrap", "runtime_minutes": round((time.time() - started) / 60, 1)},
        indent=2))
    print(f"Done in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
