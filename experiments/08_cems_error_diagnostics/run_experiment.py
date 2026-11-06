"""Paired CEMS ablation, activity-stratified evaluation and spatial error review."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parent
KEY = ["target_date", "h3_cell"]
TARGET = "target_recorded_detection"
FEATURE_SET = "cell_plus_neighbours"
SCORE = "gradient_boosting__" + FEATURE_SET
REGIONS = ["Cariboo", "Coastal", "Kamloops", "Northwest", "Prince George", "Southeast"]


def module(name, folder):
    spec = importlib.util.spec_from_file_location(name, EXP / folder / "run_experiment.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


base = module("baseline08", "03_exp_learned_baselines_bc")


def groups(frame):
    return np.select([frame.local_count_7d.gt(0), frame.neighbour_count_7d.gt(0)],
                     ["local_active", "neighbour_only"], default="quiet")


def checked_merge(left, right, on, validate):
    n = len(left)
    result = left.merge(right, on=on, how="left", validate=validate, indicator=True)
    if len(result) != n or not result.pop("_merge").eq("both").all():
        raise ValueError("Join changed row count or left unmatched keys")
    return result


def metrics(y, score, threshold):
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score)
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError("Scores must be finite probabilities")
    p = score >= threshold
    tp, fp, fn = int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum())
    return dict(rows=len(y), positives=int(y.sum()), prevalence=float(y.mean()) if len(y) else np.nan,
                ap=float(average_precision_score(y, score)) if y.any() else np.nan,
                precision=tp / (tp + fp) if tp + fp else 0.,
                recall=tp / (tp + fn) if tp + fn else np.nan,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                tp=tp, fp=fp, fn=fn, threshold=float(threshold))


def inventory(regions):
    rows = []
    for region in regions:
        for year in range(2018, 2024):
            p = base.feature_path(region, year)
            rows.append(dict(kind="features", region=region, path=str(p), exists=p.exists()))
        for name, folder in [("firms", "03_exp_learned_baselines_bc"),
                             ("cems", "03a_exp_learned_cems_bc")]:
            for relative in [f"predictions/{base.slug(region)}_2023.parquet", "metrics_by_year.csv"]:
                p = EXP / folder / "results" / relative
                rows.append(dict(kind=name, region=region, path=str(p), exists=p.exists()))
    for filename in ["dataset_CEMS_18_23.csv", "dataset_CEMS_18_24.csv"]:
        p = ROOT / "datasets" / filename
        rows.append(dict(kind="weather_alternative", region="", path=str(p), exists=p.exists()))
    return pd.DataFrame(rows)


def saved(region, frame, name):
    folder = "03_exp_learned_baselines_bc" if name == "firms" else "03a_exp_learned_cems_bc"
    directory = EXP / folder / "results"
    pred = pd.read_parquet(directory / "predictions" / f"{base.slug(region)}_2023.parquet",
                           columns=KEY + [TARGET, SCORE])
    if len(pred) != len(frame):
        raise ValueError("Prediction and feature row counts differ")
    joined = checked_merge(frame[KEY + [TARGET]], pred.rename(columns={TARGET: "saved_target"}), KEY, "one_to_one")
    if not joined[TARGET].eq(joined.saved_target).all():
        raise ValueError("Prediction labels differ from feature labels")
    table = pd.read_csv(directory / "metrics_by_year.csv")
    row = table.loc[table.region.eq(region) & table.year.eq(2022)
                    & table.model.eq("gradient_boosting") & table.feature_set.eq(FEATURE_SET)]
    if len(row) != 1:
        raise ValueError("Expected exactly one 2022 selection threshold")
    return joined[SCORE].to_numpy(), float(row.iloc[0].threshold)


def prepare_weather(regions, output):
    cems = module("cems08", "03a_exp_learned_cems_bc")
    # This loader needs only the complete weather grid, not the detection-only joined CSV.
    weather, grid, source = cems.load_cems_lookup()
    if weather.duplicated(["cems_cell_id", "history_end_date"]).any():
        raise ValueError("Duplicate CEMS grid/date keys")
    if not np.isfinite(weather[cems.CEMS].to_numpy()).all():
        raise ValueError("Nonfinite weather values")
    mappings = {r: cems.map_region_h3_to_cems(r, grid) for r in regions}
    import h3
    audit = []
    for region, mapping in mappings.items():
        coordinates = np.array([h3.cell_to_latlng(x) for x in mapping.h3_cell])
        lookup = grid.set_index("cems_cell_id").loc[mapping.cems_cell_id]
        lat1, lon1 = np.deg2rad(coordinates).T
        lat2, lon2 = np.deg2rad(lookup[["latitude", "longitude_deg"]].to_numpy()).T
        a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
        km = 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        audit.append(dict(region=region, cells=len(mapping), weather_cells=mapping.cems_cell_id.nunique(),
                          distance_km_median=np.median(km), distance_km_p95=np.quantile(km, .95),
                          distance_km_max=km.max()))
    pd.DataFrame(audit).to_csv(output / "weather_mapping.csv", index=False)
    weather.groupby(weather.history_end_date.dt.year)[cems.CEMS].agg(["min", "max", "std"]).to_csv(output / "weather_variation.csv")

    def load(region, year):
        frame = base.load(region, year, with_cells=True)
        if frame.duplicated(KEY).any():
            raise ValueError("Duplicate feature cell/date keys")
        frame = checked_merge(frame, mappings[region], ["h3_cell"], "many_to_one")
        frame = checked_merge(frame, weather.loc[weather.history_end_date.dt.year.eq(year)],
                              ["cems_cell_id", "history_end_date"], "many_to_one")
        if not np.isfinite(frame[cems.CEMS].to_numpy()).all():
            raise ValueError("Missing/nonfinite joined weather")
        return frame
    return cems, load, source


def fit_pair(regions, seed, output, cems, load):
    parts = []
    for region in regions:
        for year in range(2018, 2022):
            frame = load(region, year).sort_values(KEY).reset_index(drop=True)
            quiet = base.quiet_stratum(frame, FEATURE_SET)
            draw = np.random.default_rng([seed, year, REGIONS.index(region)]).random(len(frame)) < .01
            keep = ~quiet | frame[TARGET].eq(1).to_numpy() | draw
            sample = frame.loc[keep].copy()
            sample[KEY + [TARGET]].to_parquet(output / f"sample_{base.slug(region)}_{year}.parquet", index=False)
            parts.append(sample)
    train = pd.concat(parts, ignore_index=True)
    fitted = {}
    for name, helper in [("firms", base), ("cems", cems)]:
        X, _ = helper.design(train, FEATURE_SET)
        model = helper.gradient_boosting().set_params(random_state=seed)
        print(f"Fitting {name}: {len(train):,} identical sampled rows", flush=True)
        model.fit(X, train[TARGET].to_numpy())
        if np.max(np.abs(model.decision_function(X))) >= helper.MAX_ABS_LOG_ODDS:
            raise ValueError("Saturated training log-odds")
        import joblib
        joblib.dump(model, output / f"{name}.joblib")
        fitted[name] = model
    return fitted


def audit_maps(cases, regions, output):
    import h3
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    for region in regions:
        selected = cases.loc[cases.region.eq(region)]
        if selected.empty:
            continue
        history = base.load(region, 2023, with_cells=True)[KEY + [TARGET]]
        needed = set().union(*(h3.grid_disk(cell, 2) for cell in selected.h3_cell))
        history = history.loc[history.h3_cell.isin(needed)]
        for index, case in selected.iterrows():
            cells = set(h3.grid_disk(case.h3_cell, 2))
            local = history.loc[history.h3_cell.isin(cells)]
            fig, axes = plt.subplots(1, 5, figsize=(15, 3))
            for ax, offset in zip(axes, range(-2, 3)):
                date = case.target_date + pd.Timedelta(days=offset)
                day = local.loc[local.target_date.eq(date)].set_index("h3_cell")[TARGET]
                for cell in cells:
                    color = "#e76f51" if day.get(cell, 0) else "#eeeeee"
                    if cell not in day.index:
                        color = "#aab2bd"
                    polygon = [(lon, lat) for lat, lon in h3.cell_to_boundary(cell)]
                    ax.add_patch(Polygon(polygon, facecolor=color, edgecolor="black" if cell == case.h3_cell else "white",
                                         linewidth=2 if cell == case.h3_cell else .5))
                ax.autoscale_view()
                lat, _ = h3.cell_to_latlng(case.h3_cell)
                ax.set_aspect(1 / np.cos(np.deg2rad(lat)))
                ax.set_title(str(date.date()))
                ax.set_axis_off()
            fig.suptitle(f"{region} | {case.variant} {case.error} | {case.activity_group}\nOrange: recorded detection; grey: no detection; blue-grey: outside available table")
            fig.tight_layout()
            fig.savefig(output / f"case_{index:03d}.png", dpi=110)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["check", "saved", "paired"], default="check")
    parser.add_argument("--regions", nargs="+", choices=REGIONS, default=REGIONS)
    parser.add_argument("--variants", nargs="+", choices=["firms", "cems"], default=["firms", "cems"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-per-error", type=int, default=100)
    parser.add_argument("--maps", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    if args.audit_per_error < 0:
        parser.error("--audit-per-error must be nonnegative")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    inputs = inventory(args.regions)
    inputs.to_csv(output / "input_inventory.csv", index=False)
    if args.mode == "check":
        print(inputs.groupby(["kind", "exists"]).size().to_string())
        return
    if args.mode == "paired":
        cems, load, source = prepare_weather(args.regions, output)
        fitted = fit_pair(args.regions, args.seed, output, cems, load)
        variants = ["firms", "cems"]
    else:
        load, variants = lambda r, y: base.load(r, y, with_cells=True), args.variants
        required = inputs.loc[inputs.kind.isin(variants) & ~inputs.exists]
        if len(required):
            raise FileNotFoundError("Missing saved inputs; see input_inventory.csv:\n" + "\n".join(required.path))
    rows, candidates = [], []
    for region in args.regions:
        print(f"Evaluating {region}", flush=True)
        frame = load(region, 2023)
        frame["activity_group"] = groups(frame)
        for name in variants:
            if args.mode == "paired":
                helper = base if name == "firms" else cems
                validation = load(region, 2022)
                threshold = base.best_f1_threshold(validation[TARGET].to_numpy(), helper.predict(fitted[name], validation, FEATURE_SET))
                del validation
                score = helper.predict(fitted[name], frame, FEATURE_SET)
                predictions = frame[KEY + [TARGET]].copy()
                predictions["score"] = score
                predictions.to_parquet(output / f"predictions_{base.slug(region)}_{name}_2023.parquet", index=False)
            else:
                score, threshold = saved(region, frame, name)
            for group in ["all", "local_active", "neighbour_only", "quiet"]:
                mask = np.ones(len(frame), dtype=bool) if group == "all" else frame.activity_group.eq(group).to_numpy()
                rows.append(dict(region=region, variant=name, activity_group=group,
                                 **metrics(frame.loc[mask, TARGET], score[mask], threshold)))
            predicted = score >= threshold
            for error, mask in [("FP", predicted & frame[TARGET].eq(0).to_numpy()),
                                ("FN", ~predicted & frame[TARGET].eq(1).to_numpy())]:
                for group in ["local_active", "neighbour_only", "quiet"]:
                    eligible = np.flatnonzero(mask & frame.activity_group.eq(group).to_numpy())
                    chosen = np.random.default_rng(args.seed).choice(eligible, min(len(eligible), args.audit_per_error), replace=False)
                    part = frame.iloc[chosen][KEY + [TARGET, "activity_group"]].copy()
                    part["region"], part["variant"], part["error"] = region, name, error
                    part["score"], part["threshold"] = score[chosen], threshold
                    candidates.append(part)
    table = pd.DataFrame(rows)
    table.to_csv(output / "activity_metrics.csv", index=False)
    if set(variants) == {"firms", "cems"}:
        paired = table.pivot(index=["region", "activity_group"], columns="variant", values=["ap", "recall", "f1"])
        for metric in ["ap", "recall", "f1"]:
            paired[(metric, "cems_minus_firms")] = paired[(metric, "cems")] - paired[(metric, "firms")]
        paired.to_csv(output / "paired_deltas.csv")
    pool = pd.concat(candidates, ignore_index=True)
    # Round-robin allocation across model/region/activity strata: review sample, not prevalence estimate.
    pool = pool.sample(frac=1, random_state=args.seed)
    pool["rank"] = pool.groupby(["error", "variant", "region", "activity_group"]).cumcount()
    cases = pool.sort_values("rank", kind="stable").groupby("error", group_keys=False).head(args.audit_per_error).drop(columns="rank").reset_index(drop=True)
    cases["review_category"], cases["review_notes"] = "", ""
    cases.to_csv(output / "error_review.csv", index=False)
    if args.maps:
        audit_maps(cases, args.regions, output)
    manifest = dict(mode=args.mode, seed=args.seed, regions=args.regions, variants=variants,
                    training_years=[2018, 2019, 2020, 2021], selection_year=2022, evaluation_year=2023,
                    holdout_used=False, paired_training=args.mode == "paired",
                    warning="Saved mode does not establish identical training samples. Review maps are retrospective; no relabeling.",
                    code_sha256=base.digest(Path(__file__)))
    if args.mode == "paired":
        manifest["weather_source"] = str(source)
        manifest["weather_sha256"] = base.digest(source)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    images = "".join(f'<p>Case {i}: {row.region} {row.error}</p><img width="100%" src="case_{i:03d}.png">' for i, row in cases.iterrows()) if args.maps else ""
    (output / "REPORT.html").write_text('<!doctype html><meta charset="utf-8"><title>Experiment 08</title><h1>Activity and error diagnostics</h1><p>' + manifest["warning"] + '</p>' + table.to_html(index=False, float_format=lambda x: f"{x:.4f}") + images, encoding="utf-8")
    print(f"Wrote diagnostics to {output}")


if __name__ == "__main__":
    main()
