"""Build a standalone visual HTML report from saved Experiment 13 outputs."""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
EXPERIMENTS = HERE.parent
EXPERIMENT_03 = EXPERIMENTS / "03_exp_learned_baselines_bc"
spec = importlib.util.spec_from_file_location("experiment_03_report", EXPERIMENT_03 / "build_report.py")
report_03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report_03)
embedded_image, html_table, style = report_03.embedded_image, report_03.html_table, report_03.style
RULE, INK, MUTED = report_03.RULE, report_03.INK, report_03.MUTED
# Reference categorical slots 1-3 in fixed order; marker shape repeats the identity.
SERIES = {("gradient_boosting", "base"): ("Base features (= Experiment 03 boosting)", "#2a78d6", "o"),
          ("gradient_boosting", "all"): ("Gradient boosting, all fire-context features", "#eb6834", "^"),
          ("catboost", "all"): ("CatBoost, all fire-context features", "#1baf7a", "s")}
SET_NAMES = {"base": "Base (03 cell + neighbours)", "base+frp_daynight": "+ FRP and day/night",
             "base+wider_area": "+ wider area", "base+fire_complex": "+ fire complex and duration",
             "base+bcws": "+ BCWS incidents", "firms_context": "+ all FIRMS groups (no BCWS)",
             "all": "+ all four groups"}
GROUP_TEXT = {
    "frp_daynight": ("FRP and day/night", "Fire radiative power yesterday and over 7 days, night detections, for the cell and its ring-1 neighbours", "FIRMS detections (already downloaded)"),
    "wider_area": ("Wider area", "Detections 2 and 3 rings out (about 5–8 km), grid distance to the nearest recent detection, active cells within 5 rings", "FIRMS detections"),
    "fire_complex": ("Fire complex and duration", "Size and 3-day FRP of the connected burning complex within 2 rings; days with detections in the past week", "FIRMS detections"),
    "bcws": ("BCWS incidents", "Distance to and number of known active fires, age and lightning cause of the nearest, recently declared-out fires, new fires nearby", "BC Wildfire Service incident points (DataBC)"),
}


def ensure_figures():
    FIGURES.mkdir(exist_ok=True)


def f1_chart(table, order):
    fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    positions = np.arange(len(order))
    ax.scatter(table.loc[order, "rule_f1"], positions, marker="D", s=60, color=RULE, edgecolor="white", linewidth=1.5, zorder=3)
    for key, offset in zip(SERIES, (-.18, 0, .18)):
        name, colour, marker = SERIES[key]
        ax.scatter(table.loc[order, f"{key[0]}__{key[1]}_f1"], positions + offset, s=58, marker=marker, color=colour,
                   edgecolor="white", linewidth=1.5, zorder=4)
    handles = [Line2D([], [], linestyle="none", marker="D", markersize=7, markerfacecolor=RULE, markeredgecolor="white",
                      label="Previous-day rule (Experiment 02)")]
    handles += [Line2D([], [], linestyle="none", marker=m, markersize=8, markerfacecolor=c, markeredgecolor="white", label=n)
                for n, c, m in SERIES.values()]
    values = table.loc[order, [c for c in table.columns if c.endswith("_f1")]]
    ax.set(yticks=positions, yticklabels=order, xlim=(values.min().min() - .02, values.max().max() + .02),
           xlabel="2023 F1 at the 2022 threshold", title="2023 F1 by Fire Centre")
    ax.invert_yaxis()
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    style(ax)
    fig.savefig(FIGURES / "report_f1.png", dpi=160)
    plt.close(fig)


def ablation_chart(ablation, regions):
    sets = [s for s in SET_NAMES if s != "base"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, constrained_layout=True)
    for ax, metric, title in zip(axes, ("average_precision", "f1"), ("Average precision", "F1 at 2022 threshold")):
        for row, feature_set in enumerate(sets):
            delta = ablation.loc[regions, f"{metric}__{feature_set}"] - ablation.loc[regions, f"{metric}__base"]
            ax.scatter(delta, np.full(len(delta), row), s=26, color="#c3c2b7", edgecolor="white", linewidth=.8, zorder=3)
            ax.scatter(delta.median(), row, s=80, color="#2a78d6", edgecolor="white", linewidth=1.5, zorder=4)
            ax.annotate(f"{delta.median():+.3f}", (delta.median(), row), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=9, color=INK)
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.set(title=f"{title}: change vs base", xlabel="2023 change (each gray dot is one Fire Centre)")
        style(ax)
    axes[0].set(yticks=range(len(sets)), yticklabels=[SET_NAMES[s] for s in sets])
    axes[0].invert_yaxis()
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=8, markerfacecolor="#2a78d6",
                      markeredgecolor="white", label="Median across the six centres"),
               Line2D([], [], linestyle="none", marker="o", markersize=6, markerfacecolor="#c3c2b7",
                      markeredgecolor="white", label="One Fire Centre")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    fig.savefig(FIGURES / "report_ablation.png", dpi=160)
    plt.close(fig)


def pr_curves(curves, rule_points, order):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.2), sharex=True, sharey=True, constrained_layout=True)
    for ax, region in zip(axes.flat, order):
        for key, (name, colour, _) in SERIES.items():
            curve = curves.loc[curves.region.eq(region) & curves.model.eq(key[0]) & curves.feature_set.eq(key[1])]
            ax.plot(curve.recall, curve.precision, color=colour, linewidth=1.3 if key[1] == "base" else 2,
                    zorder=2 if key[1] == "base" else 3)
        rule = rule_points.loc[region]
        ax.scatter(rule.recall, rule.precision, s=46, marker="D", color=RULE, edgecolor="white", linewidth=1, zorder=4)
        ax.set(title=region, xlim=(0, 1), ylim=(0, 1))
        style(ax)
    for ax in axes[-1]:
        ax.set_xlabel("Recall")
    for ax in axes[:, 0]:
        ax.set_ylabel("Precision")
    handles = [Line2D([], [], color=c, linewidth=1.3 if k[1] == "base" else 2, label=n) for k, (n, c, _) in SERIES.items()]
    handles.append(Line2D([], [], linestyle="none", marker="D", markersize=6, markerfacecolor=RULE, markeredgecolor="white",
                          label="Previous-day rule (Experiment 02)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("2023 precision-recall curves by Fire Centre")
    fig.savefig(FIGURES / "report_pr_curves.png", dpi=150)
    plt.close(fig)


def importance_chart(importance, groups):
    top = importance.sort_values("importance", ascending=False).head(15).iloc[::-1]
    group_of = {c: g for g, columns in groups.items() for c in columns}
    fig, ax = plt.subplots(figsize=(10, 5.4), constrained_layout=True)
    ax.barh(np.arange(len(top)), top.importance, color="#2a78d6", height=.62)
    labels = [f"{f}  ({GROUP_TEXT[group_of[f]][0]})" if f in group_of else f"{f}  (base)" for f in top.feature]
    ax.set(yticks=np.arange(len(top)), yticklabels=labels, xlabel="CatBoost importance (prediction-value change, %)",
           title="Top 15 inputs, CatBoost with all features")
    for y, value in enumerate(top.importance):
        ax.annotate(f"{value:.1f}", (value, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
    style(ax)
    fig.savefig(FIGURES / "report_importance.png", dpi=160)
    plt.close(fig)


def earlier_best_f1():
    """Best 2023 F1 per centre in earlier experiments whose result tables are on disk."""
    frames = []
    sources = {"03 learned baselines": EXPERIMENTS / "03_exp_learned_baselines_bc/results/metrics_by_year.csv",
               "05 CatBoost full data": EXPERIMENTS / "05_catboost_full_data/results/catboost_metrics_2023.csv",
               "06 CatBoost undersampled": EXPERIMENTS / "06_catboost_undersampled/results/catboost_metrics_2023.csv",
               "07 TabPFN hybrid": EXPERIMENTS / "07_tabpfn/results/tabpfn_metrics_2023.csv"}
    for source, path in sources.items():
        if path.exists():
            frame = pd.read_csv(path)
            frame = frame.loc[frame.year.eq(2023)] if "year" in frame else frame
            frames.append(frame.assign(source=source)[["region", "source", "model", "feature_set", "f1"]])
    gnn = EXPERIMENTS / "04_exp_gnn_bc/results/comparison_2023.csv"
    if gnn.exists():
        frame = pd.read_csv(gnn)
        frame = frame.loc[frame.source.eq("04_gnn")].assign(source="04 GNN / MLP")
        frames.append(frame[["region", "source", "model", "feature_set", "f1"]])
    everything = pd.concat(frames, ignore_index=True)
    best = everything.loc[everything.groupby("region").f1.idxmax()].set_index("region")
    return best.assign(name=best.source + " · " + best.model.str.replace("_", " "))


def main():
    ensure_figures()
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    assert not manifest["holdout_loaded"]
    metrics = pd.read_csv(RESULTS / "metrics_by_year.csv")
    ablation = pd.read_csv(RESULTS / "ablation_2023.csv").set_index("region")
    curves = pd.read_csv(RESULTS / "pr_curves_2023.csv")
    importance = pd.read_csv(RESULTS / "feature_importance.csv")
    oracle = pd.read_csv(RESULTS / "best_threshold_f1_2023.csv")
    sample = pd.read_csv(RESULTS / "training_sample.csv")
    fit_checks = pd.read_csv(RESULTS / "fit_checks.csv")
    checks = pd.read_csv(RESULTS / "f1_checks_2023.csv")

    evaluation = metrics.loc[metrics.role.eq("development_evaluation")]
    rule = report_03.with_accuracy(evaluation.loc[evaluation.model.eq("rule_local_1d")].set_index("region"))
    pick = lambda model, feature_set, rule_name="2022", frame=evaluation: report_03.with_accuracy(frame.loc[
        frame.model.eq(model) & frame.feature_set.eq(feature_set) & frame.threshold_rule.eq(rule_name)].set_index("region"))
    table = pd.DataFrame({"rule_f1": rule.f1, "rule_ap": rule.average_precision})
    for model, feature_set in SERIES:
        part = pick(model, feature_set)
        for metric in ("f1", "average_precision", "precision", "recall", "balanced_accuracy", "accuracy", "always_no_accuracy"):
            table[f"{model}__{feature_set}_{'ap' if metric == 'average_precision' else metric}"] = part[metric]
    pooled = pick("gradient_boosting", "all", "pooled_2018_2022")
    pooled_base = pick("gradient_boosting", "base", "pooled_2018_2022")
    best_key = max(SERIES, key=lambda k: table[f"{k[0]}__{k[1]}_f1"].median())
    best_col = f"{best_key[0]}__{best_key[1]}"
    order = table[f"{best_col}_f1"].sort_values(ascending=False).index.tolist()
    regions = order

    f1_chart(table, order)
    ablation_chart(ablation, regions)
    rule_points = rule[["precision", "recall"]]
    pr_curves(curves, rule_points, order)
    all_importance = importance.loc[importance.model.eq("catboost") & importance.feature_set.eq("all")]
    importance_chart(all_importance, manifest["feature_groups"])

    gain_f1 = table["gradient_boosting__all_f1"] - table["gradient_boosting__base_f1"]
    gain_ap = table["gradient_boosting__all_ap"] - table["gradient_boosting__base_ap"]
    best_vs_rule = table[f"{best_col}_f1"] - table.rule_f1
    cat_vs_hgb = table["catboost__all_f1"] - table["gradient_boosting__all_f1"]
    pooled_gain = pooled.f1 - table["gradient_boosting__all_f1"]
    earlier = earlier_best_f1()
    vs_earlier = table[f"{best_col}_f1"] - earlier.f1

    summary = html_table(pd.DataFrame({
        "Fire Centre": order,
        "Rule F1": table.loc[order, "rule_f1"].to_numpy(),
        "Base F1 (= 03)": table.loc[order, "gradient_boosting__base_f1"].to_numpy(),
        "Boosting + context F1": table.loc[order, "gradient_boosting__all_f1"].to_numpy(),
        "CatBoost + context F1": table.loc[order, "catboost__all_f1"].to_numpy(),
        "Best earlier F1 (03–07)": earlier.loc[order, "f1"].to_numpy(),
        "Rule AP": table.loc[order, "rule_ap"].to_numpy(),
        "Base AP": table.loc[order, "gradient_boosting__base_ap"].to_numpy(),
        "Boosting + context AP": table.loc[order, "gradient_boosting__all_ap"].to_numpy(),
        "CatBoost + context AP": table.loc[order, "catboost__all_ap"].to_numpy(),
        "Best precision": table.loc[order, f"{best_col}_precision"].to_numpy(),
        "Best recall": table.loc[order, f"{best_col}_recall"].to_numpy(),
        "Best balanced accuracy": table.loc[order, f"{best_col}_balanced_accuracy"].to_numpy()}),
        {**{c: "{:.3f}" for c in ("Rule F1", "Base F1 (= 03)", "Boosting + context F1", "CatBoost + context F1",
                                  "Best earlier F1 (03–07)", "Rule AP", "Base AP", "Boosting + context AP",
                                  "CatBoost + context AP", "Best balanced accuracy")},
         "Best precision": "{:.1%}", "Best recall": "{:.1%}"})
    earlier_table = html_table(pd.DataFrame({
        "Fire Centre": order, "Best earlier configuration": earlier.loc[order, "name"].to_numpy(),
        "Its F1": earlier.loc[order, "f1"].to_numpy(),
        "Experiment 13 best F1": table.loc[order, f"{best_col}_f1"].to_numpy(),
        "Change": vs_earlier.loc[order].to_numpy()}),
        {"Its F1": "{:.3f}", "Experiment 13 best F1": "{:.3f}", "Change": "{:+.3f}"})

    sets = list(SET_NAMES)
    ablation_table = html_table(pd.DataFrame({
        "Feature set": [SET_NAMES[s] for s in sets],
        "Inputs": [int(fit_checks.loc[fit_checks.model.eq("gradient_boosting") & fit_checks.feature_set.eq(s), "inputs"].iloc[0]) for s in sets],
        "Median AP": [ablation[f"average_precision__{s}"].median() for s in sets],
        "Median AP change": [(ablation[f"average_precision__{s}"] - ablation["average_precision__base"]).median() for s in sets],
        "Median F1": [ablation[f"f1__{s}"].median() for s in sets],
        "Median F1 change": [(ablation[f"f1__{s}"] - ablation["f1__base"]).median() for s in sets],
        "Centres with higher F1": [f"{int(((ablation[f'f1__{s}'] - ablation['f1__base']) > 0).sum())} of 6" if s != "base" else "–" for s in sets]}),
        {"Median AP": "{:.3f}", "Median AP change": "{:+.3f}", "Median F1": "{:.3f}", "Median F1 change": "{:+.3f}"})

    oracle_all = oracle.loc[oracle.model.eq(best_key[0]) & oracle.feature_set.eq(best_key[1])].set_index("region")
    threshold_table = html_table(pd.DataFrame({
        "Fire Centre": order,
        "Boosting + context, 2022 threshold": table.loc[order, "gradient_boosting__all_f1"].to_numpy(),
        "Boosting + context, pooled 2018–2022 threshold": pooled.loc[order, "f1"].to_numpy(),
        "Base, 2022 threshold": table.loc[order, "gradient_boosting__base_f1"].to_numpy(),
        "Base, pooled threshold": pooled_base.loc[order, "f1"].to_numpy(),
        "Best model, best threshold in hindsight": oracle_all.loc[order, "best_f1_any_2023_threshold"].to_numpy()}),
        {c: "{:.3f}" for c in ("Boosting + context, 2022 threshold", "Boosting + context, pooled 2018–2022 threshold",
                               "Base, 2022 threshold", "Base, pooled threshold", "Best model, best threshold in hindsight")})

    vegetation = metrics.loc[metrics.role.eq("vegetation_target_check")]
    veg = lambda model, feature_set: vegetation.loc[vegetation.model.eq(model) & vegetation.feature_set.eq(feature_set)].set_index("region")
    veg_rule, veg_best = veg("rule_local_1d", "rule"), veg(*best_key)
    vegetation_table = html_table(pd.DataFrame({
        "Fire Centre": order,
        "Positives, recorded target": rule.loc[order, "positive_cell_days"].to_numpy(),
        "Positives, vegetation-fire target": veg_rule.loc[order, "positive_cell_days"].to_numpy(),
        "Rule F1, recorded": table.loc[order, "rule_f1"].to_numpy(),
        "Rule F1, vegetation": veg_rule.loc[order, "f1"].to_numpy(),
        "Best model F1, recorded": table.loc[order, f"{best_col}_f1"].to_numpy(),
        "Best model F1, vegetation": veg_best.loc[order, "f1"].to_numpy()}),
        {"Positives, recorded target": "{:,}", "Positives, vegetation-fire target": "{:,}",
         **{c: "{:.3f}" for c in ("Rule F1, recorded", "Rule F1, vegetation", "Best model F1, recorded",
                                  "Best model F1, vegetation")}})

    check = lambda model, feature_set: checks.loc[checks.model.eq(model) & checks.feature_set.eq(feature_set)].set_index("region")
    check_base, check_best = check("gradient_boosting", "base"), check(*best_key)
    interval = lambda frame, low, high: [f"{a:+.3f} to {b:+.3f}" for a, b in zip(frame.loc[order, low], frame.loc[order, high])]
    noise_table = html_table(pd.DataFrame({
        "Fire Centre": order,
        "Base minus rule": check_base.loc[order, "difference"].to_numpy(),
        "95% interval (base)": interval(check_base, "difference_ci_low", "difference_ci_high"),
        "Best model minus rule": check_best.loc[order, "difference"].to_numpy(),
        "95% interval (best)": interval(check_best, "difference_ci_low", "difference_ci_high"),
        "Best model, best threshold in hindsight": oracle_all.loc[order, "best_f1_any_2023_threshold"].to_numpy()
                                                   - table.loc[order, "rule_f1"].to_numpy(),
        "Best model at rule volume, minus rule": check_best.loc[order, "rule_volume_difference"].to_numpy(),
        "95% interval (rule volume)": interval(check_best, "rule_volume_ci_low", "rule_volume_ci_high")}),
        {"Base minus rule": "{:+.3f}", "Best model minus rule": "{:+.3f}",
         "Best model, best threshold in hindsight": "{:+.3f}", "Best model at rule volume, minus rule": "{:+.3f}"})
    excludes_zero = lambda frame, low, high: int(((frame[low] > 0) | (frame[high] < 0)).sum())
    base_significant = int((check_base.difference_ci_high < 0).sum())
    best_significant = excludes_zero(check_best, "difference_ci_low", "difference_ci_high")
    volume_significant = excludes_zero(check_best, "rule_volume_ci_low", "rule_volume_ci_high")
    hindsight_gap = oracle_all.best_f1_any_2023_threshold - table.rule_f1
    full = report_03.with_accuracy(evaluation.loc[evaluation.model.ne("rule_local_1d")].copy())
    full = full.assign(name=full.model.str.replace("_", " ") + " · " + full.feature_set.map(SET_NAMES) + " · "
                       + full.threshold_rule.map({"2022": "2022 threshold", "pooled_2018_2022": "pooled threshold"}))
    full_table = html_table(full.sort_values(["region", "model", "feature_set", "threshold_rule"])[
        ["region", "name", "average_precision", "threshold", "precision", "recall", "f1", "balanced_accuracy", "tp", "fp", "fn"]].rename(columns={
            "region": "Fire Centre", "name": "Model · features · threshold", "average_precision": "Average precision",
            "threshold": "Threshold", "precision": "Precision", "recall": "Recall", "f1": "F1",
            "balanced_accuracy": "Balanced accuracy", "tp": "Correct flags", "fp": "False flags", "fn": "Missed"}),
        {"Average precision": "{:.3f}", "Threshold": "{:.3f}", "Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.3f}",
         "Balanced accuracy": "{:.3f}", "Correct flags": "{:,}", "False flags": "{:,}", "Missed": "{:,}"})
    groups_table = html_table(pd.DataFrame(
        [{"Group": GROUP_TEXT[g][0], "What it adds": GROUP_TEXT[g][1], "Source": GROUP_TEXT[g][2], "Columns": ", ".join(cols)}
         for g, cols in manifest["feature_groups"].items()]), {})
    totals = sample.drop(columns=["region", "year"]).sum()
    balance_table = html_table(pd.DataFrame({
        "Row type": ["All 2018–2021 cell-days", "Positives (all kept)", "Active rows (all kept)",
                     "Quiet positives (all kept)", "Sampled quiet negatives (1%)", "Training rows"],
        "Count": [totals.rows, totals.positives, totals.active_rows, totals.quiet_positive_rows,
                  totals.sampled_quiet_negatives, totals.kept_rows]}), {"Count": "{:,.0f}"})
    images = {"f1": embedded_image(FIGURES / "report_f1.png", "2023 F1 by Fire Centre"),
              "ablation": embedded_image(FIGURES / "report_ablation.png", "Change in AP and F1 from each feature group"),
              "pr": embedded_image(FIGURES / "report_pr_curves.png", "2023 precision-recall curves by Fire Centre"),
              "importance": embedded_image(FIGURES / "report_importance.png", "CatBoost feature importance")}
    best_name = SERIES[best_key][0]
    per_group = {g: (ablation[f"f1__base+{g}"] - ablation["f1__base"]).median() for g in GROUP_TEXT}
    strongest = max(per_group, key=per_group.get)
    per_group_ap = {g: (ablation[f"average_precision__base+{g}"] - ablation["average_precision__base"]).median() for g in GROUP_TEXT}
    strongest_ap = max(per_group_ap, key=per_group_ap.get)
    bcws_extra = (ablation["f1__all"] - ablation["f1__firms_context"]).median()

    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BC | Fire-context features</title>
<style>
:root{{--ink:#192c31;--muted:#54676b;--accent:#bd5417;--teal:#237e85;--line:#dce4e3}}
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f4;color:var(--ink);font:16px/1.55 system-ui,sans-serif}}
main{{max-width:1320px;margin:auto;padding:42px 28px}}header{{padding:15px 0 30px;border-bottom:2px solid var(--ink)}}
.eyebrow{{font-size:12px;font-weight:750;letter-spacing:2px;color:var(--accent);text-transform:uppercase}}
h1{{font-size:clamp(30px,4vw,48px);line-height:1.1;margin:12px 0 18px}}h2{{font-size:25px;margin:0 0 12px}}h3{{font-size:19px;margin:0 0 8px}}
p{{margin:8px 0}}.purpose{{font-size:21px;max-width:900px}}section{{padding:30px 0;border-bottom:1px solid var(--line)}}
ul{{padding-left:21px;margin:12px 0}}li{{margin:5px 0}}.muted,figcaption{{color:var(--muted);font-size:14px}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin:20px 0}}figure{{margin:0;background:white;border:1px solid var(--line);border-radius:12px;padding:12px}}
img{{display:block;width:100%;height:auto}}figcaption{{padding:8px 4px}}
.card{{background:white;border:1px solid var(--line);border-radius:12px;padding:20px}}.value{{font-size:40px;font-weight:750;color:var(--teal)}}
details{{margin-top:18px}}summary{{cursor:pointer;font-weight:650;padding:12px;background:#e9efec;border-radius:8px}}
.conclusion{{background:#e9f0ed;border-radius:12px;padding:24px}}
.table{{overflow-x:auto;background:white;border:1px solid var(--line);border-radius:12px;margin:16px 0}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:8px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}th{{color:var(--muted);font-weight:650}}
td{{font-variant-numeric:tabular-nums}}tbody tr:last-child td{{border-bottom:none}}
.wrap td{{white-space:normal;text-align:left;min-width:160px}}
footer{{color:var(--muted);font-size:13px;padding:20px 0}}@media(max-width:800px){{.cards{{grid-template-columns:1fr}}main{{padding:24px 16px}}}}
@media print{{body{{background:white}}main{{padding:0}}figure,.card{{break-inside:avoid}}}}
</style></head><body><main>
<header><div class="eyebrow">Experiment 13 · Fire-context features</div>
<h1>Do richer fire-context features raise F1?</h1>
<p class="purpose"><strong>Purpose:</strong> the models in Experiments 03–07 all reach F1 of about 0.45–0.58, close to what their inputs allow. This experiment adds new information instead of a new model: fire intensity, a wider neighbourhood, fire-complex size and duration, and BC Wildfire Service incidents.</p>
<p class="muted">Six Fire Centres · same rows, target, split, sampling and correction as Experiment 03 · train 2018–2021 · thresholds 2022 · evaluate 2023 · 2024 excluded</p></header>

<section><h2>01 / Setup: four new feature groups</h2>
<div class="table wrap">{groups_table[len('<div class="table">'):-len('</div>')]}</div>
<ul><li><strong>Base set:</strong> Experiment 03's cell + neighbour inputs. Gradient boosting on the base set reproduces Experiment 03's 2023 AP exactly in every centre (checked by the script), so every difference below comes from the new inputs.</li>
<li><strong>Past only:</strong> every FIRMS feature for target day t uses detections up to t−1. BCWS incidents are used one day later still (ignition and out dates up to t−2) in case entries were back-dated. Final fire size and final perimeters are never used, because they are only known after the fire.</li>
<li><strong>Neighbourhood:</strong> new ring features count detections in any detected cell, across Fire Centre borders; the base neighbour counts stay inside each centre, as in Experiment 02.</li>
<li><strong>Models:</strong> gradient boosting with Experiment 03's fixed settings for every feature set, and CatBoost with Experiments 05–06's settings on the base and full sets. No tuning.</li>
<li><strong>Thresholds:</strong> as before, each centre's threshold maximizes 2022 F1. A second rule pools evidence from 2018–2022 (see section 04), because 2022 was a quiet year.</li></ul>
<details><summary>Show the training sample (identical to Experiment 03 cell + neighbours)</summary>{balance_table}
<p class="muted">Quiet means no detection in the cell or its in-centre neighbours over the previous 7 days. Quiet-row log-odds are shifted by log(0.01) at prediction; 2022 and 2023 use every row.</p></details></section>

<section><h2>02 / Results</h2>
<div class="cards"><div class="card"><h3>F1 gain from new features</h3><div class="value">{gain_f1.median():+.3f}</div><p>Median change, gradient boosting with all groups vs base; higher in {int((gain_f1 > 0).sum())} of 6 centres. AP changes by {gain_ap.median():+.3f}.</p></div>
<div class="card"><h3>Best F1 vs previous-day rule</h3><div class="value">{best_vs_rule.median():+.3f}</div><p>{best_name}; above the rule in {int((best_vs_rule > 0).sum())} of 6 centres.</p></div>
<div class="card"><h3>Best F1 vs best earlier model</h3><div class="value">{vs_earlier.median():+.3f}</div><p>Against each centre's best F1 from Experiments 03–07; higher in {int((vs_earlier > 0).sum())} of 6.</p></div></div>
{summary}
{report_03.accuracy_note(pd.DataFrame({"always_no_accuracy": table[f"{best_col}_always_no_accuracy"]}))}
<figure>{images["f1"]}<figcaption>Gray diamonds are the previous-day rule, which Experiments 03–07 never clearly beat on F1. Colour and marker identify each model.</figcaption></figure>
<figure style="margin-top:20px">{images["pr"]}<figcaption>Each curve shows every precision–recall trade-off a model offers. Where the gray diamond sits below a curve, the model beats the rule at that operating point.</figcaption></figure>
<details><summary>Show the best earlier configuration per Fire Centre</summary>{earlier_table}
<p class="muted">From result tables on disk for Experiments 03, 04, 05, 06 and 07, all at 2022-selected thresholds. Experiments 03a/04a (CEMS weather) store results only in their HTML reports and are not included; their F1 was within 0.03 of the rule.</p></details></section>

<section><h2>03 / Which feature groups help</h2>
<p>Each group is added to the base set on its own, then together. Gradient boosting throughout.</p>
<figure>{images["ablation"]}<figcaption>Blue dots are the median change across centres; gray dots are individual centres.</figcaption></figure>
{ablation_table}
<ul><li>For ranking, the strongest single group is <strong>{GROUP_TEXT[strongest_ap][0]}</strong> (median AP {per_group_ap[strongest_ap]:+.3f}). For F1 it is <strong>{GROUP_TEXT[strongest][0]}</strong> (median {per_group[strongest]:+.3f}), but single-group F1 changes are within the noise shown in section 05.</li>
<li>Adding BCWS incidents on top of all FIRMS groups changes F1 by a median of {bcws_extra:+.3f}.</li></ul>
<figure>{images["importance"]}<figcaption>Importance from CatBoost on all features. Correlated inputs share credit, so treat this as a ranking of where the model looked, not a causal effect.</figcaption></figure></section>

<section><h2>04 / Thresholds</h2>
<p>2022 had few positives (as few as {int(metrics.loc[metrics.year.eq(2022) & metrics.model.eq('rule_local_1d')].positive_cell_days.min()):,} in one centre), so the 2022-only threshold is noisy. The pooled rule also uses out-of-year scores for 2018–2021: each training year is scored by a model fitted on the other three years, with quiet negatives weighted ×100 to stand for the unsampled rows. The last column picks the best threshold on 2023 itself. It cannot be used in practice and only shows how much is left on the table.</p>
{threshold_table}
<p class="muted">Pooled thresholds change F1 by a median of {pooled_gain.median():+.3f} for boosting with all groups.</p></section>

<section><h2>05 / Is the remaining gap to the rule real?</h2>
<p>Two checks on the 2023 predictions. <strong>Noise:</strong> the 95% intervals resample whole calendar weeks of 2023 (2,000 draws), because fires span many days and single days are not independent. <strong>Threshold or scores:</strong> "rule volume" flags, each day, as many cells as the rule does (the number detected yesterday, known in advance), but picks them by model score. If the model ranked active cells better than "detected yesterday", this would beat the rule.</p>
{noise_table}
<ul><li><strong>The gap closed.</strong> Base boosting was clearly below the rule in {base_significant} of 6 centres (interval entirely below zero). For {best_name}, {best_significant} of 6 intervals exclude zero: the remaining differences are within noise.</li>
<li><strong>But nothing beats the rule at its own operating point.</strong> Even the best threshold chosen on 2023 in hindsight is a median {hindsight_gap.median():+.3f} from the rule, and at the rule's alert volume the model is a median {check_best.rule_volume_difference.median():+.3f} ({volume_significant} of 6 intervals exclude zero). Given the same number of alerts, the model picks cells about as well as "detected yesterday".</li>
<li>So the previous-day rule sits on the best precision–recall curve these inputs allow, near its F1 peak. The AP gains are real, but they are away from that point: fewer, more confident alerts, or more recall at lower precision. To raise F1 itself, the model needs information on which burning cells stop the next day (day-t rain and wind), and labels with fewer missed days (NOAA-20).</li></ul></section>

<section><h2>06 / Label check: vegetation fires only</h2>
<p>FIRMS flags some detections as static sources (type 2, such as industrial sites) or offshore (type 3). The table re-scores the same predictions against a target that ignores those detections. Models were not retrained.</p>
{vegetation_table}
<p class="muted">Largest effect: {(rule.positive_cell_days - veg_rule.positive_cell_days).idxmax()}, where {int((rule.positive_cell_days - veg_rule.positive_cell_days).max()):,} positives come only from static or offshore detections. These persistent sources favour the rule, so dropping them lowers its F1 there; elsewhere the effect is negligible.</p></section>

<section><h2>07 / All configurations</h2>
<details><summary>Show every model, feature set and threshold rule for every Fire Centre (2023)</summary>{full_table}</details></section>

<section><div class="conclusion"><h2>08 / Conclusion</h2><ul>
<li><strong>The new inputs are the largest ranking gain so far:</strong> all four groups change boosting AP by a median of {gain_ap.median():+.3f} and F1 by {gain_f1.median():+.3f} over the identical base model. FRP and day/night carries most of the AP gain.</li>
<li><strong>F1 now ties the previous-day rule but does not beat it:</strong> the best configuration ({best_name}) is {best_vs_rule.median():+.3f} median F1 from the rule, with {best_significant} of 6 differences outside week-resampled noise. Base boosting was clearly below the rule in {base_significant} of 6.</li>
<li><strong>The ceiling is the inputs, not the threshold:</strong> at the rule's own alert volume, or at the best threshold in hindsight, the model only matches the rule. Beating it needs information about which active cells stop: day-t rain and wind (ERA5-Land; download script ready) and NOAA-20 labels (requested). This pipeline reruns unchanged on both.</li></ul></div>
<ul class="muted"><li>Targets are <strong>recorded detections</strong>, not ignitions, unique fires or confirmed absence. Coverage remains unknown.</li>
<li>BCWS incidents: only records typed "Fire"; a missing out date is treated as active for 30 days and activity is capped at 180 days. Ignition dates may still be set after the fact for some fires; the one-day delay reduces but cannot rule out that leak.</li>
<li>Untuned models and a single seed: differences under about 0.01 F1 are within noise. 2023 is a development evaluation; 2024 remains excluded.</li></ul></section>
<footer>Generated by build_report.py from saved Experiment 13 results, with Experiment 02–07 tables for comparison. All figures are embedded: this report works offline as one HTML file.</footer>
</main></body></html>'''
    output = HERE / "RESULTS.html"
    output.write_text(page, encoding="utf-8")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
