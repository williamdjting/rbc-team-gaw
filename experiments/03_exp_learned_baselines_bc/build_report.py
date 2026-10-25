"""Build a standalone visual HTML report from saved Experiment 03 outputs."""
import base64
import html as escape
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from report_class_balance import add_class_balance

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
REFERENCE = HERE.parent / "02_hotspot_persistence_bc"
MODEL_NAMES = {"logistic_regression": "Logistic regression", "gradient_boosting": "Gradient boosting"}
SET_NAMES = {"cell_only": "cell only", "cell_plus_neighbours": "cell + neighbours"}
# Reference categorical slots 1-2; feature set is carried by marker or line weight, not colour.
MODEL_COLOURS = {"logistic_regression": "#2a78d6", "gradient_boosting": "#eb6834"}
SET_MARKERS = {"cell_only": "o", "cell_plus_neighbours": "^"}
RULE, INK, MUTED, GRID = "#898781", "#192c31", "#54676b", "#e1e0d9"
CONFIGS = [(m, s) for s in SET_NAMES for m in MODEL_NAMES]


def embedded_image(path, alt):
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="{alt}" loading="lazy">'


def label(model, feature_set):
    return f"{MODEL_NAMES[model]}, {SET_NAMES[feature_set]}"


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(color=GRID, linewidth=.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED)


def with_accuracy(frame):
    """Accuracy, the accuracy of always predicting "no", and balanced accuracy, from confusion counts."""
    rows = frame.tp + frame.fp + frame.fn + frame.tn
    return frame.assign(accuracy=(frame.tp + frame.tn) / rows, always_no_accuracy=(frame.fp + frame.tn) / rows,
                        balanced_accuracy=(frame.tp / (frame.tp + frame.fn) + frame.tn / (frame.tn + frame.fp)) / 2)


def accuracy_note(frame):
    always_no = frame.always_no_accuracy
    return (f'<p class="muted"><strong>Accuracy</strong> is shown for completeness but cannot separate these models. '
            f'Only {1 - always_no.max():.2%}–{1 - always_no.min():.2%} of cell-days have a detection, so predicting '
            f'"no detection" everywhere scores {always_no.min():.2%}–{always_no.max():.2%} while catching nothing. '
            f'<strong>Balanced accuracy</strong> averages recall on active and on quiet cell-days, so always "no" '
            f'scores 0.500. It still depends on the chosen threshold; average precision remains the headline metric.</p>')


def html_table(frame, formats):
    head = "".join(f"<th>{escape.escape(column)}</th>" for column in frame.columns)
    body = "".join("<tr>" + "".join(
        f"<td>{escape.escape(formats.get(column, '{}').format(value))}</td>" for column, value in row.items())
        + "</tr>" for _, row in frame.iterrows())
    return f'<div class="table"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def ap_chart(evaluation, rules, order):
    fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    positions = {region: index for index, region in enumerate(order)}
    offsets = dict(zip(CONFIGS, (-.24, -.08, .08, .24)))
    for region, index in positions.items():
        ax.scatter(rules.loc[region, "selected_rule_average_precision"], index, marker="D", s=60, color=RULE,
                   edgecolor="white", linewidth=1.5, zorder=3)
        ax.scatter(rules.loc[region, "best_rule_average_precision_2023"], index, marker="D", s=60,
                   facecolor="white", edgecolor=RULE, linewidth=1.5, zorder=3)
    for (model, feature_set), offset in offsets.items():
        rows = evaluation.loc[evaluation.model.eq(model) & evaluation.feature_set.eq(feature_set)].set_index("region")
        ax.scatter(rows.loc[order, "average_precision"], np.arange(len(order)) + offset, s=58,
                   marker=SET_MARKERS[feature_set], color=MODEL_COLOURS[model], edgecolor="white", linewidth=1.5, zorder=4)
    handles = [Line2D([], [], linestyle="none", marker="D", markersize=7, markerfacecolor=RULE, markeredgecolor="white",
                      label="Selected rule (chosen on 2022)"),
               Line2D([], [], linestyle="none", marker="D", markersize=7, markerfacecolor="white", markeredgecolor=RULE,
                      label="Best rule in hindsight (2023)")]
    handles += [Line2D([], [], linestyle="none", marker=SET_MARKERS[s], markersize=8, markerfacecolor=MODEL_COLOURS[m],
                       markeredgecolor="white", label=label(m, s)) for m, s in CONFIGS]
    ax.set(yticks=list(positions.values()), yticklabels=order, xlim=(0, 1), xlabel="2023 average precision",
           title="2023 average precision by Fire Centre: learned models vs fixed rules")
    ax.invert_yaxis()
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    style(ax)
    fig.savefig(FIGURES / "report_average_precision.png", dpi=160)
    plt.close(fig)


def pr_curves(curves, rule_points, order):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.2), sharex=True, sharey=True, constrained_layout=True)
    for ax, region in zip(axes.flat, order):
        for model, feature_set in CONFIGS:
            curve = curves.loc[curves.region.eq(region) & curves.model.eq(model) & curves.feature_set.eq(feature_set)]
            neighbours = feature_set == "cell_plus_neighbours"
            ax.plot(curve.recall, curve.precision, color=MODEL_COLOURS[model], linewidth=2 if neighbours else 1,
                    alpha=1 if neighbours else .55, zorder=3 if neighbours else 2)
        points = rule_points.loc[rule_points.region.eq(region)]
        for rule in points.itertuples():
            ax.scatter(rule.recall, rule.precision, s=36, color=RULE, edgecolor="white", linewidth=1,
                       marker="^" if "neighbour" in rule.rule else "o", zorder=4)
        ax.set(title=region, xlim=(0, 1), ylim=(0, 1))
        style(ax)
    for ax in axes[-1]:
        ax.set_xlabel("Recall")
    for ax in axes[:, 0]:
        ax.set_ylabel("Precision")
    handles = [Line2D([], [], color=MODEL_COLOURS[m], linewidth=2 if s == "cell_plus_neighbours" else 1,
                      alpha=1 if s == "cell_plus_neighbours" else .55, label=label(m, s)) for m, s in CONFIGS]
    handles += [Line2D([], [], linestyle="none", marker="o", markersize=6, markerfacecolor=RULE, markeredgecolor="white",
                       label="Same-cell rules (Experiment 02)"),
                Line2D([], [], linestyle="none", marker="^", markersize=6, markerfacecolor=RULE, markeredgecolor="white",
                       label="Cell + neighbour rules (Experiment 02)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("2023 precision-recall curves: each learned model can trade precision for recall; each rule is one point")
    fig.savefig(FIGURES / "report_pr_curves.png", dpi=150)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    assert not manifest["holdout_loaded"]
    metrics = pd.read_csv(RESULTS / "metrics_by_year.csv")
    comparison = pd.read_csv(RESULTS / "comparison_2023.csv").set_index("region")
    curves = pd.read_csv(RESULTS / "pr_curves_2023.csv")
    coefficients = pd.read_csv(RESULTS / "logistic_coefficients.csv")
    sample = pd.read_csv(RESULTS / "training_sample.csv")
    rules = pd.read_csv(REFERENCE / "results/all_region_metrics.csv")
    evaluation = metrics.loc[metrics.year.eq(manifest["evaluation_year"])]
    rule_points = rules.loc[rules.year.eq(manifest["evaluation_year"]) & rules.rule.ne("always_no_detection")]

    best_config = evaluation.groupby(["model", "feature_set"]).average_precision.median().idxmax()
    best = evaluation.loc[evaluation.model.eq(best_config[0]) & evaluation.feature_set.eq(best_config[1])].set_index("region")
    order = best.average_precision.sort_values(ascending=False).index.tolist()
    ap_chart(evaluation, comparison, order)
    pr_curves(curves, rule_points, order)

    ap_gain = best.average_precision - comparison.selected_rule_average_precision
    f1_gain = best.f1 - comparison.selected_rule_f1
    hindsight_gain = best.average_precision - comparison.best_rule_average_precision_2023
    neighbour_gain = (evaluation.loc[evaluation.feature_set.eq("cell_plus_neighbours")].set_index(["region", "model"]).average_precision
                      - evaluation.loc[evaluation.feature_set.eq("cell_only")].set_index(["region", "model"]).average_precision)
    model_gain = (evaluation.loc[evaluation.model.eq("gradient_boosting")].set_index(["region", "feature_set"]).average_precision
                  - evaluation.loc[evaluation.model.eq("logistic_regression")].set_index(["region", "feature_set"]).average_precision)
    # Best precision the model reaches at or beyond the selected rule's recall (upper envelope of its curve).
    rule_beats_curve = []
    for region in order:
        selected_rule = rule_points.loc[rule_points.region.eq(region) & rule_points.rule.eq(comparison.loc[region, "selected_rule"])].iloc[0]
        curve = curves.loc[curves.region.eq(region) & curves.model.eq(best_config[0]) & curves.feature_set.eq(best_config[1])]
        if selected_rule.precision >= curve.loc[curve.recall >= selected_rule.recall, "precision"].max():
            rule_beats_curve.append(region)
    best = with_accuracy(best)
    selected_rows = with_accuracy(rule_points.merge(comparison.selected_rule.reset_index(), left_on=["region", "rule"],
                                                    right_on=["region", "selected_rule"]).set_index("region"))
    summary_table = html_table(pd.DataFrame({
        "Fire Centre": order,
        "Selected rule AP": comparison.loc[order, "selected_rule_average_precision"].to_numpy(),
        "Best rule AP (hindsight)": comparison.loc[order, "best_rule_average_precision_2023"].to_numpy(),
        "Model AP": best.loc[order, "average_precision"].to_numpy(),
        "Selected rule F1": comparison.loc[order, "selected_rule_f1"].to_numpy(),
        "Model F1": best.loc[order, "f1"].to_numpy(),
        "Model precision": best.loc[order, "precision"].to_numpy(),
        "Model recall": best.loc[order, "recall"].to_numpy(),
        "Threshold (2022)": best.loc[order, "threshold"].to_numpy(),
        "Model accuracy": best.loc[order, "accuracy"].to_numpy(),
        "Always-no accuracy": best.loc[order, "always_no_accuracy"].to_numpy(),
        "Model balanced accuracy": best.loc[order, "balanced_accuracy"].to_numpy(),
        "Rule balanced accuracy": selected_rows.loc[order, "balanced_accuracy"].to_numpy()}),
        {"Selected rule AP": "{:.3f}", "Best rule AP (hindsight)": "{:.3f}", "Model AP": "{:.3f}",
         "Selected rule F1": "{:.3f}", "Model F1": "{:.3f}", "Model precision": "{:.1%}", "Model recall": "{:.1%}",
         "Threshold (2022)": "{:.3f}", "Model accuracy": "{:.3%}", "Always-no accuracy": "{:.3%}",
         "Model balanced accuracy": "{:.3f}", "Rule balanced accuracy": "{:.3f}"})
    full = evaluation.assign(order=evaluation.model.map(list(MODEL_NAMES).index) + 2 * evaluation.feature_set.map(list(SET_NAMES).index))
    full = with_accuracy(full.sort_values(["region", "order"]))
    full_table = html_table(pd.DataFrame({
        "Fire Centre": full.region, "Model": [label(m, s) for m, s in zip(full.model, full.feature_set)],
        "Average precision": full.average_precision, "Threshold (2022)": full.threshold, "Precision": full.precision,
        "Recall": full.recall, "F1": full.f1, "Accuracy": full.accuracy, "Balanced accuracy": full.balanced_accuracy,
        "Correct flags": full.tp, "False flags": full.fp, "Missed": full.fn}),
        {"Average precision": "{:.3f}", "Threshold (2022)": "{:.3f}", "Precision": "{:.1%}", "Recall": "{:.1%}",
         "F1": "{:.3f}", "Accuracy": "{:.3%}", "Balanced accuracy": "{:.3f}", "Correct flags": "{:,}",
         "False flags": "{:,}", "Missed": "{:,}"})
    coefficient_table = html_table(
        coefficients.loc[coefficients.feature_set.eq("cell_plus_neighbours")].sort_values("coefficient", key=abs, ascending=False)
        [["feature", "coefficient"]].rename(columns={"feature": "Transformed feature", "coefficient": "Coefficient"}),
        {"Coefficient": "{:+.3f}"})
    totals = sample.drop(columns=["region", "year"]).sum()
    images = {"ap": embedded_image(FIGURES / "report_average_precision.png", "2023 average precision by Fire Centre for learned models and rules"),
              "pr": embedded_image(FIGURES / "report_pr_curves.png", "2023 precision-recall curves by Fire Centre")}
    best_label = label(*best_config)
    selection = metrics.loc[metrics.year.eq(manifest["selection_year"])].drop_duplicates("region").set_index("region")
    thin = ", ".join(f"{region} ({count:,})" for region, count in
                     selection.positive_cell_days.sort_values().items() if count < 200) or "none below 200"

    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BC | Learned baselines experiment</title>
<style>
:root{{--ink:#192c31;--muted:#54676b;--accent:#bd5417;--teal:#237e85;--line:#dce4e3}}
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f4;color:var(--ink);font:16px/1.55 system-ui,sans-serif}}
main{{max-width:1320px;margin:auto;padding:42px 28px}}header{{padding:15px 0 30px;border-bottom:2px solid var(--ink)}}
.eyebrow{{font-size:12px;font-weight:750;letter-spacing:2px;color:var(--accent);text-transform:uppercase}}
h1{{font-size:clamp(30px,4vw,48px);line-height:1.1;margin:12px 0 18px}}h2{{font-size:25px;margin:0 0 12px}}h3{{font-size:19px;margin:0 0 8px}}
p{{margin:8px 0}}.purpose{{font-size:21px;max-width:900px}}section{{padding:30px 0;border-bottom:1px solid var(--line)}}
ul{{padding-left:21px;margin:12px 0}}li{{margin:5px 0}}.muted,figcaption{{color:var(--muted);font-size:14px}}
.columns,.cards{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}figure{{margin:0;background:white;border:1px solid var(--line);border-radius:12px;padding:12px}}
img{{display:block;width:100%;height:auto}}figcaption{{padding:8px 4px}}.cards{{grid-template-columns:repeat(3,1fr);margin:20px 0}}
.card{{background:white;border:1px solid var(--line);border-radius:12px;padding:20px}}.value{{font-size:40px;font-weight:750;color:var(--teal)}}
.rule{{background:#e9f0ed;border-left:4px solid var(--teal);padding:18px 22px;margin:18px 0;font-size:20px}}
.timeline{{display:grid;grid-template-columns:1fr 45px 1fr 45px 1fr;align-items:center;gap:8px;margin:18px 0}}
.step{{padding:20px;border:1px solid var(--line);border-radius:12px;background:white}}.arrow{{font-size:30px;text-align:center}}
details{{margin-top:18px}}summary{{cursor:pointer;font-weight:650;padding:12px;background:#e9efec;border-radius:8px}}
details .columns{{margin-top:16px}}.conclusion{{background:#e9f0ed;border-radius:12px;padding:24px}}
.table{{overflow-x:auto;background:white;border:1px solid var(--line);border-radius:12px;margin:16px 0}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:8px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}th{{color:var(--muted);font-weight:650}}
td{{font-variant-numeric:tabular-nums}}tbody tr:last-child td{{border-bottom:none}}
footer{{color:var(--muted);font-size:13px;padding:20px 0}}@media(max-width:800px){{.columns,.cards,.timeline{{grid-template-columns:1fr}}.arrow{{transform:rotate(90deg)}}main{{padding:24px 16px}}}}
@media print{{body{{background:white}}main{{padding:0}}details .columns{{display:grid}}figure,.card{{break-inside:avoid}}}}
</style></head><body><main>
<header><div class="eyebrow">Experiment 03 · Learned baselines</div>
<h1>Do learned models beat the persistence rules?</h1>
<p class="purpose"><strong>Purpose:</strong> fit the first two models in the planned comparison, cell-only and cell + neighbours, on the Experiment 02 features, and set the bar a graph model must clear.</p>
<p class="muted">Six Fire Centres · same features, target and split as Experiment 02 · train 2018–2021 · thresholds 2022 · evaluate 2023 · 2024 excluded</p></header>

<section><h2>01 / Setup: same data, fitted instead of fixed</h2>
<div class="timeline"><div class="step"><div class="eyebrow">Inputs · Before target date</div><h3>Experiment 02 features</h3>Past 1/3/7-day counts, days since last detection, day of year; optionally neighbour counts.</div><div class="arrow">→</div>
<div class="step"><div class="eyebrow">Model · Trained on 2018–2021</div><h3>Two model types</h3>Logistic regression and gradient boosting, one pooled model each across all six centres.</div><div class="arrow">→</div>
<div class="step"><div class="eyebrow">Output · Next day</div><h3>Probability of a recorded detection</h3>Ranked for average precision; cut at the 2022 best-F1 threshold for precision, recall and F1.</div></div>
<ul><li><strong>Cell only</strong> (planned step 1): the cell's own past 1/3/7-day counts, days since its previous detection, a no-history flag, and day of year.</li>
<li><strong>Cell + neighbours</strong> (planned step 2): adds past 1/3/7-day counts in neighbouring cells inside the same Fire Centre.</li>
<li><strong>Training data:</strong> {int(totals.rows):,} cell-days from 2018–2021. All {int(totals.positives):,} positives and every cell-day with activity in the past 7 days are kept. Quiet negatives are sampled at {manifest["sampling"]["quiet_sample_rate"]:.0%}, giving {int(totals.cell_only_kept_rows):,} training rows (cell only) and {int(totals.cell_plus_neighbours_kept_rows):,} (cell + neighbours). Models train on the sample, then each quiet row's log-odds are shifted by log({manifest["sampling"]["quiet_sample_rate"]}) to undo the sampling exactly.</li>
<li><strong>No tuning:</strong> model settings are fixed; 2022 is used only to choose each centre's alert threshold.</li></ul></section>

<section><h2>02 / Results: {best_label.lower()}</h2>
<div class="cards"><div class="card"><h3>Average precision vs selected rule</h3><div class="value">{ap_gain.median():+.3f}</div><p>Median change across centres; better in {int((ap_gain > 0).sum())} of 6.</p></div>
<div class="card"><h3>F1 vs selected rule</h3><div class="value">{f1_gain.median():+.3f}</div><p>Median change across centres; better in {int((f1_gain > 0).sum())} of 6.</p></div>
<div class="card"><h3>AP vs best rule in hindsight</h3><div class="value">{hindsight_gain.median():+.3f}</div><p>Against whichever rule scored best on 2023 itself; better in {int((hindsight_gain > 0).sum())} of 6.</p></div></div>
<p>The configuration with the highest median 2023 average precision is <strong>{best_label}</strong>. Its per-centre results:</p>
{summary_table}
{accuracy_note(best)}
<ul><li><strong>Neighbour features:</strong> adding them changes AP by a median of {neighbour_gain.median():+.3f} (better in {int((neighbour_gain > 0).sum())} of {len(neighbour_gain)} centre × model pairs).</li>
<li><strong>Model type:</strong> gradient boosting minus logistic regression AP has a median of {model_gain.median():+.3f} (boosting better in {int((model_gain > 0).sum())} of {len(model_gain)} centre × feature-set pairs).</li>
<li><strong>The rule still wins at its own operating point.</strong> At the 2022 thresholds, F1 is below the selected rule in {int((f1_gain < 0).sum())} of 6 centres. In {len(rule_beats_curve)} of 6 ({", ".join(rule_beats_curve) or "none"}), the previous-day rule's point sits on or above the model's whole precision–recall curve at that recall. So for "flag about as many cells as were active yesterday", the rule is as good a cut as the model offers. The models' AP gain comes from ranking: confident top predictions, and more recall once neighbour features are used.</li></ul>
<figure>{images["ap"]}<figcaption>Gray diamonds are the Experiment 02 rules: filled is the rule chosen on 2022 (the fair comparison), hollow is the best rule picked after seeing 2023. Colour is the model type; circles are cell-only and triangles add neighbours.</figcaption></figure>
<figure style="margin-top:20px">{images["pr"]}<figcaption>Each curve shows every precision–recall trade-off a model offers as its threshold moves; each gray point is one fixed rule. Thick lines add neighbour features. Where a gray point sits above a curve, that rule beats the model at that operating point.</figcaption></figure>
<details><summary>Show all four configurations for every Fire Centre (2023)</summary>{full_table}</details>
<details><summary>Show logistic regression coefficients (cell + neighbours)</summary>
<p class="muted">Coefficients on standardized, transformed features: log1p counts and age, day-of-year spline terms. Correlated inputs share credit, so individual signs are not causal.</p>{coefficient_table}</details></section>

<section><div class="conclusion"><h2>03 / Conclusion</h2><ul>
<li>Learned models rank cell-days better than any fixed rule: AP is higher than the selected rule in {int((ap_gain > 0).sum())} of 6 centres. But they do not beat "active yesterday" as an alert rule: F1 is lower in {int((f1_gain < 0).sum())} of 6.</li>
<li>These set two bars for planned steps 3 and 4. A standard GNN or the graph foundation model must beat <strong>{best_label}</strong> on average precision, and the previous-day rule on F1, in each centre under the same split.</li>
<li>All inputs here describe past fire activity and the date. Weather, FRP and coverage are not yet included; they are the main candidates for further gains.</li></ul></div>
<ul class="muted"><li>Targets are <strong>recorded detections</strong>, not ignitions, unique fires or confirmed absence. Coverage remains unknown.</li>
<li>One pooled model per configuration; neighbours outside each Fire Centre are excluded, as in Experiment 02. Thresholds chosen on few 2022 positives are noisy: {thin}.</li>
<li>2023 is a development evaluation; 2024 remains excluded.</li></ul></section>
<footer>Generated by build_report.py from saved Experiment 03 tables and Experiment 02 rule metrics. All figures are embedded: this report works offline as one HTML file.</footer>
</main></body></html>'''
    output = HERE / "RESULTS.html"
    output.write_text(add_class_balance(page, HERE), encoding="utf-8")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
