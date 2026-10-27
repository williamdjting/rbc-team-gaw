"""Build a standalone visual HTML report from saved Experiment 03a outputs."""
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
REFERENCE_02 = HERE.parent / "02_hotspot_persistence_bc"
REFERENCE_03 = HERE.parent / "03_exp_learned_baselines_bc"

MODEL_NAMES = {"logistic_regression": "Logistic regression (+CEMS)", "gradient_boosting": "Gradient boosting (+CEMS)"}
SET_NAMES = {"cell_only": "cell + CEMS", "cell_plus_neighbours": "cell + neighbours + CEMS"}
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
    ax.grid(color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED)


def html_table(frame, formats):
    head = "".join(f"<th>{escape.escape(column)}</th>" for column in frame.columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape.escape(formats.get(column, '{}').format(value))}</td>" for column, value in row.items())
        + "</tr>"
        for _, row in frame.iterrows()
    )
    return f'<div class="table"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def ap_chart(evaluation, comparison, order):
    fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    positions = {region: index for index, region in enumerate(order)}
    offsets = dict(zip(CONFIGS, (-0.24, -0.08, 0.08, 0.24)))
    for region, index in positions.items():
        ax.scatter(
            comparison.loc[region, "selected_rule_average_precision"],
            index,
            marker="D",
            s=60,
            color=RULE,
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
        )
        if "exp03_average_precision__gradient_boosting__cell_plus_neighbours" in comparison.columns:
            ax.scatter(
                comparison.loc[region, "exp03_average_precision__gradient_boosting__cell_plus_neighbours"],
                index,
                marker="s",
                s=60,
                facecolor="white",
                edgecolor="#237e85",
                linewidth=1.5,
                zorder=3,
            )
    for (model, feature_set), offset in offsets.items():
        rows = evaluation.loc[evaluation.model.eq(model) & evaluation.feature_set.eq(feature_set)].set_index("region")
        ax.scatter(
            rows.loc[order, "average_precision"],
            np.arange(len(order)) + offset,
            s=58,
            marker=SET_MARKERS[feature_set],
            color=MODEL_COLOURS[model],
            edgecolor="white",
            linewidth=1.5,
            zorder=4,
        )
    handles = [
        Line2D([], [], linestyle="none", marker="D", markersize=7, markerfacecolor=RULE, markeredgecolor="white", label="Selected rule (Exp 02)"),
        Line2D([], [], linestyle="none", marker="s", markersize=7, markerfacecolor="white", markeredgecolor="#237e85", label="Best FIRMS-only model (Exp 03)"),
    ]
    handles += [
        Line2D([], [], linestyle="none", marker=SET_MARKERS[s], markersize=8, markerfacecolor=MODEL_COLOURS[m], markeredgecolor="white", label=label(m, s))
        for m, s in CONFIGS
    ]
    ax.set(
        yticks=list(positions.values()),
        yticklabels=order,
        xlim=(0, 1),
        xlabel="2023 average precision",
        title="2023 average precision by Fire Centre: FIRMS + CEMS models vs baselines",
    )
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
            ax.plot(
                curve.recall,
                curve.precision,
                color=MODEL_COLOURS[model],
                linewidth=2 if neighbours else 1,
                alpha=1 if neighbours else 0.55,
                zorder=3 if neighbours else 2,
            )
        points = rule_points.loc[rule_points.region.eq(region)]
        for rule in points.itertuples():
            ax.scatter(
                rule.recall,
                rule.precision,
                s=36,
                color=RULE,
                edgecolor="white",
                linewidth=1,
                marker="^" if "neighbour" in rule.rule else "o",
                zorder=4,
            )
        ax.set(title=region, xlim=(0, 1), ylim=(0, 1))
        style(ax)
    for ax in axes[-1]:
        ax.set_xlabel("Recall")
    for ax in axes[:, 0]:
        ax.set_ylabel("Precision")
    handles = [
        Line2D([], [], color=MODEL_COLOURS[m], linewidth=2 if s == "cell_plus_neighbours" else 1, alpha=1 if s == "cell_plus_neighbours" else 0.55, label=label(m, s))
        for m, s in CONFIGS
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("2023 precision-recall curves with CEMS Fire Weather features")
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
    rules = pd.read_csv(REFERENCE_02 / "results/all_region_metrics.csv")
    evaluation = metrics.loc[metrics.year.eq(manifest["evaluation_year"])]
    rule_points = rules.loc[rules.year.eq(manifest["evaluation_year"]) & rules.rule.ne("always_no_detection")]

    best_config = evaluation.groupby(["model", "feature_set"]).average_precision.median().idxmax()
    best = evaluation.loc[evaluation.model.eq(best_config[0]) & evaluation.feature_set.eq(best_config[1])].set_index("region")
    order = best.average_precision.sort_values(ascending=False).index.tolist()
    ap_chart(evaluation, comparison, order)
    pr_curves(curves, rule_points, order)

    ap_gain_vs_rule = best.average_precision - comparison.selected_rule_average_precision
    f1_gain_vs_rule = best.f1 - comparison.selected_rule_f1

    exp03_col = f"exp03_average_precision__{best_config[0]}__{best_config[1]}"
    if exp03_col in comparison.columns:
        ap_gain_vs_exp03 = best.average_precision - comparison[exp03_col]
    else:
        ap_gain_vs_exp03 = pd.Series(0.0, index=order)

    summary_df = pd.DataFrame({
        "Fire Centre": order,
        "Selected Rule AP": comparison.loc[order, "selected_rule_average_precision"].to_numpy(),
        "FIRMS-only AP (Exp 03)": comparison.loc[order, exp03_col].to_numpy() if exp03_col in comparison.columns else np.nan,
        "FIRMS + CEMS AP (Exp 03a)": best.loc[order, "average_precision"].to_numpy(),
        "CEMS AP Gain": ap_gain_vs_exp03.loc[order].to_numpy(),
        "Selected Rule F1": comparison.loc[order, "selected_rule_f1"].to_numpy(),
        "FIRMS + CEMS F1": best.loc[order, "f1"].to_numpy(),
        "Precision": best.loc[order, "precision"].to_numpy(),
        "Recall": best.loc[order, "recall"].to_numpy(),
    })
    summary_table = html_table(
        summary_df,
        {
            "Selected Rule AP": "{:.3f}",
            "FIRMS-only AP (Exp 03)": "{:.3f}",
            "FIRMS + CEMS AP (Exp 03a)": "{:.3f}",
            "CEMS AP Gain": "{:+.4f}",
            "Selected Rule F1": "{:.3f}",
            "FIRMS + CEMS F1": "{:.3f}",
            "Precision": "{:.1%}",
            "Recall": "{:.1%}",
        },
    )

    coefficient_table = html_table(
        coefficients.loc[coefficients.feature_set.eq("cell_plus_neighbours")]
        .sort_values("coefficient", key=abs, ascending=False)[["feature", "coefficient"]]
        .rename(columns={"feature": "Transformed feature", "coefficient": "Coefficient"}),
        {"Coefficient": "{:+.3f}"},
    )
    images = {
        "ap": embedded_image(FIGURES / "report_average_precision.png", "2023 average precision with CEMS"),
        "pr": embedded_image(FIGURES / "report_pr_curves.png", "2023 PR curves with CEMS"),
    }
    best_label = label(*best_config)

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BC | Experiment 03a: Learned baselines + CEMS Weather</title>
<style>
:root{{--ink:#192c31;--muted:#54676b;--accent:#bd5417;--teal:#237e85;--line:#dce4e3}}
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f4;color:var(--ink);font:16px/1.55 system-ui,sans-serif}}
main{{max-width:1320px;margin:auto;padding:42px 28px}}header{{padding:15px 0 30px;border-bottom:2px solid var(--ink)}}
.eyebrow{{font-size:12px;font-weight:750;letter-spacing:2px;color:var(--accent);text-transform:uppercase}}
h1{{font-size:clamp(30px,4vw,48px);line-height:1.1;margin:12px 0 18px}}h2{{font-size:25px;margin:0 0 12px}}h3{{font-size:19px;margin:0 0 8px}}
p{{margin:8px 0}}.purpose{{font-size:21px;max-width:900px}}section{{padding:30px 0;border-bottom:1px solid var(--line)}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin:20px 0}}
.card{{background:white;border:1px solid var(--line);border-radius:12px;padding:20px}}.value{{font-size:40px;font-weight:750;color:var(--teal)}}
figure{{margin:0;background:white;border:1px solid var(--line);border-radius:12px;padding:12px}}
img{{display:block;width:100%;height:auto}}figcaption{{padding:8px 4px;color:var(--muted);font-size:14px}}
.table{{overflow-x:auto;background:white;border:1px solid var(--line);border-radius:12px;margin:16px 0}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:8px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}}
th:first-child,td:first-child{{text-align:left}}th{{color:var(--muted);font-weight:650}}
details{{margin-top:18px}}summary{{cursor:pointer;font-weight:650;padding:12px;background:#e9efec;border-radius:8px}}
</style></head><body><main>
<header><div class="eyebrow">Experiment 03a · Learned Baselines + CEMS Fire Weather</div>
<h1>Does adding CEMS Fire Weather beat FIRMS-only baselines?</h1>
<p class="purpose"><strong>Purpose:</strong> evaluate the impact of adding the 4 shortlisted CEMS Fire Weather Index features (<code>fwinx</code>, <code>ffmcode</code>, <code>drtcode</code>, <code>infsinx</code>) on top of Experiment 03's learned baselines.</p></header>
<section><h2>01 / Headline Results ({best_label})</h2>
<div class="cards">
<div class="card"><h3>AP vs Selected Rule (Exp 02)</h3><div class="value">{ap_gain_vs_rule.median():+.3f}</div><p>Median change across the 6 Fire Centres.</p></div>
<div class="card"><h3>AP vs FIRMS-Only (Exp 03)</h3><div class="value">{ap_gain_vs_exp03.median():+.4f}</div><p>Direct gain from adding CEMS weather indices.</p></div>
<div class="card"><h3>F1 vs Selected Rule</h3><div class="value">{f1_gain_vs_rule.median():+.3f}</div><p>Median F1 change across centres.</p></div>
</div>
{summary_table}
<figure>{images["ap"]}<figcaption>2023 Average Precision comparison across all six BC Fire Centres.</figcaption></figure>
<figure style="margin-top:20px">{images["pr"]}<figcaption>2023 Precision–Recall curves with CEMS Fire Weather features.</figcaption></figure>
<details><summary>Show Logistic Regression Coefficients (including CEMS Weather)</summary>{coefficient_table}</details>
</section></main></body></html>"""

    output = HERE / "RESULTS.html"
    output.write_text(add_class_balance(page, HERE), encoding="utf-8")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()

