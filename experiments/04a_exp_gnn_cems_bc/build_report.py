"""Build a standalone visual HTML report from saved Experiment 04a (GNN + CEMS) outputs."""
import importlib.util
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
EXPERIMENT_03 = HERE.parent / "03_exp_learned_baselines_bc"
EXPERIMENT_04 = HERE.parent / "04_exp_gnn_bc"
EXPERIMENT_03A = HERE.parent / "03a_exp_learned_cems_bc"

spec = importlib.util.spec_from_file_location("experiment_03_report", EXPERIMENT_03 / "build_report.py")
report_03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report_03)
embedded_image, html_table, style = report_03.embedded_image, report_03.html_table, report_03.style
RULE, INK = report_03.RULE, report_03.INK

SERIES = {
    "gnn": ("GNN + CEMS (message passing)", "#2a78d6", "o"),
    "mlp": ("MLP + CEMS (no message passing)", "#1baf7a", "s"),
    "boosting": ("Exp 03a: Gradient Boosting + CEMS (cell + neighbours)", "#eb6834", "^"),
}


def ap_chart(table, order):
    fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    positions = np.arange(len(order))
    ax.scatter(
        table.loc[order, "rule_ap"],
        positions,
        marker="D",
        s=60,
        color=RULE,
        edgecolor="white",
        linewidth=1.5,
        zorder=3,
    )
    handles = [
        Line2D([], [], linestyle="none", marker="D", markersize=7, markerfacecolor=RULE, markeredgecolor="white", label="Experiment 02: selected rule")
    ]
    for key, offset in zip(SERIES, (-0.18, 0, 0.18)):
        col = f"{key}_ap"
        if col not in table.columns or table[col].isna().all():
            continue
        name, colour, marker = SERIES[key]
        ax.scatter(
            table.loc[order, col],
            positions + offset,
            s=58,
            marker=marker,
            color=colour,
            edgecolor="white",
            linewidth=1.5,
            zorder=4,
        )
        handles.append(
            Line2D([], [], linestyle="none", marker=marker, markersize=8, markerfacecolor=colour, markeredgecolor="white", label=name)
        )
    ax.set(
        yticks=positions,
        yticklabels=order,
        xlim=(0, 1),
        xlabel="2023 average precision",
        title="2023 average precision by Fire Centre (GNN + CEMS)",
    )
    ax.invert_yaxis()
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    style(ax)
    fig.savefig(FIGURES / "report_average_precision.png", dpi=160)
    plt.close(fig)


def pr_curves(curves, boosting_curves, rule_points, order):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.2), sharex=True, sharey=True, constrained_layout=True)
    for ax, region in zip(axes.flat, order):
        series = {
            "gnn": curves.loc[curves.region.eq(region) & curves.model.eq("gnn")],
            "mlp": curves.loc[curves.region.eq(region) & curves.model.eq("mlp")],
        }
        if boosting_curves is not None:
            series["boosting"] = boosting_curves.loc[boosting_curves.region.eq(region)]
        for key, curve in series.items():
            if len(curve) == 0:
                continue
            ax.plot(
                curve.recall,
                curve.precision,
                color=SERIES[key][1],
                linewidth=2 if key == "gnn" else 1.3,
                zorder=3 if key == "gnn" else 2,
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
        Line2D([], [], color=c, linewidth=2 if k == "gnn" else 1.3, label=n)
        for k, (n, c, _) in SERIES.items()
        if k != "boosting" or boosting_curves is not None
    ]
    handles += [
        Line2D([], [], linestyle="none", marker="o", markersize=6, markerfacecolor=RULE, markeredgecolor="white", label="Same-cell rules (Experiment 02)"),
        Line2D([], [], linestyle="none", marker="^", markersize=6, markerfacecolor=RULE, markeredgecolor="white", label="Cell + neighbour rules (Experiment 02)"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("2023 precision-recall curves by Fire Centre (GNN + CEMS)")
    fig.savefig(FIGURES / "report_pr_curves.png", dpi=150)
    plt.close(fig)


def training_chart(history, best_epochs):
    fig, ax = plt.subplots(figsize=(11, 4.2), constrained_layout=True)
    for key in ("gnn", "mlp"):
        run = history.loc[history.network.eq(key)]
        name, colour, marker = SERIES[key]
        ax.plot(
            run.epoch,
            run.selection_mean_region_ap,
            color=colour,
            linewidth=2,
            marker=marker,
            markersize=6,
            markeredgecolor="white",
            label=name,
        )
        best = run.loc[run.epoch.eq(best_epochs[key])].iloc[0]
        ax.annotate(
            f"kept: epoch {int(best.epoch)}, {best.selection_mean_region_ap:.3f}",
            (best.epoch, best.selection_mean_region_ap),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=INK,
        )
    ax.set(xlabel="Epoch", ylabel="2022 mean per-centre AP", title="Epoch selection on 2022 (2023 not used)")
    ax.legend(loc="lower right", frameon=False)
    style(ax)
    fig.savefig(FIGURES / "report_training.png", dpi=160)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    assert not manifest["holdout_loaded"]
    comparison = pd.read_csv(RESULTS / "comparison_2023.csv")
    rows = (
        pd.read_csv(RESULTS / "metrics_by_year.csv")
        .query(f"year == {manifest['evaluation_year']}")
        .drop_duplicates("region")
        .set_index("region")
        .rows
    )
    comparison = report_03.with_accuracy(
        comparison.assign(tn=comparison.region.map(rows) - comparison.tp - comparison.fp - comparison.fn)
    )
    curves = pd.read_csv(RESULTS / "pr_curves_2023.csv")
    history = pd.read_csv(RESULTS / "training_history.csv")

    boosting_curves = None
    if (EXPERIMENT_03A / "results/pr_curves_2023.csv").exists():
        bc = pd.read_csv(EXPERIMENT_03A / "results/pr_curves_2023.csv")
        boosting_curves = bc.loc[bc.model.eq("gradient_boosting") & bc.feature_set.eq("cell_plus_neighbours")]

    rules = pd.read_csv(HERE.parent / "02_hotspot_persistence_bc/results/all_region_metrics.csv")
    rule_points = rules.loc[rules.year.eq(manifest["evaluation_year"]) & rules.rule.ne("always_no_detection")]
    best_epochs = manifest["training"]["best_epochs"]

    def pick(source, model, feature_set=None):
        sub = comparison.loc[
            comparison.source.isin([source] if isinstance(source, str) else source)
            & comparison.model.eq(model)
            & (comparison.feature_set.eq(feature_set) if feature_set else True)
        ]
        return sub.set_index("region") if len(sub) else None

    parts = {
        "rule": pick("02_rules", "selected_rule"),
        "mlp": pick(["04a_gnn_cems", "06_gnn_cems"], "mlp"),
        "gnn": pick(["04a_gnn_cems", "06_gnn_cems"], "gnn"),
    }
    boosting_part = pick(["03a_learned_cems", "05_learned_cems"], "gradient_boosting", "cell_plus_neighbours")
    if boosting_part is not None:
        parts["boosting"] = boosting_part
    exp04_gnn_part = pick("04_gnn_firms_only", "gnn")
    if exp04_gnn_part is not None:
        parts["exp04_gnn"] = exp04_gnn_part

    table = pd.concat(
        {
            f"{key}_{metric}": frame[column]
            for key, frame in parts.items()
            if frame is not None
            for metric, column in (
                ("ap", "average_precision"),
                ("f1", "f1"),
                ("precision", "precision"),
                ("recall", "recall"),
                ("accuracy", "accuracy"),
                ("always_no", "always_no_accuracy"),
                ("balanced", "balanced_accuracy"),
            )
        },
        axis=1,
    )
    order = table.gnn_ap.sort_values(ascending=False).index.tolist()
    ap_chart(table, order)
    pr_curves(curves, boosting_curves, rule_points, order)
    training_chart(history, best_epochs)

    graph_gain = table.gnn_ap - table.mlp_ap
    vs_rule_f1 = table.gnn_f1 - table.rule_f1
    vs_exp04 = (table.gnn_ap - table.exp04_gnn_ap) if "exp04_gnn_ap" in table.columns else pd.Series(0.0, index=order)

    summary_cols = {
        "Fire Centre": order,
        "Rule AP": table.loc[order, "rule_ap"].to_numpy(),
    }
    if "exp04_gnn_ap" in table.columns:
        summary_cols["GNN FIRMS-Only AP (04)"] = table.loc[order, "exp04_gnn_ap"].to_numpy()
    if "boosting_ap" in table.columns:
        summary_cols["Boosting + CEMS AP (03a)"] = table.loc[order, "boosting_ap"].to_numpy()
    summary_cols.update({
        "MLP + CEMS AP": table.loc[order, "mlp_ap"].to_numpy(),
        "GNN + CEMS AP": table.loc[order, "gnn_ap"].to_numpy(),
        "Rule F1": table.loc[order, "rule_f1"].to_numpy(),
        "MLP + CEMS F1": table.loc[order, "mlp_f1"].to_numpy(),
        "GNN + CEMS F1": table.loc[order, "gnn_f1"].to_numpy(),
        "GNN precision": table.loc[order, "gnn_precision"].to_numpy(),
        "GNN recall": table.loc[order, "gnn_recall"].to_numpy(),
    })
    summary = html_table(
        pd.DataFrame(summary_cols),
        {
            "Rule AP": "{:.3f}",
            "GNN FIRMS-Only AP (04)": "{:.3f}",
            "Boosting + CEMS AP (03a)": "{:.3f}",
            "MLP + CEMS AP": "{:.3f}",
            "GNN + CEMS AP": "{:.3f}",
            "Rule F1": "{:.3f}",
            "MLP + CEMS F1": "{:.3f}",
            "GNN + CEMS F1": "{:.3f}",
            "GNN precision": "{:.1%}",
            "GNN recall": "{:.1%}",
        },
    )
    images = {
        "ap": embedded_image(FIGURES / "report_average_precision.png", "2023 average precision by Fire Centre"),
        "pr": embedded_image(FIGURES / "report_pr_curves.png", "2023 precision-recall curves by Fire Centre"),
        "training": embedded_image(FIGURES / "report_training.png", "2022 mean AP by training epoch"),
    }

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BC | Experiment 04a: GNN + CEMS Weather</title>
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
</style></head><body><main>
<header><div class="eyebrow">Experiment 04a · Graph Neural Network + CEMS Weather</div>
<h1>Does GNN message passing + CEMS Fire Weather beat FIRMS-only GNN?</h1>
<p class="purpose"><strong>Purpose:</strong> add the 4 shortlisted CEMS Fire Weather Index features (<code>fwinx</code>, <code>ffmcode</code>, <code>drtcode</code>, <code>infsinx</code>) as node inputs to the 2-layer GraphSAGE GNN and no-message-passing MLP control.</p>
<p class="muted">Six Fire Centres · {manifest["graph"]["nodes"]:,} nodes · train 2018–2021 · epochs & thresholds 2022 · evaluate 2023 · 2024 excluded</p></header>
<section><h2>01 / Training & Epoch Selection</h2>
<figure>{images["training"]}<figcaption>Mean per-centre average precision on 2022 after each epoch.</figcaption></figure></section>
<section><h2>02 / 2023 Evaluation Results</h2>
<div class="cards">
<div class="card"><h3>GNN vs MLP (Message Passing Gain)</h3><div class="value">{graph_gain.median():+.3f}</div><p>Median AP change across the 6 Fire Centres.</p></div>
<div class="card"><h3>GNN + CEMS vs GNN FIRMS-Only (04)</h3><div class="value">{vs_exp04.median():+.4f}</div><p>Median AP change from adding CEMS weather to node inputs.</p></div>
<div class="card"><h3>GNN + CEMS F1 vs Rule</h3><div class="value">{vs_rule_f1.median():+.3f}</div><p>Median F1 change at 2022 thresholds.</p></div>
</div>
{summary}
<figure>{images["ap"]}<figcaption>2023 Average Precision by Fire Centre.</figcaption></figure>
<figure style="margin-top:20px">{images["pr"]}<figcaption>2023 Precision–Recall curves by Fire Centre.</figcaption></figure>
</section></main></body></html>"""

    output = HERE / "RESULTS.html"
    output.write_text(add_class_balance(page, HERE), encoding="utf-8")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()

