"""Build a standalone, chart-first Result.html from Experiment 08 outputs."""
from __future__ import annotations

import argparse
import base64
from html import escape
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
GROUPS = ["local_active", "neighbour_only", "quiet"]
LABELS = {"local_active": "Local activity", "neighbour_only": "Neighbours only", "quiet": "Quiet"}
COLORS = ["#168b82", "#e7a33e", "#d56067"]


def embed(fig):
    stream = BytesIO()
    fig.savefig(stream, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return '<img alt="Diagnostic chart" src="data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode() + '">'


def table(frame):
    return '<div class="scroll">' + frame.to_html(index=False, border=0, escape=True) + '</div>'


def quiet_sections(directory):
    if not (directory / "quiet_budget_summary.csv").exists():
        return []
    summary = pd.read_csv(directory / "quiet_score_summary.csv")
    histogram = pd.read_csv(directory / "quiet_score_histograms.csv")
    budgets = pd.read_csv(directory / "quiet_budget_summary.csv")
    sections = ['<h2>Quiet cells · Ranking versus threshold</h2>',
                '<p class="note">Budgets apply to quiet cells only, per centre per day. All-centre totals use six separate budgets. Cutoff ties use expected random selection; fractional counts are expectations.</p>']
    for variant, scores in summary.groupby("variant", sort=False):
        hist = histogram.loc[histogram.variant.eq(variant)].groupby(["label", "log10_lower", "log10_upper"])["count"].sum().reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
        for label, color, name in [(0, COLORS[0], "No recorded detection"), (1, COLORS[2], "Recorded detection")]:
            values = hist.loc[hist.label.eq(label)]
            axes[0].stairs(values["count"] / values["count"].sum() * 100,
                           np.r_[values.log10_lower, values.log10_upper.iloc[-1]], color=color, label=name, linewidth=2)
        lower, upper = scores.threshold.min(), scores.threshold.max()
        axes[0].axvspan(np.log10(max(lower, 1e-12)), np.log10(max(upper, 1e-12)), color="#596674", alpha=.18, label="Regional threshold range")
        axes[0].set(title=f"{variant.upper()} · Quiet-cell score distributions", xlabel="log10(corrected probability)", ylabel="Share within each label (%)")
        axes[0].legend(frameon=False, fontsize=8)
        positive = scores.loc[scores.label.eq(1)]
        x = np.arange(len(positive))
        for values, name, marker, color in [(positive["median"], "Positive median", "o", COLORS[0]),
                                            (positive["max"], "Positive maximum", "^", COLORS[2]),
                                            (positive.threshold, "Selected threshold", "x", "#243746")]:
            axes[1].scatter(x, np.maximum(values, 1e-12), label=name, marker=marker, color=color)
        axes[1].set(yscale="log", xticks=x, xticklabels=positive.region, title="Positive scores versus selected thresholds", ylabel="Corrected probability (log scale)")
        axes[1].tick_params(axis="x", labelrotation=30)
        axes[1].legend(frameon=False, fontsize=8)
        sections.append(embed(fig))
        sections.append('<p class="note">Each label distribution sums to 100%; class sizes differ greatly. Plot values below 10⁻¹² are clipped for display only. Scores and ranking retain their original values.</p>')
        overall = budgets.loc[budgets.variant.eq(variant) & budgets.region.eq("All centres")].sort_values("budget")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
        axes[0].plot(overall.budget, overall.recall * 100, "o-", color=COLORS[0], label="Model ranking")
        axes[0].plot(overall.budget, overall.random_recall * 100, "o--", color="#73838b", label="Random within each region-day")
        axes[0].set(xscale="log", ylabel="Quiet-positive recall (%)", xlabel="Quiet alerts / centre / day", title="Recall at fixed daily budgets")
        axes[0].legend(frameon=False)
        axes[1].plot(overall.budget, overall.precision * 100, "o-", color=COLORS[2])
        axes[1].set(xscale="log", ylabel="Expected precision (%)", xlabel="Quiet alerts / centre / day", title="Cost of detecting quiet positives")
        for ax in axes:
            ax.set_xticks(overall.budget, labels=overall.budget.astype(str))
            ax.grid(alpha=.2)
        sections.append(embed(fig))
        display = overall[["budget", "alerts", "expected_tp", "expected_fp", "recall", "precision", "random_recall", "lift_over_random"]].copy()
        for col in ["recall", "precision", "random_recall"]:
            display[col] = display[col].map(lambda v: f"{v:.3%}")
        for col in ["expected_tp", "expected_fp", "lift_over_random"]:
            display[col] = display[col].map(lambda v: f"{v:,.2f}")
        sections.append(table(display.rename(columns={"budget": "Alerts / centre / day", "alerts": "Season alerts", "expected_tp": "Expected TP", "expected_fp": "Expected FP", "recall": "Recall", "precision": "Precision", "random_recall": "Random recall", "lift_over_random": "Lift vs random"})))
        regional = budgets.loc[budgets.variant.eq(variant) & budgets.region.ne("All centres")].pivot(index="region", columns="budget", values="recall")
        regional = regional.map(lambda v: f"{v:.2%}").reset_index()
        sections.append('<h3>Quiet recall by centre and daily budget</h3>' + table(regional))
        diagnostic = positive[["region", "rows", "median", "p90", "max", "threshold", "above_threshold"]].copy()
        for col in ["median", "p90", "max", "threshold"]:
            diagnostic[col] = diagnostic[col].map(lambda v: f"{v:.3g}")
        sections.append('<details><summary>Quiet-positive scores and existing threshold</summary>' + table(diagnostic) + '</details>')
    return sections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "results/baseline")
    parser.add_argument("--output", type=Path, default=HERE / "Result.html")
    args = parser.parse_args()
    metrics = pd.read_csv(args.input / "activity_metrics.csv")
    cases = pd.read_csv(args.input / "error_review.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold"})
    sections = []
    for variant, data in metrics.groupby("variant", sort=False):
        grouped = data.loc[data.activity_group.ne("all")]
        pooled = grouped.groupby("activity_group")[["rows", "positives", "tp", "fp", "fn"]].sum().reindex(GROUPS)
        pooled["recall"] = pooled.tp / pooled.positives
        pooled["precision"] = pooled.tp / (pooled.tp + pooled.fp)
        pooled["f1"] = 2 * pooled.tp / (2 * pooled.tp + pooled.fp + pooled.fn)
        sections.append(f'<h2>{escape(variant.upper())} · Where predictions succeed</h2>')
        cards = ''.join(f'<div class="card"><span>{LABELS[g]} recall</span><strong>{pooled.loc[g,"recall"]:.1%}</strong><small>{int(pooled.loc[g,"tp"]):,} / {int(pooled.loc[g,"positives"]):,} positives found</small></div>' for g in GROUPS)
        sections.append('<div class="cards">' + cards + '</div>')

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
        names = [LABELS[g] for g in GROUPS]
        x = np.arange(3)
        axes[0].bar(x, pooled.tp, color=COLORS[0], label="Found (TP)")
        axes[0].bar(x, pooled.fn, bottom=pooled.tp, color=COLORS[2], label="Missed (FN)")
        axes[0].set(xticks=x, xticklabels=names, ylabel="Positive region-cell-days", title="Recorded positives: found vs missed")
        axes[0].legend(frameon=False)
        axes[1].barh(names, pooled.fn / pooled.fn.sum() * 100, color=COLORS)
        axes[1].set(xlabel="Share of all false negatives (%)", title="Where missed detections occur", xlim=(0, 100))
        for i, value in enumerate(pooled.fn / pooled.fn.sum() * 100):
            axes[1].text(value + 1, i, f"{value:.1f}%", va="center")
        sections.append(embed(fig))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
        for ax, metric, title in zip(axes, ["recall", "ap"], ["Recall at 2022-selected thresholds", "Average precision within each activity group"]):
            pivot = grouped.pivot(index="region", columns="activity_group", values=metric).reindex(columns=GROUPS)
            values = pivot.to_numpy()
            ax.imshow(values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
            ax.set(xticks=range(3), xticklabels=names, yticks=range(len(pivot)), yticklabels=pivot.index, title=title)
            for row in range(len(pivot)):
                for col in range(3):
                    value = values[row, col]
                    label = f"{value:.1%}" if metric == "recall" else f"{value:.4f}"
                    ax.text(col, row, label, ha="center", va="center", color="white" if value > .55 else "#152c37")
        sections.append(embed(fig))
        sections.append('<p class="note">AP depends on prevalence; compare models within the same group.</p>')

        display = pooled.reset_index().rename(columns={"activity_group": "Activity", "rows": "Cell-days", "positives": "Positives", "tp": "TP", "fp": "FP", "fn": "FN", "recall": "Recall", "precision": "Precision", "f1": "F1"})
        display["Activity"] = display.Activity.map(LABELS)
        for col in ["Cell-days", "Positives", "TP", "FP", "FN"]:
            display[col] = display[col].map(lambda v: f"{v:,}")
        for col in ["Recall", "Precision", "F1"]:
            display[col] = display[col].map(lambda v: f"{v:.1%}")
        sections.append(table(display))

        overall = data.loc[data.activity_group.eq("all")].copy()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        x = np.arange(len(overall))
        for offset, metric, color in [(-.23, "precision", COLORS[0]), (0, "recall", COLORS[1]), (.23, "f1", COLORS[2])]:
            axes[0].bar(x + offset, overall[metric], width=.23, color=color, label=metric.title())
        axes[0].set(xticks=x, xticklabels=overall.region, ylim=(0, 1), title="Regional operating performance")
        axes[0].tick_params(axis="x", labelrotation=30)
        axes[0].legend(frameon=False, ncol=3)
        axes[1].bar(overall.region, overall.ap, color=COLORS[0])
        axes[1].set(ylim=(0, 1), title="Regional average precision")
        axes[1].tick_params(axis="x", labelrotation=30)
        sections.append(embed(fig))
        selected = overall[["region", "ap", "precision", "recall", "f1", "threshold", "tp", "fp", "fn"]].copy()
        for col in ["ap", "precision", "recall", "f1", "threshold"]:
            selected[col] = selected[col].map(lambda v: f"{v:.4f}")
        sections.append(table(selected.rename(columns={"region": "Region", "ap": "AP", "precision": "Precision", "recall": "Recall", "f1": "F1", "threshold": "Threshold", "tp": "TP", "fp": "FP", "fn": "FN"})))

    sections.extend(quiet_sections(args.input))
    if set(metrics.variant) == {"firms", "cems"}:
        sections.append('<h2>CEMS minus FIRMS</h2>')
        delta = metrics.pivot(index=["region", "activity_group"], columns="variant", values="ap")
        delta["AP change"] = delta.cems - delta.firms
        sections.append(table(delta.reset_index().round(4)))

    sections.append('<h2>Error review · 200 sampled cases</h2>')
    counts = pd.crosstab([cases.region, cases.activity_group], cases.error).reset_index()
    counts["activity_group"] = counts.activity_group.map(LABELS)
    sections.append(table(counts.rename(columns={"region": "Region", "activity_group": "Activity"})))
    sections.append('<p class="note">Stratified review sample, not population error rates. Black outline: target cell. Future dates: retrospective review only.</p>')
    # Every map is embedded so the HTML remains portable without an asset directory.
    for error in ["FN", "FP"]:
        for group in GROUPS:
            subset = cases.loc[cases.error.eq(error) & cases.activity_group.eq(group)]
            if subset.empty:
                continue
            sections.append(f'<details><summary>{error} · {LABELS[group]} · {len(subset)} cases</summary>')
            for index, row in subset.iterrows():
                path = args.input / f"case_{index:03d}.png"
                if not path.exists():
                    raise FileNotFoundError(f"Missing review map: {path}; run diagnostics with --maps")
                encoded = base64.b64encode(path.read_bytes()).decode()
                sections.append(f'<figure><figcaption>#{index:03d} · {escape(row.region)} · {escape(str(row.target_date))} · score {row.score:.5f} / threshold {row.threshold:.5f}</figcaption><img loading="lazy" alt="Five-day hotspot neighbourhood" src="data:image/png;base64,{encoded}"></figure>')
            sections.append('</details>')
    status = "Completed" if "cems" in set(metrics.variant) else "Pending: complete daily CEMS weather grid missing"
    setup = pd.DataFrame([
        ["Model", "Gradient boosting · local + neighbour features"],
        ["Training / threshold / evaluation", "2018–2021 / 2022 / 2023"],
        ["2024 evaluation", "Not used"], ["CEMS comparison", status],
        ["Target", "Recorded hotspot; zero does not confirm fire absence"],
        ["Activity window", "Previous seven days"],
        ["Spatial counts", "Region-cell-days; boundary cells may repeat across regions"],
    ], columns=["Setting", "Value"])
    style = 'body{font:15px system-ui,sans-serif;color:#20333c;background:#f3f6f7;margin:0}main{max-width:1180px;margin:auto;padding:32px}h1{font-size:34px;margin:8px 0}h2{margin-top:38px}.tag{color:#168b82;font-weight:700;text-transform:uppercase;letter-spacing:2px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.card,details{background:white;border:1px solid #dce5e8;border-radius:12px;padding:20px}.card span,.card small{display:block}.card strong{display:block;font-size:40px;margin:12px 0}img{width:100%;height:auto;background:white;border-radius:10px;margin:14px 0}table{width:100%;border-collapse:collapse;background:white;font-size:13px}th,td{padding:11px;text-align:right;border-bottom:1px solid #e1e7e9}th:first-child,td:first-child{text-align:left}th{background:#e7eff1}details{margin:12px 0}summary{cursor:pointer;font-weight:650}.scroll{overflow:auto}.note,figcaption{color:#60727b;font-size:13px}figure{margin:22px 0}@media(max-width:700px){main{padding:16px}.cards{grid-template-columns:1fr}}'
    document = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Experiment 08 · Results</title><style>' + style + '</style></head><body><main><div class="tag">Experiment 08 · 2023 evaluation</div><h1>Where hotspot forecasts miss</h1>' + table(setup) + ''.join(sections) + '</main></body></html>'
    args.output.write_text(document, encoding="utf-8")
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
