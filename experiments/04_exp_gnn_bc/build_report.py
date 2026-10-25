"""Build a standalone visual HTML report from saved Experiment 04 outputs."""
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
spec = importlib.util.spec_from_file_location("experiment_03_report", EXPERIMENT_03 / "build_report.py")
report_03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report_03)
embedded_image, html_table, style = report_03.embedded_image, report_03.html_table, report_03.style
RULE, INK = report_03.RULE, report_03.INK
# Reference categorical slots 1-3 (validated all-pairs); marker shape repeats the identity.
SERIES = {"gnn": ("GNN (message passing)", "#2a78d6", "o"),
          "mlp": ("Same network, no message passing", "#1baf7a", "s"),
          "boosting": ("Experiment 03: gradient boosting, cell + neighbours", "#eb6834", "^")}


def ap_chart(table, order):
    fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    positions = np.arange(len(order))
    ax.scatter(table.loc[order, "rule_ap"], positions, marker="D", s=60, color=RULE, edgecolor="white", linewidth=1.5, zorder=3)
    for key, offset in zip(SERIES, (-.18, 0, .18)):
        name, colour, marker = SERIES[key]
        ax.scatter(table.loc[order, f"{key}_ap"], positions + offset, s=58, marker=marker, color=colour,
                   edgecolor="white", linewidth=1.5, zorder=4)
    handles = [Line2D([], [], linestyle="none", marker="D", markersize=7, markerfacecolor=RULE, markeredgecolor="white",
                      label="Experiment 02: selected rule")]
    handles += [Line2D([], [], linestyle="none", marker=m, markersize=8, markerfacecolor=c, markeredgecolor="white", label=n)
                for n, c, m in SERIES.values()]
    ax.set(yticks=positions, yticklabels=order, xlim=(0, 1), xlabel="2023 average precision",
           title="2023 average precision by Fire Centre")
    ax.invert_yaxis()
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    style(ax)
    fig.savefig(FIGURES / "report_average_precision.png", dpi=160)
    plt.close(fig)


def pr_curves(curves, boosting_curves, rule_points, order):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.2), sharex=True, sharey=True, constrained_layout=True)
    for ax, region in zip(axes.flat, order):
        series = {"gnn": curves.loc[curves.region.eq(region) & curves.model.eq("gnn")],
                  "mlp": curves.loc[curves.region.eq(region) & curves.model.eq("mlp")],
                  "boosting": boosting_curves.loc[boosting_curves.region.eq(region)]}
        for key, curve in series.items():
            ax.plot(curve.recall, curve.precision, color=SERIES[key][1], linewidth=2 if key == "gnn" else 1.3,
                    zorder=3 if key == "gnn" else 2)
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
    handles = [Line2D([], [], color=c, linewidth=2 if k == "gnn" else 1.3, label=n) for k, (n, c, _) in SERIES.items()]
    handles += [Line2D([], [], linestyle="none", marker="o", markersize=6, markerfacecolor=RULE, markeredgecolor="white",
                       label="Same-cell rules (Experiment 02)"),
                Line2D([], [], linestyle="none", marker="^", markersize=6, markerfacecolor=RULE, markeredgecolor="white",
                       label="Cell + neighbour rules (Experiment 02)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("2023 precision-recall curves by Fire Centre; each gray point is one fixed rule")
    fig.savefig(FIGURES / "report_pr_curves.png", dpi=150)
    plt.close(fig)


def training_chart(history, best_epochs):
    fig, ax = plt.subplots(figsize=(11, 4.2), constrained_layout=True)
    for key in ("gnn", "mlp"):
        run = history.loc[history.network.eq(key)]
        name, colour, marker = SERIES[key]
        ax.plot(run.epoch, run.selection_mean_region_ap, color=colour, linewidth=2, marker=marker, markersize=6,
                markeredgecolor="white", label=name)
        best = run.loc[run.epoch.eq(best_epochs[key])].iloc[0]
        ax.annotate(f"kept: epoch {int(best.epoch)}, {best.selection_mean_region_ap:.3f}", (best.epoch, best.selection_mean_region_ap),
                    xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9, color=INK)
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
    # True negatives are not stored in the combined table; every entry covers the centre's full 2023 rows.
    rows = pd.read_csv(RESULTS / "metrics_by_year.csv").query(f"year == {manifest['evaluation_year']}") \
        .drop_duplicates("region").set_index("region").rows
    comparison = report_03.with_accuracy(comparison.assign(
        tn=comparison.region.map(rows) - comparison.tp - comparison.fp - comparison.fn))
    curves = pd.read_csv(RESULTS / "pr_curves_2023.csv")
    history = pd.read_csv(RESULTS / "training_history.csv")
    boosting_curves = pd.read_csv(EXPERIMENT_03 / "results/pr_curves_2023.csv")
    boosting_curves = boosting_curves.loc[boosting_curves.model.eq("gradient_boosting")
                                          & boosting_curves.feature_set.eq("cell_plus_neighbours")]
    rules = pd.read_csv(HERE.parent / "02_hotspot_persistence_bc/results/all_region_metrics.csv")
    rule_points = rules.loc[rules.year.eq(manifest["evaluation_year"]) & rules.rule.ne("always_no_detection")]
    best_epochs = manifest["training"]["best_epochs"]

    pick = lambda source, model, feature_set=None: comparison.loc[
        comparison.source.eq(source) & comparison.model.eq(model)
        & (comparison.feature_set.eq(feature_set) if feature_set else True)].set_index("region")
    parts = {"rule": pick("02_rules", "selected_rule"), "boosting": pick("03_learned_baselines", "gradient_boosting", "cell_plus_neighbours"),
             "mlp": pick("04_gnn", "mlp"), "gnn": pick("04_gnn", "gnn")}
    table = pd.concat({f"{key}_{metric}": frame[column] for key, frame in parts.items()
                       for metric, column in (("ap", "average_precision"), ("f1", "f1"), ("precision", "precision"),
                                              ("recall", "recall"), ("accuracy", "accuracy"),
                                              ("always_no", "always_no_accuracy"), ("balanced", "balanced_accuracy"))},
                      axis=1)
    order = table.gnn_ap.sort_values(ascending=False).index.tolist()
    ap_chart(table, order)
    pr_curves(curves, boosting_curves, rule_points, order)
    training_chart(history, best_epochs)

    graph_gain = table.gnn_ap - table.mlp_ap
    vs_boosting = table.gnn_ap - table.boosting_ap
    vs_rule_f1 = table.gnn_f1 - table.rule_f1
    summary = html_table(pd.DataFrame({
        "Fire Centre": order, "Rule AP": table.loc[order, "rule_ap"].to_numpy(),
        "Boosting AP (03)": table.loc[order, "boosting_ap"].to_numpy(), "MLP AP": table.loc[order, "mlp_ap"].to_numpy(),
        "GNN AP": table.loc[order, "gnn_ap"].to_numpy(), "Rule F1": table.loc[order, "rule_f1"].to_numpy(),
        "Boosting F1 (03)": table.loc[order, "boosting_f1"].to_numpy(), "MLP F1": table.loc[order, "mlp_f1"].to_numpy(),
        "GNN F1": table.loc[order, "gnn_f1"].to_numpy(), "GNN precision": table.loc[order, "gnn_precision"].to_numpy(),
        "GNN recall": table.loc[order, "gnn_recall"].to_numpy(),
        "Always-no accuracy": table.loc[order, "gnn_always_no"].to_numpy(),
        "GNN accuracy": table.loc[order, "gnn_accuracy"].to_numpy(),
        "Rule balanced accuracy": table.loc[order, "rule_balanced"].to_numpy(),
        "Boosting balanced accuracy (03)": table.loc[order, "boosting_balanced"].to_numpy(),
        "MLP balanced accuracy": table.loc[order, "mlp_balanced"].to_numpy(),
        "GNN balanced accuracy": table.loc[order, "gnn_balanced"].to_numpy()}),
        {**{c: "{:.3f}" for c in ("Rule AP", "Boosting AP (03)", "MLP AP", "GNN AP", "Rule F1", "Boosting F1 (03)",
                                  "MLP F1", "GNN F1", "Rule balanced accuracy", "Boosting balanced accuracy (03)",
                                  "MLP balanced accuracy", "GNN balanced accuracy")},
         "GNN precision": "{:.1%}", "GNN recall": "{:.1%}", "Always-no accuracy": "{:.3%}", "GNN accuracy": "{:.3%}"})
    everything = comparison.assign(name=comparison.source + " · " + comparison.model + comparison.feature_set.map(
        lambda s: "" if pd.isna(s) else f" · {s}"))
    full = html_table(everything.sort_values(["region", "source", "model"])[
        ["region", "name", "average_precision", "precision", "recall", "f1", "accuracy", "balanced_accuracy"]].rename(columns={
            "region": "Fire Centre", "name": "Source · model", "average_precision": "Average precision",
            "precision": "Precision", "recall": "Recall", "f1": "F1", "accuracy": "Accuracy",
            "balanced_accuracy": "Balanced accuracy"}),
        {"Average precision": "{:.3f}", "Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.3f}",
         "Accuracy": "{:.3%}", "Balanced accuracy": "{:.3f}"})
    images = {"ap": embedded_image(FIGURES / "report_average_precision.png", "2023 average precision by Fire Centre"),
              "pr": embedded_image(FIGURES / "report_pr_curves.png", "2023 precision-recall curves by Fire Centre"),
              "training": embedded_image(FIGURES / "report_training.png", "2022 mean AP by training epoch")}
    graph_word = "helps" if graph_gain.median() > 0 else "does not help"
    boosting_word = "edges out" if (vs_boosting > 0).sum() > 3 else "does not beat"
    tied = vs_boosting.index[vs_boosting.abs() < .005].tolist()
    tied_text = f"; {' and '.join(tied)} are within 0.005" if tied else ""
    boosting_vs_rule_f1 = table.boosting_f1 - table.rule_f1

    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BC | GNN experiment</title>
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
.timeline{{display:grid;grid-template-columns:1fr 45px 1fr 45px 1fr;align-items:center;gap:8px;margin:18px 0}}
.step{{padding:20px;border:1px solid var(--line);border-radius:12px;background:white}}.arrow{{font-size:30px;text-align:center}}
details{{margin-top:18px}}summary{{cursor:pointer;font-weight:650;padding:12px;background:#e9efec;border-radius:8px}}
.conclusion{{background:#e9f0ed;border-radius:12px;padding:24px}}
.table{{overflow-x:auto;background:white;border:1px solid var(--line);border-radius:12px;margin:16px 0}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:8px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}th{{color:var(--muted);font-weight:650}}
td{{font-variant-numeric:tabular-nums}}tbody tr:last-child td{{border-bottom:none}}
footer{{color:var(--muted);font-size:13px;padding:20px 0}}@media(max-width:800px){{.columns,.cards,.timeline{{grid-template-columns:1fr}}.arrow{{transform:rotate(90deg)}}main{{padding:24px 16px}}}}
@media print{{body{{background:white}}main{{padding:0}}figure,.card{{break-inside:avoid}}}}
</style></head><body><main>
<header><div class="eyebrow">Experiment 04 · Graph neural network</div>
<h1>Does message passing beat hand-built neighbour features?</h1>
<p class="purpose"><strong>Purpose:</strong> planned step 3. Train a standard GNN on daily Fire Centre graphs and compare it with the same network without the graph, the best Experiment 03 model, and the persistence rule.</p>
<p class="muted">Six Fire Centres · {manifest["graph"]["nodes"]:,} nodes · same rows, target and split as Experiments 02–03 · train 2018–2021 · epochs and thresholds 2022 · evaluate 2023 · 2024 excluded</p></header>

<section><h2>01 / Setup: one graph per day</h2>
<div class="timeline"><div class="step"><div class="eyebrow">Graph · Each target day</div><h3>Cells are nodes</h3>{manifest["graph"]["nodes"]:,} grid cells; edges join H3 neighbours in the same Fire Centre ({manifest["graph"]["directed_edges"]:,} directed).</div><div class="arrow">→</div>
<div class="step"><div class="eyebrow">Model · Trained on 2018–2021</div><h3>2 message-passing layers</h3>Each cell starts from its own history only, then pools its neighbours' states twice (2 hops).</div><div class="arrow">→</div>
<div class="step"><div class="eyebrow">Output · Next day</div><h3>Probability per cell</h3>Same metrics and 2022 thresholds as Experiment 03.</div></div>
<ul><li><strong>Node inputs:</strong> the cell's own past 1/3/7-day counts, days since its last detection, no-history and quiet flags, day of year, and neighbour count. No neighbour activity is given directly.</li>
<li><strong>Control:</strong> the identical network with message passing removed. GNN minus control = what the graph adds.</li>
<li><strong>Training:</strong> one daily snapshot per step; all positives and active cells plus a fresh 1% of quiet cells in each loss, with the same exact sampling correction as Experiment 03. The epoch with the best 2022 mean per-centre AP is kept.</li></ul>
<figure>{images["training"]}<figcaption>Mean per-centre average precision on 2022 after each epoch. 2023 played no part in choosing epochs or thresholds.</figcaption></figure></section>

<section><h2>02 / Results</h2>
<div class="cards"><div class="card"><h3>GNN vs no message passing</h3><div class="value">{graph_gain.median():+.3f}</div><p>Median AP change; GNN better in {int((graph_gain > 0).sum())} of 6 centres.</p></div>
<div class="card"><h3>GNN vs boosting (03)</h3><div class="value">{vs_boosting.median():+.3f}</div><p>Median AP change against gradient boosting with hand-built neighbour counts; GNN better in {int((vs_boosting > 0).sum())} of 6.</p></div>
<div class="card"><h3>GNN F1 vs rule</h3><div class="value">{vs_rule_f1.median():+.3f}</div><p>Median F1 change at 2022 thresholds; GNN better in {int((vs_rule_f1 > 0).sum())} of 6.</p></div></div>
{summary}
{report_03.accuracy_note(pd.DataFrame({"always_no_accuracy": table.gnn_always_no}))}
<figure>{images["ap"]}<figcaption>Gray diamonds are the Experiment 02 rule chosen on 2022. Colour and marker identify each model.</figcaption></figure>
<figure style="margin-top:20px">{images["pr"]}<figcaption>Curves show every precision–recall trade-off a model offers; gray points are fixed rules. Where a point sits above a curve, that rule beats the model at that operating point.</figcaption></figure>
<details><summary>Show every model and rule for every Fire Centre (2023)</summary>{full}</details></section>

<section><div class="conclusion"><h2>03 / Conclusion</h2><ul>
<li><strong>Message passing {graph_word}:</strong> learned neighbour pooling changes AP by a median of {graph_gain.median():+.3f} over the same network without it.</li>
<li><strong>The GNN {boosting_word} hand-built neighbour features</strong> (Experiment 03 gradient boosting): median AP change {vs_boosting.median():+.3f}, higher in {int((vs_boosting > 0).sum())} of 6 centres{tied_text}.</li>
<li><strong>As an alert rule, it nearly matches "active yesterday".</strong> GNN F1 is a median {vs_rule_f1.median():+.3f} from the selected rule (above it in {int((vs_rule_f1 > 0).sum())} of 6), where Experiment 03 boosting was {boosting_vs_rule_f1.median():+.3f}. Its precision–recall curve passes through the rule's point instead of below it.</li>
<li>All inputs are still fire history and date. Weather, FRP and coverage are the next additions, and the graph foundation model (step 4) must clear these same bars.</li></ul></div>
<ul class="muted"><li>Targets are <strong>recorded detections</strong>, not ignitions, unique fires or confirmed absence. Coverage remains unknown.</li>
<li>Untuned: one architecture, learning rate and epoch budget; a single seed. Small AP differences between models are within run-to-run noise and should not be over-read.</li>
<li>Neighbours outside each Fire Centre are excluded, as in Experiments 02–03. 2023 is a development evaluation; 2024 remains excluded.</li></ul></section>
<footer>Generated by build_report.py from saved Experiment 04 results, with Experiment 02 rules and Experiment 03 models for comparison. All figures are embedded: this report works offline as one HTML file.</footer>
</main></body></html>'''
    output = HERE / "RESULTS.html"
    output.write_text(add_class_balance(page, HERE), encoding="utf-8")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
