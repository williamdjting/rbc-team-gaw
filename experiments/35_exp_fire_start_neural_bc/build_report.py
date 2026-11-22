"""Build a standalone visual HTML report from saved Experiment 35 outputs.

    python build_report.py                 # results/ -> RESULTS.html
    python build_report.py <results dir>   # another results folder (e.g. a smoke test)
"""
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
FIGURES = HERE / "figures"
EXPERIMENT_20 = HERE.parent / "34_exp_fire_start_bc/results"
spec = importlib.util.spec_from_file_location("experiment_03_report", HERE.parent / "03_exp_learned_baselines_bc/build_report.py")
report_03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report_03)
embedded_image, html_table, style = report_03.embedded_image, report_03.html_table, report_03.style
RULE = report_03.RULE
# Reference categorical slots 1-5 for the models; grays for simple rankings.
SERIES = {"start_mlp": ("Fire-start MLP", "#eb6834", "s"),
          "start_gru": ("Fire-start MLP + GRU", "#eda100", "^"),
          "start_drivers": ("Fire-start CatBoost, Experiment 34", "#1baf7a", "o"),
          "catboost_spread_noday": ("Next-day CatBoost, Experiment 21", "#2a78d6", "D"),
          "mlp": ("Next-day MLP, Experiment 22", "#8a5cc2", "h"),
          "nearby_activity": ("Ranking: fire nearby in 3 days", RULE, "v"),
          "lightning": ("Ranking: lightning in 3 days", "#b5b3aa", "P"),
          "random": ("Random quiet cells", "#d6d4cc", "x")}
NETS = ("start_mlp", "start_gru")
BASELINE_NAMES = {"start_drivers": "Experiment 34 fire-start CatBoost", "catboost_spread_noday": "Experiment 21 next-day CatBoost"}
DEFINITION_NAMES = {"res7_exact": "exact cell", "res7_margin1": "one-cell margin"}
GROUP_NAMES = {"new BCWS wildfire": "New BCWS wildfire", "older BCWS wildfire, not recently detected": "Older wildfire flaring up",
               "no BCWS record": "No BCWS record"}  # "other BCWS record" has a single 2023 fire start


def watch_chart(pooled):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for ax, column, title in zip(axes, ("share_caught_next_day", "share_caught_within_7d"),
                                 ("Listed on the start day", "Listed at least once in the 7 days up to the start")):
        for name, (label, colour, marker) in SERIES.items():
            part = pooled.loc[pooled.model.eq(name)].sort_values("list_size")
            ax.plot(part.list_size, part[column], color=colour, marker=marker, markersize=6, linewidth=2,
                    linestyle="--" if name == "random" else "-")
        ax.set(xscale="log", xticks=[10, 25, 50, 100], xticklabels=["10", "25", "50", "100"], ylim=(0, None),
               xlabel="Quiet cells listed per Fire Centre per day", ylabel="Share of 2023 fire starts", title=title)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:g}%"))
        style(ax)
    handles = [Line2D([], [], color=c, marker=m, markersize=7, linewidth=2, label=n) for n, c, m in SERIES.values()]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    fig.savefig(FIGURES / "report_watch.png", dpi=150)
    plt.close(fig)


def ap_chart(evaluation, coverage, order):
    fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    positions = np.arange(len(order))
    names = [n for n in SERIES if n != "random"]
    offsets = np.linspace(-.3, .3, len(names))
    base = coverage.set_index("region").loc[order, "base_rate"].to_numpy()
    for name, offset in zip(names, offsets):
        label, colour, marker = SERIES[name]
        rows = evaluation.loc[evaluation.definition.eq("res7_exact") & evaluation.model.eq(name)].set_index("region")
        ax.scatter(rows.loc[order, "average_precision"] / base, positions + offset, s=46, marker=marker, color=colour,
                   edgecolor="white", linewidth=1.1, zorder=3, label=label)
    ax.axvline(1, color="#8a8a82", linewidth=1, linestyle="--")
    ax.set(xscale="log", yticks=positions, yticklabels=order, xlabel="Average precision ÷ base rate (1 = random)",
           title="How much better than random, 2023")
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax)
    fig.savefig(FIGURES / "report_ap.png", dpi=150)
    plt.close(fig)


def curves_chart(curves):
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    for name in NETS:
        label, colour, marker = SERIES[name]
        part = curves.loc[curves.model.eq(name)]
        ax.plot(part.epoch, part.train_log_loss, color=colour, linestyle="--", linewidth=1.3, alpha=.6)
        ax.plot(part.epoch, part.valid_log_loss, color=colour, marker=marker, markersize=5, linewidth=2, label=label)
    ax.set(xlabel="Epoch", ylabel="Log loss", title="Training (dashed) and held-out (solid) loss per epoch")
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    fig.savefig(FIGURES / "report_curves.png", dpi=150)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    assert not manifest["holdout_loaded"]
    regions = manifest["regions"]
    metrics = pd.read_csv(RESULTS / "metrics_by_year.csv")
    summary = pd.read_csv(RESULTS / "summary_2023.csv").set_index(["definition", "model"])
    coverage = pd.read_csv(RESULTS / "coverage.csv")
    pooled = pd.read_csv(RESULTS / "watch_list_bc_2023.csv")
    groups = pd.read_csv(RESULTS / "onset_groups_2023.csv")
    boot = pd.read_csv(RESULTS / "bootstrap_2023.csv")
    fits = pd.read_csv(RESULTS / "fit_checks.csv")
    sample = pd.read_csv(EXPERIMENT_20 / "training_sample.csv")  # same rows; Experiment 34 also counted quiet cell-days
    curves = pd.read_csv(RESULTS / "training_curves.csv")
    evaluation = metrics.loc[metrics.year.eq(manifest["evaluation_year"])]
    cov23 = coverage.loc[coverage.year.eq(manifest["evaluation_year"])]
    order = cov23.sort_values("fire_starts", ascending=False).region.tolist()
    watch_chart(pooled)
    ap_chart(evaluation, cov23, order)
    curves_chart(curves)
    s = lambda d, m, col="bc_wide_f1": summary.loc[(d, m), col]
    w = pooled.set_index(["model", "list_size"])

    cov_rows = []
    for year, part in coverage.groupby("year"):
        cov_rows.append({"Year": int(year), "Quiet cell-days": int(part.quiet_cell_days.sum()), "Fire starts": int(part.fire_starts.sum()),
                         "Label positives": int(part.label_positives.sum()),
                         "Starts as share of positives": part.fire_starts.sum() / part.label_positives.sum(),
                         "Base rate (starts per quiet cell-day)": part.fire_starts.sum() / part.quiet_cell_days.sum()})
    for part in [sample]:
        cov_rows.insert(0, {"Year": "2018–2021 (training)", "Quiet cell-days": int(part.quiet_rows.sum()),
                            "Fire starts": int(part.fire_starts.sum()), "Label positives": np.nan,
                            "Starts as share of positives": np.nan,
                            "Base rate (starts per quiet cell-day)": part.fire_starts.sum() / part.quiet_rows.sum()})
    coverage_table = html_table(pd.DataFrame(cov_rows), {"Quiet cell-days": "{:,}", "Fire starts": "{:,}", "Label positives": "{:,.0f}",
                                                         "Starts as share of positives": "{:.1%}",
                                                         "Base rate (starts per quiet cell-day)": "{:.5f}"})

    base_rate = cov23.fire_starts.sum() / cov23.quiet_cell_days.sum()
    rows = []
    for name in SERIES:
        if name == "random":
            continue
        rows.append({"Forecast": SERIES[name][0], "F1, exact cell": s("res7_exact", name), "F1, one-cell margin": s("res7_margin1", name),
                     "Precision (exact)": s("res7_exact", name, "bc_wide_precision"), "Recall (exact)": s("res7_exact", name, "bc_wide_recall"),
                     "Median AP": s("res7_exact", name, "median_average_precision"),
                     "Top 50: on the day": w.loc[(name, 50), "share_caught_next_day"],
                     "Top 50: within 7 days": w.loc[(name, 50), "share_caught_within_7d"]})
    rows.append({"Forecast": SERIES["random"][0], "F1, exact cell": np.nan, "F1, one-cell margin": np.nan,
                 "Precision (exact)": base_rate, "Recall (exact)": np.nan, "Median AP": cov23.base_rate.median(),
                 "Top 50: on the day": w.loc[("random", 50), "share_caught_next_day"],
                 "Top 50: within 7 days": w.loc[("random", 50), "share_caught_within_7d"]})
    summary_table = html_table(pd.DataFrame(rows), {"F1, exact cell": "{:.3f}", "F1, one-cell margin": "{:.3f}", "Precision (exact)": "{:.3f}",
                                                    "Recall (exact)": "{:.3f}", "Median AP": "{:.4f}", "Top 50: on the day": "{:.1%}",
                                                    "Top 50: within 7 days": "{:.1%}"})

    watch_rows = []
    for name in SERIES:
        for n in (10, 25, 50, 100):
            r = w.loc[(name, n)]
            watch_rows.append({"Forecast": SERIES[name][0], "Cells per centre per day": n, "On the day": r.share_caught_next_day,
                               "On the day, cell or neighbour": r.share_caught_next_day_margin, "Within 7 days": r.share_caught_within_7d,
                               "Listed cells that start next day": r.hit_rate_next_day,
                               "Listed cells that start within 7 days": r.hit_rate_within_7d})
    watch_table = html_table(pd.DataFrame(watch_rows), {c: "{:.1%}" for c in watch_rows[0] if c not in ("Forecast", "Cells per centre per day")})

    g = groups.groupby(["measure", "model", "onset_group"])[["fire_starts", "caught"]].sum().reset_index()
    g["share"] = g.caught / g.fire_starts
    group_rows = []
    for name in [n for n in SERIES if n != "random"]:
        row = {"Forecast": SERIES[name][0]}
        for measure, label in (("f1_2022_res7_margin1", "flagged (margin)"), ("top50_within_7d", "top 50 in 7 days")):
            for group, group_label in GROUP_NAMES.items():
                hit = g.loc[g.measure.eq(measure) & g.model.eq(name) & g.onset_group.eq(group)]
                if len(hit):
                    row[f"{group_label}, {label}"] = float(hit.share.iloc[0])
        group_rows.append(row)
    counts = g.loc[g.measure.eq("top50_within_7d") & g.model.eq("start_mlp")].set_index("onset_group").fire_starts.reindex(list(GROUP_NAMES))
    group_table = html_table(pd.DataFrame(group_rows), {c: "{:.1%}" for c in group_rows[0] if c != "Forecast"})

    def boot_table(baseline):
        b = boot.loc[boot.scope.eq("BC-wide") & boot.baseline.eq(baseline)].copy()
        b["Forecast"] = b.model.map(lambda m: SERIES[m][0])
        b["Definition"] = b.definition.map(DEFINITION_NAMES)
        b["95% interval"] = [f"{lo:+.3f} to {hi:+.3f}" for lo, hi in zip(b.ci_low, b.ci_high)]
        label = f"F1 minus {BASELINE_NAMES[baseline]}"
        return html_table(b[["Definition", "Forecast", "difference", "95% interval", "share_above"]].rename(
            columns={"difference": label, "share_above": "Share of resamples above 0"}),
            {label: "{:+.3f}", "Share of resamples above 0": "{:.0%}"})

    columns = {f"{SERIES[m][0]} ({DEFINITION_NAMES[d]})": evaluation.loc[evaluation.definition.eq(d) & evaluation.model.eq(m)].set_index(
        "region").loc[order, "f1"].to_numpy() for d in DEFINITION_NAMES for m in SERIES if m != "random"}
    per_centre = html_table(pd.DataFrame({"Fire Centre": order, "2023 fire starts": cov23.set_index("region").loc[order, "fire_starts"].to_numpy(),
                                          **columns}), {"2023 fire starts": "{:,}", **{c: "{:.3f}" for c in columns}})
    fit_table = html_table(fits.assign(model=fits.model.map(lambda m: SERIES[m][0]))[
        ["model", "rows", "positives", "valid_positives", "inputs", "parameters", "best_epoch", "epochs_run", "valid_log_loss", "fit_seconds"]].rename(
        columns={"model": "Model", "rows": "Training rows", "positives": "Fire starts", "valid_positives": "Held-out fire starts",
                 "inputs": "Inputs", "parameters": "Parameters", "best_epoch": "Best epoch", "epochs_run": "Epochs run",
                 "valid_log_loss": "Held-out log loss", "fit_seconds": "Fit time (s)"}),
        {"Training rows": "{:,}", "Fire starts": "{:,}", "Held-out fire starts": "{:,}", "Parameters": "{:,}",
         "Held-out log loss": "{:.5f}", "Fit time (s)": "{:.0f}"})

    images = (embedded_image(FIGURES / "report_watch.png", "Share of fire starts on the daily watch list"),
              embedded_image(FIGURES / "report_ap.png", "Average precision relative to random"),
              embedded_image(FIGURES / "report_curves.png", "Loss per epoch"))
    starts23 = int(cov23.fire_starts.sum())
    cov22 = coverage.loc[coverage.year.eq(manifest["selection_year"])]
    ref, cat = "catboost_spread_noday", "start_drivers"
    bb = boot.loc[boot.scope.eq("BC-wide")].set_index(["baseline", "definition", "model"])
    ci = lambda b, d, m: (f'{bb.loc[(b, d, m), "difference"]:+.3f} (95% interval {bb.loc[(b, d, m), "ci_low"]:+.3f} to '
                          f'{bb.loc[(b, d, m), "ci_high"]:+.3f})')
    ap = lambda m: s("res7_exact", m, "median_average_precision")
    share = lambda m, group, measure="top50_within_7d": float(g.loc[g.measure.eq(measure) & g.model.eq(m) & g.onset_group.eq(group), "share"].iloc[0])
    conclusions = ""

    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BC | Fire starts</title>
<style>
:root{{--ink:#192c31;--muted:#54676b;--accent:#bd5417;--teal:#237e85;--line:#dce4e3}}
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f4;color:var(--ink);font:16px/1.55 system-ui,sans-serif}}
main{{max-width:1320px;margin:auto;padding:42px 28px}}header{{padding:15px 0 30px;border-bottom:2px solid var(--ink)}}
.eyebrow{{font-size:12px;font-weight:750;letter-spacing:2px;color:var(--accent);text-transform:uppercase}}
h1{{font-size:clamp(30px,4vw,48px);line-height:1.1;margin:12px 0 18px;text-wrap:balance}}h2{{font-size:25px;margin:0 0 12px}}h3{{font-size:19px;margin:18px 0 8px}}
p{{margin:8px 0;max-width:920px}}.purpose{{font-size:21px}}section{{padding:30px 0;border-bottom:1px solid var(--line)}}
ul{{padding-left:21px;margin:12px 0;max-width:980px}}li{{margin:5px 0}}.muted,figcaption{{color:var(--muted);font-size:14px}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin:20px 0}}
.card{{background:white;border:1px solid var(--line);border-radius:12px;padding:20px}}.value{{font-size:40px;font-weight:750;color:var(--teal)}}
figure{{margin:16px 0 0;background:white;border:1px solid var(--line);border-radius:12px;padding:12px}}img{{display:block;width:100%;height:auto}}
figure.narrow img{{max-width:860px}}
details{{margin-top:18px}}summary{{cursor:pointer;font-weight:650;padding:12px;background:#e9efec;border-radius:8px}}
.conclusion{{background:#e9f0ed;border-radius:12px;padding:24px}}
.table{{overflow-x:auto;background:white;border:1px solid var(--line);border-radius:12px;margin:16px 0}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:8px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}}
th:first-child,td:first-child{{text-align:left}}th{{color:var(--muted);font-weight:650}}td{{font-variant-numeric:tabular-nums}}tbody tr:last-child td{{border-bottom:none}}
footer{{color:var(--muted);font-size:13px;padding:20px 0}}@media(max-width:800px){{.cards{{grid-template-columns:1fr}}main{{padding:24px 16px}}}}
</style></head><body><main>
<header><div class="eyebrow">Experiment 35 · Fire starts with networks</div>
<h1>Do an MLP and an MLP + GRU predict fire starts better than CatBoost?</h1>
<p class="purpose"><strong>Purpose:</strong> Experiment 34 trained CatBoost on fire starts only, the first detection in a cell after a quiet week. This experiment trains Experiment 23's regularised MLP and MLP + 14-day GRU on the same fire starts, to see whether a network, or a model that sees the past two weeks day by day, ranks the next fire start better.</p>
<p class="muted">{len(regions)} Fire Centres · train 2018–2021 · thresholds 2022 · evaluate 2023 · 2024 excluded</p></header>

<section><h2>Setup</h2>
<ul><li><strong>Fire start:</strong> a cell positive on the label on day t (S-NPP or NOAA-20 from 2021, S-NPP before, one-day gaps filled) with no detection from the same source in the cell or its six neighbours on days t-7 to t-1. Only these <em>quiet</em> cell-days can start a fire, so every forecast is trained and scored on quiet cell-days only.</li>
<li><strong>Networks:</strong> Experiment 23's MLP (256-256-128) and MLP + 64-unit GRU over the past 14 days, with its regularised settings (learning rate 1e-4, dropout 0.3, weight decay 0.1, early stopping on held-out weeks). Trained on Experiment 34's rows: quiet cell-days of 2018–2021, all fire starts plus Experiment 21's 1% draw of the rest, with Experiment 21's inputs. Input scaling is refit on these rows; the output starts at the training base rate.</li>
<li><strong>References, loaded unchanged:</strong> Experiment 34's fire-start CatBoost, Experiment 21's next-day CatBoost, Experiment 22's next-day MLP, two simple rankings and random lists. Experiment 34's numbers reproduce exactly.</li>
<li><strong>Two ways to judge:</strong> F1 with a threshold tuned on 2022, as before; and a daily watch list of the top N quiet cells per Fire Centre, which suits rare events better than F1.</li></ul>
{coverage_table}
<p class="muted">The base rate is the share of quiet cell-days that become a fire start. It is what a random guess would score as precision and as average precision.</p>
{fit_table}
<figure class="narrow">{images[2]}<figcaption>Log loss per epoch. The saved network is the epoch with the lowest held-out loss.</figcaption></figure></section>

<section><h2>Results (2023)</h2>
{summary_table}
<p class="muted">F1, precision and recall pool all six Fire Centres. AP (average precision) is the median over centres and needs no threshold. Top 50: the share of 2023's {starts23:,} fire starts among each day's 50 highest-ranked quiet cells per Fire Centre.</p>
<figure>{images[0]}<figcaption>Share of 2023 fire starts on the daily watch list, BC-wide, for list sizes from 10 to 100 quiet cells per Fire Centre per day. The dashed line is a random list.</figcaption></figure>
<figure class="narrow">{images[1]}<figcaption>Average precision divided by each centre's base rate (log scale). 1 is random.</figcaption></figure>
<h3>Watch lists, BC-wide</h3>{watch_table}
<h3>Which fire starts are caught</h3>
<p>Fire starts that are also onsets in Experiment 16 (S-NPP) and have its BC Wildfire Service (BCWS) grouping: {", ".join(f"{GROUP_NAMES[k].lower()} {int(v):,}" for k, v in counts.items())}.</p>
{group_table}
<h3>Week-block bootstrap against Experiment 34's fire-start CatBoost, BC-wide</h3>{boot_table("start_drivers")}
<h3>Week-block bootstrap against Experiment 21's next-day CatBoost, BC-wide</h3>{boot_table("catboost_spread_noday")}
<details><summary>Show F1 for every centre</summary>{per_centre}</details></section>

<section><div class="conclusion"><h2>Conclusion</h2>{conclusions}</div>
<ul class="muted"><li>Training uses S-NPP-only fire starts in 2018–2020; 2021–2023 also use NOAA-20, which finds more starts.</li>
<li>About half of quiet onsets have no BCWS record (Experiment 16): many are likely agricultural, slash or industrial burns, not wildfires.</li>
<li>2023 has been used to compare many experiments; only 2024 will give an untouched number.</li></ul></section>
<footer>Generated by build_report.py from saved Experiment 35 results. All figures are embedded: this report works offline as one HTML file.</footer>
</main></body></html>'''
    (HERE / "RESULTS.html").write_text(page, encoding="utf-8")
    print(f"Report: {HERE / 'RESULTS.html'}")


if __name__ == "__main__":
    main()
