"""Build a standalone visual HTML report from saved Experiment 38 outputs.

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
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
FIGURES = HERE / "figures"
spec = importlib.util.spec_from_file_location("experiment_03_report", HERE.parent / "03_exp_learned_baselines_bc/build_report.py")
report_03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report_03)
embedded_image, html_table, style = report_03.embedded_image, report_03.html_table, report_03.style
MODEL_NAMES = {"exp22_ign_full": "Exp 36: ignitions, all inputs", "exp23_ign_full": "Exp 37: big lightning, all inputs",
               "exp23_ign_lightning": "Exp 37: big lightning, + 14-day lightning", "exp16_spread": "Exp 21: next-day satellite",
               "exp20_start": "Exp 34: satellite fire start", "fwi": "Ranking: FWI",
               "recent_fire_nearby": "Ranking: satellite fire next to the cell, 7 days", "lightning_14d": "Ranking: lightning, 14 days",
               "everything_is_fire": "Everything is fire"}
LABEL_NAMES = {"burn_8d": "Burns (satellite), 8 days", "new_burn_8d": "New satellite fire, 8 days", "ignition_8d": "BCWS wildfire starts, 8 days",
               "big_lightning_8d": "Lightning fire ≥ 10 ha starts, 8 days"}
SETS = {"test": ("CanadaFireSat Test", "#2a78d6"), "test_hard": ("CanadaFireSat Test Hard", "#eb6834"),
        "full_grid": ("Every cell (our usual scoring)", "#1baf7a")}
PAPER = {"test": 0.603, "test_hard": 0.374}  # their best model, ResNet-50 multi-modal, 2023


def f1_chart(summary, grid):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), sharey=False, constrained_layout=True)
    for ax, (label, title) in zip(axes.ravel(), LABEL_NAMES.items()):
        s = summary.loc[summary.label.eq(label)].pivot(index="model", columns="set", values="f1")
        s["full_grid"] = grid.loc[grid.label.eq(label)].set_index("model").f1
        s = s.drop(index="everything_is_fire").sort_values("test")
        positions = np.arange(len(s))
        for offset, (column, (name, colour)) in zip((.26, 0, -.26), SETS.items()):
            ax.barh(positions + offset, s[column], height=.25, color=colour, label=name)
        ax.axvline(.5, color="#54676b", linestyle="--", linewidth=1.2)
        for key, value in PAPER.items():
            ax.axvline(value, color=SETS[key][1], linestyle=":", linewidth=1.4)
        ax.set(yticks=positions, yticklabels=[MODEL_NAMES[m] for m in s.index], xlim=(0, 1), xlabel="F1, 2023", title=title)
        ax.tick_params(axis="y", labelsize=8)
        style(ax)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(FIGURES / "report_f1.png", dpi=150)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    assert not manifest["holdout_loaded"]
    summary = pd.read_csv(RESULTS / "protocol_summary.csv")
    grid = pd.read_csv(RESULTS / "full_grid_metrics.csv")
    sizes = pd.read_csv(RESULTS / "sample_sizes.csv")
    f1_chart(summary, grid)

    size_rows = sizes.groupby(["label", "set"], sort=False)[["positives", "negatives", "positive_share"]].mean().reset_index()
    full = grid.loc[grid.model.eq("everything_is_fire")].set_index("label")
    for label in LABEL_NAMES:
        p = full.loc[label, "positive_share"]
        size_rows.loc[len(size_rows)] = {"label": label, "set": "full_grid_2023", "positives": np.nan, "negatives": np.nan, "positive_share": p}
    size_rows["Everything-is-fire F1"] = 2 * size_rows.positive_share / (1 + size_rows.positive_share)
    size_rows["label"] = size_rows.label.map(LABEL_NAMES)
    size_table = html_table(size_rows.rename(columns={"label": "Label", "set": "Set", "positives": "Positives", "negatives": "Negatives",
                                                      "positive_share": "Share positive"}),
                            {"Positives": "{:,.0f}", "Negatives": "{:,.0f}", "Share positive": "{:.4f}", "Everything-is-fire F1": "{:.3f}"})

    tables = {}
    for label in LABEL_NAMES:
        s = summary.loc[summary.label.eq(label)]
        wide = s.pivot(index="model", columns="set", values=["f1", "f1_sd", "prauc", "f1_best_threshold"])
        g = grid.loc[grid.label.eq(label)].set_index("model")
        rows = pd.DataFrame({"Forecast": [MODEL_NAMES[m] for m in wide.index],
                             "F1 Test": wide[("f1", "test")].to_numpy(), "± sd": wide[("f1_sd", "test")].to_numpy(),
                             "F1 Test Hard": wide[("f1", "test_hard")].to_numpy(),
                             "F1 Test, best threshold": wide[("f1_best_threshold", "test")].to_numpy(),
                             "PRAUC Test": wide[("prauc", "test")].to_numpy(), "PRAUC Test Hard": wide[("prauc", "test_hard")].to_numpy(),
                             "F1 every cell": g.loc[wide.index, "f1"].to_numpy(), "PRAUC every cell": g.loc[wide.index, "prauc"].to_numpy()})
        tables[label] = html_table(rows.sort_values("F1 Test", ascending=False),
                                   {c: "{:.3f}" for c in rows.columns if c != "Forecast"})

    s = summary.set_index(["label", "set", "model"]).sort_index()
    g = grid.set_index(["label", "model"]).sort_index()
    f = lambda label, model, key="test": s.loc[(label, key, model), "f1"]
    conclusions = f"""<ul>
<li><strong>The paper's 60% F1 comes mostly from how it is measured.</strong> With two negatives per positive, a third of the sample is fire, and flagging everything already scores F1 0.50. CanadaFireSat's best model is 10 points above that on Test (0.603) and below it on Test Hard (0.374), if its Test Hard set also has two negatives per positive.</li>
<li><strong>Scored their way, our models land in the same range or higher.</strong> For "the cell burns in the next 8 days", the closest to their label, Experiment 21's next-day satellite model reaches F1 {f("burn_8d", "exp16_spread"):.3f} on Test and {f("burn_8d", "exp16_spread", "test_hard"):.3f} on Test Hard, and Experiment 34's model {f("burn_8d", "exp20_start"):.3f} / {f("burn_8d", "exp20_start", "test_hard"):.3f}. That is above the paper's 0.603 / 0.374, though on a satellite label rather than NBAC polygons, so it is not a like-for-like comparison. Most of it comes from fires already burning: the simple "satellite fire next to the cell in the last week" ranking alone gets {f("burn_8d", "recent_fire_nearby"):.3f}.</li>
<li><strong>New satellite fires</strong> (a detection with no fire in the cell or its neighbours in the past week): the best learned model gets {s.loc[("new_burn_8d", "test")].drop(index=["everything_is_fire", "fwi", "recent_fire_nearby", "lightning_14d"]).f1.max():.3f} on Test and {s.loc[("new_burn_8d", "test_hard")].drop(index=["everything_is_fire", "fwi", "recent_fire_nearby", "lightning_14d"]).f1.max():.3f} on Test Hard; the best simple ranking {s.loc[("new_burn_8d", "test"), "f1"].loc[["fwi", "recent_fire_nearby", "lightning_14d"]].max():.3f} / {s.loc[("new_burn_8d", "test_hard"), "f1"].loc[["fwi", "recent_fire_nearby", "lightning_14d"]].max():.3f}. Every cell: {g.loc["new_burn_8d", "f1"].drop(index="everything_is_fire").max():.3f} at best.</li>
<li><strong>For new ignitions, the project's real target, there is little above the trivial 0.50.</strong> Experiment 36's model gets {f("ignition_8d", "exp22_ign_full"):.3f} on Test and {f("ignition_8d", "exp22_ign_full", "test_hard"):.3f} on Test Hard (PRAUC {s.loc[("ignition_8d", "test", "exp22_ign_full"), "prauc"]:.3f} against 0.333 at random). For big lightning fires, the best is {f("big_lightning_8d", "exp22_ign_full"):.3f}; Experiment 37's own models fall below 0.50 at the 2022 threshold, which rests on only {int(sizes.loc[sizes.label.eq("big_lightning_8d") & sizes.set.eq("val_2022"), "positives"].mean())} positives.</li>
<li><strong>The same forecasts on every cell</strong> give F1 {g.loc[("burn_8d", "exp16_spread"), "f1"]:.3f} for burning, {g.loc[("ignition_8d", "exp22_ign_full"), "f1"]:.3f} for ignitions and {g.loc[("big_lightning_8d", "exp22_ign_full"), "f1"]:.3f} for big lightning fires: the share of positive cells drops from 33% to {full.loc["ignition_8d", "positive_share"]:.2%} for ignitions, and precision falls with it.</li>
<li><strong>Which number to quote:</strong> CanadaFireSat-style F1 shows the model separates fire from comparable non-fire places. Every-cell F1 shows how many false alarms a real daily map would carry. Both are valid, but only every-cell F1 tells an operator what to expect, and the two cannot be compared with each other.</li></ul>"""

    image = embedded_image(FIGURES / "report_f1.png", "F1 on CanadaFireSat-style samples and on every cell")
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BC | CanadaFireSat protocol</title>
<style>
:root{{--ink:#192c31;--muted:#54676b;--accent:#bd5417;--teal:#237e85;--line:#dce4e3}}
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f4;color:var(--ink);font:16px/1.55 system-ui,sans-serif}}
main{{max-width:1320px;margin:auto;padding:42px 28px}}header{{padding:15px 0 30px;border-bottom:2px solid var(--ink)}}
.eyebrow{{font-size:12px;font-weight:750;letter-spacing:2px;color:var(--accent);text-transform:uppercase}}
h1{{font-size:clamp(30px,4vw,48px);line-height:1.1;margin:12px 0 18px;text-wrap:balance}}h2{{font-size:25px;margin:0 0 12px}}h3{{font-size:19px;margin:18px 0 8px}}
p{{margin:8px 0;max-width:920px}}.purpose{{font-size:21px}}section{{padding:30px 0;border-bottom:1px solid var(--line)}}
ul{{padding-left:21px;margin:12px 0;max-width:980px}}li{{margin:5px 0}}.muted,figcaption{{color:var(--muted);font-size:14px}}
figure{{margin:16px 0 0;background:white;border:1px solid var(--line);border-radius:12px;padding:12px}}img{{display:block;width:100%;height:auto}}
details{{margin-top:18px}}summary{{cursor:pointer;font-weight:650;padding:12px;background:#e9efec;border-radius:8px}}
.conclusion{{background:#e9f0ed;border-radius:12px;padding:24px}}
.table{{overflow-x:auto;background:white;border:1px solid var(--line);border-radius:12px;margin:16px 0}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:8px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}}
th:first-child,td:first-child{{text-align:left}}th{{color:var(--muted);font-weight:650}}td{{font-variant-numeric:tabular-nums}}tbody tr:last-child td{{border-bottom:none}}
footer{{color:var(--muted);font-size:13px;padding:20px 0}}@media(max-width:800px){{main{{padding:24px 16px}}}}
</style></head><body><main>
<header><div class="eyebrow">Experiment 38 · CanadaFireSat protocol</div>
<h1>How do our models score when measured the way CanadaFireSat measures?</h1>
<p class="purpose"><strong>Purpose:</strong> CanadaFireSat (Porta et al., arXiv 2506.08690) reports 60.3% F1 for 2023, while our F1 is 0.03–0.12. Their F1 is computed on a sample with two non-fire places per fire place. This experiment applies that protocol to our saved models, with no retraining.</p>
<p class="muted">{len(manifest["regions"])} Fire Centres · res-7 cells (about 5 km²) · forecast dates every 8 days, {manifest["forecast_dates"][0]} to {manifest["forecast_dates"][-1]} · threshold from 2022 · evaluate 2023 · {len(manifest["seeds"])} negative draws · 2024 excluded</p></header>

<section><h2>Setup</h2>
<ul><li><strong>Labels:</strong> the cell burns (an S-NPP or NOAA-20 detection) in days t to t+7, closest to the paper's NBAC burned-area label; a new satellite fire starts there (Experiment 34's fire start: a detection after 7 days with none in the cell or its neighbours); a BCWS wildfire starts there (Experiment 36's target); a lightning fire that grows to 10 ha or more starts there (Experiment 37's target).</li>
<li><strong>Test:</strong> every positive cell-date plus 2 negatives per positive, per Fire Centre and year. Negatives come from dates between the year's first and last positive, from cells that had not burned in earlier years (and, in 2023, were not 2022 negatives), with equal numbers from each FWI decile.</li>
<li><strong>Test Hard:</strong> negatives matched to the positives' FWI deciles within each Fire Centre × dominant fuel class × month, so weather and land cover cannot separate the classes.</li>
<li><strong>Threshold:</strong> one per model and label, maximizing F1 on a 2022 sample drawn like Test. "Best threshold" columns show the most a threshold could give on 2023.</li>
<li><strong>Every cell:</strong> the same forecast dates scored on every cell with no sampling, the way Experiments 36–37 score.</li></ul>
{size_table}
<p class="muted">On a sample that is one-third fire, flagging everything scores F1 0.50: this is the floor any model must clear.</p></section>

<section><h2>Results (2023)</h2>
<figure>{image}<figcaption>F1 per forecast on CanadaFireSat-style Test and Test Hard samples, and on every cell. Dashed gray: flagging everything (0.50). Dotted: the paper's best model on Test (0.603, blue) and Test Hard (0.374, orange).</figcaption></figure>
{"".join(f"<h3>{LABEL_NAMES[label]}</h3>{tables[label]}" for label in LABEL_NAMES)}
<p class="muted">Test columns are means over {len(manifest["seeds"])} negative draws; "± sd" is their standard deviation. PRAUC is 0.333 for a random ranking on the samples and the positive share on every cell.</p></section>

<section><div class="conclusion"><h2>Conclusion</h2>{conclusions}</div>
<ul class="muted"><li>The paper's label is NBAC burned-area polygons; ours is satellite detections or BCWS ignition points. The paper does not give the 2023 positive share of its test set, so its own "flag everything" floor is not known exactly.</li>
<li>Experiments 36–37 were trained for a 7-day window; here they are scored on 8 days to match the paper.</li>
<li>2023 has been used to compare many experiments; only 2024 will give an untouched number.</li></ul></section>
<footer>Generated by build_report.py from saved Experiment 38 results. All figures are embedded: this report works offline as one HTML file.</footer>
</main></body></html>'''
    (HERE / "RESULTS.html").write_text(page, encoding="utf-8")
    print(f"Report: {HERE / 'RESULTS.html'}")


if __name__ == "__main__":
    main()
