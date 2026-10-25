"""Build a standalone visual HTML report from saved experiment outputs."""
import base64
import calendar
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from report_class_balance import add_class_balance

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
from shapely import from_wkt

from prepare_daily_grid import BOUNDARY, polygon_parts
from shapely.geometry import shape
from shapely import make_valid

HERE = Path(__file__).resolve().parent


def embedded_image(path, alt):
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="{alt}" loading="lazy">'


def main():
    figures = HERE / "figures"
    figures.mkdir(exist_ok=True)
    grid = pd.read_parquet(HERE / "data/kamloops_h3_grid.parquet")
    polygons = {row.h3_cell: list(from_wkt(row.geometry_wkt).exterior.coords)
                for row in grid.itertuples()}
    boundary = json.loads(BOUNDARY.read_text(encoding="utf-8"))
    region = make_valid(shape(next(f["geometry"] for f in boundary["features"]
                                  if f["properties"]["MOF_FIRE_CENTRE_NAME"] == "Kamloops Fire Centre")))
    active = pd.read_parquet(HERE / "data/active_cell_days.parquet")
    selections = {}
    for year in (2018, 2023):
        annual = active.loc[active.acq_date.dt.year.eq(year)]
        selections[(year, "monthly")] = [
            (calendar.month_name[month], annual.loc[annual.acq_date.dt.month.eq(month)]
             .groupby("h3_cell").detection_count.sum()) for month in (6, 7, 8, 9)]
        august = annual.loc[annual.acq_date.dt.month.eq(8)]
        selections[(year, "weekly")] = [
            (f"August {start}-{end}", august.loc[august.acq_date.dt.day.between(start, end)]
             .groupby("h3_cell").detection_count.sum())
            for start, end in ((1, 7), (8, 14), (15, 21), (22, 31))]
    # Share colour scales between years within each time aggregation.
    scales = {kind: LogNorm(1, max(2, max(int(counts.max()) for (year, k), groups in selections.items()
                                         if k == kind for _, counts in groups if len(counts))))
              for kind in ("monthly", "weekly")}
    for (year, kind), groups in selections.items():
        fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True,
                                 sharex=True, sharey=True)
        for ax, (label, counts) in zip(axes.flat, groups):
            for polygon in polygon_parts(region):
                ax.plot(*polygon.exterior.xy, color="gray", linewidth=.6, zorder=2)
            if len(counts):
                ax.add_collection(PolyCollection([polygons[cell] for cell in counts.index],
                                                 array=counts.to_numpy(dtype=float),
                                                 cmap="Oranges", norm=scales[kind], edgecolors="none"))
            left, bottom, right, top = region.bounds
            ax.set(xlim=(left-.08, right+.08), ylim=(bottom-.08, top+.08),
                   title=f"{label} ({int(counts.sum()):,})", xlabel="Longitude", ylabel="Latitude")
            ax.set_aspect(1/np.cos(np.deg2rad(50.6745)))
            ax.set_axisbelow(True)
            ax.grid(linestyle=":", linewidth=.5, alpha=.4)
        fig.colorbar(ScalarMappable(norm=scales[kind], cmap="Oranges"), ax=axes.ravel().tolist(),
                     shrink=.8, label="Recorded detections per H3 cell (log scale)")
        fig.suptitle(f"Kamloops Fire Centre | {year} | {kind.capitalize()} activity", fontsize=14)
        fig.savefig(figures / f"report_{kind}_{year}.png", dpi=160)
        plt.close(fig)

    metrics = pd.read_csv(HERE / "results/metrics_by_year.csv")
    manifest = json.loads((HERE / "results/manifest.json").read_text())
    selected = manifest["selected_rule"]
    chosen = metrics.loc[metrics.year.eq(2023) & metrics.rule.eq(selected)].iloc[0]
    neighbours = metrics.loc[metrics.year.eq(2023) & metrics.rule.eq("local_or_neighbour_1d")].iloc[0]
    names = {"local_1d": "Same cell: previous day", "local_3d": "Same cell: previous 3 days",
             "local_7d": "Same cell: previous 7 days",
             "local_or_neighbour_1d": "Cell + neighbours: previous day",
             "local_or_neighbour_3d": "Cell + neighbours: previous 3 days",
             "local_or_neighbour_7d": "Cell + neighbours: previous 7 days"}
    comparison = metrics.loc[metrics.year.eq(2023) & metrics.rule.ne("always_no_detection")].set_index("rule").loc[list(names)]
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    positions = np.arange(len(comparison))
    ax.barh(positions-.18, comparison.precision*100, height=.34, label="Precision", color="#ca601c")
    ax.barh(positions+.18, comparison.recall*100, height=.34, label="Recall", color="#237e85")
    for index, row in enumerate(comparison.itertuples()):
        ax.text(row.precision*100+1, index-.18, f"{row.precision:.1%}", va="center", fontsize=9)
        ax.text(row.recall*100+1, index+.18, f"{row.recall:.1%}", va="center", fontsize=9)
    ax.set(yticks=positions, yticklabels=list(names.values()), xlim=(0, 105), xlabel="Percent of cell-days", title="2023: broader rules catch more activity but flag more false positives")
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle=":", alpha=.3)
    ax.set_axisbelow(True)
    fig.savefig(figures / "report_precision_recall.png", dpi=160)
    plt.close(fig)
    images = {(year, kind): embedded_image(figures / f"report_{kind}_{year}.png", f"{year} Kamloops {kind} recorded hotspot activity")
              for year in (2018, 2023) for kind in ("monthly", "weekly")}
    chart = embedded_image(figures / "report_precision_recall.png", "2023 precision and recall comparison of six persistence rules")
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kamloops | Hotspot persistence experiment</title>
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
.timeline{{display:grid;grid-template-columns:1fr 45px 1fr;align-items:center;gap:8px;margin:18px 0}}
.step{{padding:20px;border:1px solid var(--line);border-radius:12px;background:white}}.arrow{{font-size:30px;text-align:center}}
details{{margin-top:18px}}summary{{cursor:pointer;font-weight:650;padding:12px;background:#e9efec;border-radius:8px}}
details .columns{{margin-top:16px}}.conclusion{{background:#e9f0ed;border-radius:12px;padding:24px}}
footer{{color:var(--muted);font-size:13px;padding:20px 0}}@media(max-width:800px){{.columns,.cards{{grid-template-columns:1fr}}main{{padding:24px 16px}}}}
@media print{{body{{background:white}}main{{padding:0}}details .columns{{display:grid}}figure,.card{{break-inside:avoid}}}}
</style></head><body><main>
<header><div class="eyebrow">Experiment 01 · Hotspot persistence</div>
<h1>Does recorded hotspot activity persist?</h1>
<p class="purpose"><strong>Purpose:</strong> test whether recent hotspot observations help predict a detection in the same area the next day, and establish a benchmark for future graph models.</p>
<p class="muted">Kamloops Fire Centre · H3 resolution 7 · Spatial examples: 2018 &amp; 2023 · 2024 excluded</p></header>

<section><h2>01 / Spatial evidence: recurring areas of activity</h2>
<ul><li>We inspected Kamloops in <strong>2018 and 2023</strong>, first monthly, then through four August periods.</li>
<li>Clusters recur in similar areas across panels. This suggests spatial persistence; maps alone do not identify the same fires or prove spread.</li></ul>
<div class="columns"><figure>{images[(2018, 'weekly')]}<figcaption>2018: recurring southern clusters, with activity elsewhere varying across periods.</figcaption></figure>
<figure>{images[(2023, 'weekly')]}<figcaption>2023: compare the locations of clusters across successive August periods.</figcaption></figure></div>
<p class="muted">August 22–31 contains 10 days; the other periods contain 7. Raw totals are not equal-duration comparisons. Weekly maps share one colour scale across both years.</p>
<details><summary>Show monthly context: June–September, 2018 and 2023</summary><div class="columns">
<figure>{images[(2018, 'monthly')]}<figcaption>2018 monthly recorded activity.</figcaption></figure>
<figure>{images[(2023, 'monthly')]}<figcaption>2023 monthly recorded activity.</figcaption></figure></div>
<p class="muted">Monthly maps share a scale across years, separate from the weekly scale.</p></details></section>

<section><h2>02 / Temporal experiment: predict the next day</h2>
<div class="timeline"><div class="step"><div class="eyebrow">Inputs · Before target date</div><h3>Past 1, 3 or 7 days</h3>Detection counts in the cell, or the cell plus immediate H3 neighbours.</div><div class="arrow">→</div>
<div class="step"><div class="eyebrow">Target · Next day</div><h3>Any recorded detection?</h3><strong>1</strong> = one or more records<br><strong>0</strong> = no record in the archive</div></div>
<div class="rule"><strong>Selected rule:</strong> a detection in the same cell yesterday → predict a detection today.</div>
<ul><li><strong>Example:</strong> to predict August 15, the 1-day rule uses August 14; the 3-day rule uses August 12–14; the 7-day rule uses August 8–14.</li>
<li><strong>Rule, not a trained model:</strong> predict 1 when the past count exceeds zero. No GNN or learned weights yet.</li>
<li><strong>Split:</strong> 2018–2021 diagnostics → compare fixed rules on 2022 → select highest F1 → evaluate on 2023.</li>
<li><strong>Scope:</strong> 16,233 fixed cells; May 8–September 30 targets; seven-day warmup omitted; all cells retained.</li></ul></section>

<section><h2>03 / Results: same-cell, previous-day rule</h2>
<p>2023 development evaluation · {int(chosen.rows):,} cell-days · {int(chosen.positive_cell_days):,} recorded-positive targets</p>
<div class="cards"><div class="card"><h3>Precision</h3><div class="value">{chosen.precision:.1%}</div><p>Of flagged cell-days, about 58 in 100 had a recorded detection.</p></div>
<div class="card"><h3>Recall</h3><div class="value">{chosen.recall:.1%}</div><p>Of cell-days with detections, about 58 in 100 were flagged.</p></div>
<div class="card"><h3>F1</h3><div class="value">{chosen.f1:.3f}</div><p>Combines precision and recall. The rule was selected on 2022.</p></div></div>
<p><strong>{int(chosen.tp):,}</strong> correct positive flags · <strong>{int(chosen.fp):,}</strong> false positive flags · <strong>{int(chosen.fn):,}</strong> missed positive cell-days</p>
<figure>{chart}<figcaption>Each rule predicts positive when its past count is above zero. Broader history and neighbour rules increase recall but reduce precision.</figcaption></figure>
<ul><li><strong>Adding yesterday’s neighbour activity:</strong> recall rises to {neighbours.recall:.1%}, while precision falls to {neighbours.precision:.1%}.</li>
<li><strong>Context:</strong> only {chosen.prevalence:.3%} of all evaluated cell-days had recorded detections.</li></ul></section>

<section><div class="conclusion"><h2>04 / Conclusion</h2><ul>
<li><strong>Spatial plots suggested recurring locations; the baseline quantified day-to-day persistence.</strong></li>
<li>Recent same-cell activity is a useful signal. Broader neighbour rules catch more activity at the cost of more false positives.</li>
<li>Future learned models and GNNs should improve on this benchmark under the same evaluation setup.</li></ul></div>
<ul class="muted"><li>Counts and targets describe <strong>recorded observations</strong>, not ignition, unique fires, severity or confirmed fire absence. Coverage remains unknown.</li>
<li>2023 is a development evaluation because its plots were already inspected; 2024 remains excluded.</li>
<li>Past-only feature checks and count reconciliation passed. Prior-day records are assumed available; operational publication delays are not modeled.</li></ul></section>
<footer>Generated from saved experiment tables and metrics. All figures are embedded: this report works offline as one HTML file.</footer>
</main></body></html>'''
    output = HERE / "RESULTS.html"
    output.write_text(add_class_balance(html, HERE), encoding="utf-8")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
