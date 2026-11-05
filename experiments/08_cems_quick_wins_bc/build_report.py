"""Build standalone HTML report and figures for Experiment 08 (All 5 CEMS & FIRMS Quick Wins)."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

MODEL_LABELS = {
    "M0_firms_only": "M0: FIRMS Only (Baseline)",
    "M1_raw_cems": "M1: +Raw CEMS (Exp 03a)",
    "M2_frp_brightness": "M2: +FRP/Brightness/2-Hop (QW2)",
    "M3_engineered_cems": "M3: +Engineered CEMS (QW3)",
    "M4_frp_plus_eng_cems": "M4: +FRP + Engineered CEMS (QW2+3)",
    "M5_ensemble_5seeds": "M5: 5-Seed Ensemble (QW5)",
    "04a_gnn_cems": "Exp 04a: GNN + Raw CEMS",
    "M6_gbdt_gnn_blend": "M6: 5-Seed GBDT + GNN Blend",
}


def embed_png(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def make_ablation_chart(summary_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 5.6), constrained_layout=True)
    regions_df = summary_df.loc[summary_df["region"] != "Mean (6 Centres)"].set_index("region")
    x = np.arange(len(regions_df))
    width = 0.13
    models_to_plot = [
        ("M0_firms_only", "#7f8c8d", "M0: FIRMS Only"),
        ("M1_raw_cems", "#3498db", "M1: +Raw CEMS"),
        ("M2_frp_brightness", "#f39c12", "M2: +FRP/Brightness/2-Hop"),
        ("M3_engineered_cems", "#9b59b6", "M3: +Engineered CEMS"),
        ("M5_ensemble_5seeds", "#2ecc71", "M5: +FRP+Eng CEMS (5-Seed)"),
        ("M6_gbdt_gnn_blend", "#e74c3c", "M6: GBDT+GNN Blend"),
    ]
    offsets = np.linspace(-2.5 * width, 2.5 * width, len(models_to_plot))
    for offset, (col, color, label) in zip(offsets, models_to_plot):
        ax.bar(x + offset, regions_df[col], width=width, color=color, label=label, edgecolor="white", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(regions_df.index, fontsize=10)
    ax.set_ylim(0.50, 0.90)
    ax.set_ylabel("2023 Average Precision (AP)")
    ax.set_title("Quick Wins Ablation Ladder Across All 6 BC Fire Centres (2023 Evaluation)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(ncol=3, frameon=False, fontsize=9, loc="upper left")
    out = FIGURES / "ablation_ap_by_region.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def make_regime_breakdown_chart(subset_df: pd.DataFrame) -> Path:
    mean_regimes = (
        subset_df.groupby(["regime", "model"], as_index=False)["average_precision"]
        .mean()
        .pivot(index="regime", columns="model", values="average_precision")
    )
    order = ["all_rows", "active_local_7d", "no_local_7d", "neighbour_spread", "pure_ignition"]
    labels = [
        "All Rows\n(34.24M)",
        "Active Persistence\n(local_7d > 0)",
        "No Local Fire\n(local_7d == 0)",
        "Neighbour Spread\n(local=0, nb>0)",
        "Pure New Ignition\n(local=0, nb=0)",
    ]
    mean_regimes = mean_regimes.loc[order]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)

    # Panel A: AP on All Rows, Active Persistence, No Local Fire, Neighbour Spread
    x1 = np.arange(4)
    w = 0.24
    axes[0].bar(x1 - w, mean_regimes["M0_firms_only"].iloc[:4], width=w, color="#7f8c8d", label="M0: FIRMS Only")
    axes[0].bar(x1, mean_regimes["M1_raw_cems"].iloc[:4], width=w, color="#3498db", label="M1: +Raw CEMS")
    axes[0].bar(x1 + w, mean_regimes["M5_ensemble_5seeds"].iloc[:4], width=w, color="#2ecc71", label="M5: Full Quick Wins")
    axes[0].set_xticks(x1)
    axes[0].set_xticklabels(labels[:4], fontsize=9)
    axes[0].set_ylabel("Mean 6-Centre AP (2023)")
    axes[0].set_title("Quick Win 1A: AP by Fire Activity Regime")
    axes[0].grid(axis="y", linestyle=":", alpha=0.5)
    axes[0].legend(frameon=False, fontsize=9)

    # Panel B: Relative % AP Lift over M0_firms_only across all 5 regimes (highlighting Pure New Ignitions & Spread)
    x2 = np.arange(5)
    rel_m1 = 100.0 * (mean_regimes["M1_raw_cems"] - mean_regimes["M0_firms_only"]) / mean_regimes["M0_firms_only"]
    rel_m5 = 100.0 * (mean_regimes["M5_ensemble_5seeds"] - mean_regimes["M0_firms_only"]) / mean_regimes["M0_firms_only"]
    rel_m6 = 100.0 * (mean_regimes["M6_gbdt_gnn_blend"] - mean_regimes["M0_firms_only"]) / mean_regimes["M0_firms_only"]

    axes[1].bar(x2 - w, rel_m1, width=w, color="#3498db", label="M1 (+Raw CEMS) vs M0")
    axes[1].bar(x2, rel_m5, width=w, color="#2ecc71", label="M5 (Full Quick Wins) vs M0")
    axes[1].bar(x2 + w, rel_m6, width=w, color="#e74c3c", label="M6 (GBDT+GNN Blend) vs M0")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("Relative AP Gain over FIRMS-Only (%)")
    axes[1].set_title("Quick Win 1B: Relative % AP Lift by Regime")
    axes[1].grid(axis="y", linestyle=":", alpha=0.5)
    axes[1].legend(frameon=False, fontsize=9)

    out = FIGURES / "regime_ap_breakdown.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def make_bootstrap_ci_chart(boot_df: pd.DataFrame) -> Path:
    sub = boot_df.loc[boot_df["region"] == "Mean (6 Centres)"].copy()
    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    y = np.arange(len(sub))
    deltas = sub["ap_delta"].to_numpy()
    err_low = deltas - sub["ci95_low"].to_numpy()
    err_high = sub["ci95_high"].to_numpy() - deltas

    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    ax.errorbar(
        deltas,
        y,
        xerr=[err_low, err_high],
        fmt="o",
        color="#2a78d6",
        ecolor="#2a78d6",
        elinewidth=2.0,
        capsize=4,
        markersize=7,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(sub["comparison"], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("2023 Mean 6-Centre AP Delta (with 95% Paired Day-Block Bootstrap CI)")
    ax.set_title("Quick Win 5: Paired Day-Block Bootstrap 95% Confidence Intervals (1,000 Replicates)")
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    out = FIGURES / "bootstrap_confidence_intervals.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary_df = pd.read_csv(RESULTS / "summary_ap_table_2023.csv")
    subset_df = pd.read_csv(RESULTS / "active_vs_quiet_ap_2023.csv")
    multi_val_df = pd.read_csv(RESULTS / "multi_year_validation_2023.csv")
    boot_df = pd.read_csv(RESULTS / "bootstrap_significance_2023.csv")
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))

    fig1 = make_ablation_chart(summary_df)
    fig2 = make_regime_breakdown_chart(subset_df)
    fig3 = make_bootstrap_ci_chart(boot_df)

    # Format HTML report
    mean_sub = (
        subset_df.groupby(["regime", "model"], as_index=False)["average_precision"]
        .mean()
        .pivot(index="regime", columns="model", values="average_precision")
        .loc[["all_rows", "active_local_7d", "no_local_7d", "neighbour_spread", "pure_ignition"]]
        .reset_index()
    )
    boot_mean = boot_df.loc[boot_df["region"] == "Mean (6 Centres)"][
        ["comparison", "ap_ref", "ap_new", "ap_delta", "ap_rel_pct", "ci95_low", "ci95_high", "p_value_two_sided"]
    ]
    val_piv = (
        multi_val_df.loc[multi_val_df["model"] == "M5_ensemble_5seeds"]
        .pivot(index="region", columns="threshold_scheme", values=["threshold", "f1"])
    )
    val_piv.columns = [f"{a}_{b}" for a, b in val_piv.columns]
    val_piv.loc["Mean (6 Centres)"] = val_piv.mean(axis=0)
    val_piv = val_piv.reset_index()

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Experiment 08 — All 5 CEMS & FIRMS Quick Wins (BC Fire Centres)</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #1c2430; max-width: 1180px; margin: 32px auto; padding: 0 24px; line-height: 1.5; }}
  h1, h2, h3 {{ color: #111827; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; font-size: 13.5px; }}
  th, td {{ border: 1px solid #d8dee9; padding: 7px 10px; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; font-weight: 600; }}
  th {{ background: #f3f6fa; }}
  tr:nth-child(even) {{ background: #fafbfc; }}
  .figure {{ margin: 20px 0 32px; text-align: center; }}
  .figure img {{ max-width: 100%; border: 1px solid #e5e9f0; border-radius: 6px; }}
  .callout {{ background: #f0f7ff; border-left: 4px solid #2a78d6; padding: 12px 16px; margin: 16px 0; border-radius: 4px; }}
</style>
</head>
<body>
<h1>Experiment 08: All 5 CEMS &amp; FIRMS Quick Wins Across All 6 BC Fire Centres</h1>
<div class="callout">
  <strong>Protocol:</strong> Trained strictly on <code>2018–2021</code>, validated on <code>2022</code> (and out-of-fold <code>2021+2022</code> for Quick Win 4), evaluated on all <code>34,244,884</code> H3 cell-days of <code>2023</code> across all 6 BC Fire Centres. Holdout year <code>2024</code> is untouched (<code>holdout_loaded = {manifest['holdout_loaded']}</code>).
</div>

<h2>1. Overall 2023 Average Precision (AP) Ablation Ladder</h2>
<div class="figure"><img src="{embed_png(fig1)}" alt="Ablation AP by Region"></div>
{summary_df.round(4).to_html(index=False)}

<h2>2. Quick Win 1 — Active Persistence vs. Quiet / Spread / New-Ignition AP Breakdown (Mean Across 6 Centres)</h2>
<div class="figure"><img src="{embed_png(fig2)}" alt="Regime AP Breakdown"></div>
{mean_sub.round(5).to_html(index=False)}

<h2>3. Quick Win 5 — Paired Day-Block Bootstrap 95% Confidence Intervals &amp; p-Values (1,000 Replicates)</h2>
<div class="figure"><img src="{embed_png(fig3)}" alt="Bootstrap Confidence Intervals"></div>
{boot_mean.round(5).to_html(index=False)}

<h2>4. Quick Win 4 — Multi-Year Validation (`2021+2022` OOF &amp; Empirical-Bayes Shrinkage vs. `2022`-Only F1)</h2>
{val_piv.round(4).to_html(index=False)}

</body>
</html>
"""
    (HERE / "RESULTS.html").write_text(html, encoding="utf-8")
    print(f"Wrote {HERE / 'RESULTS.html'} and figures in {FIGURES}", flush=True)


if __name__ == "__main__":
    main()

