"""
Exploratory Data Analysis (EDA) on CEMS Fire Weather Index (FWI) Data for British Columbia (2018-2023).
Saves figures and summary tables to data/exploration/
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------
# Dynamic Paths Setup
# ROOT resolves to: /Users/arshdeepkaur/wildfire_detection/rbc-team-gaw
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "datasets"
OUT = ROOT / "data" / "exploration"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 1. Load Cleaned BC CEMS Dataset (2018-2024)
# ---------------------------------------------------------
csv_path = DATASETS_DIR / "dataset_CEMS_18_24.csv"
if not csv_path.exists():
    csv_path = DATASETS_DIR / "dataset_CEMS_18_23.csv"
if not csv_path.exists():
    csv_path = ROOT.parent / "dataset_CEMS_18_23.csv"
if not csv_path.exists():
    csv_path = ROOT / "dataset_CEMS_18_23.csv"

if not csv_path.exists():
    raise FileNotFoundError(
        f"Could not find dataset_CEMS_18_24.csv or dataset_CEMS_18_23.csv in {DATASETS_DIR} or {ROOT}. "
        "Please run check_CEMS_coverage.py first to generate it."
    )

print(f"Loading {csv_path}...")
df = pd.read_csv(csv_path)

fwi_vars = ["ffmcode", "dufmcode", "drtcode", "infsinx", "fbupinx", "fwinx", "fdsrte"]
df["valid_time"] = pd.to_datetime(df["valid_time"])
df["year"] = df["valid_time"].dt.year
df["month"] = df["valid_time"].dt.month

print(f"Total Rows: {len(df):,}")

# ---------------------------------------------------------
# 2. Statistical Summary (Percentiles & Skewness)
# ---------------------------------------------------------
print("\n=== 1. NUMERIC PERCENTILES & SKEWNESS ===")
summary = df[fwi_vars].describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T
summary["skew"] = df[fwi_vars].skew()
summary.to_csv(OUT / "cems_numeric_summary.csv")
print(summary[["mean", "50%", "95%", "max", "skew"]].round(2))

# ---------------------------------------------------------
# 3. Fire Danger Class Distribution (FWI standard bins)
# ---------------------------------------------------------
# Standard Canadian / EFFIS FWI Danger Classes
bins = [-np.inf, 5.2, 11.2, 21.3, 38.0, np.inf]
labels = ["Very Low", "Low", "Moderate", "High", "Extreme"]
df["danger_class"] = pd.cut(df["fwinx"], bins=bins, labels=labels)

danger_summary = (
    pd.crosstab(df["year"], df["danger_class"], normalize="index") * 100
).round(2)
danger_summary.to_csv(OUT / "cems_danger_classes_by_year.csv")
print("\n=== 2. FIRE DANGER CLASS BREAKDOWN (%) BY YEAR ===")
print(danger_summary)

# ---------------------------------------------------------
# 4. Correlation Matrix
# ---------------------------------------------------------
corr = df[fwi_vars].corr(method="spearman")
corr.to_csv(OUT / "cems_correlation_matrix.csv")

# ---------------------------------------------------------
# 5. Plot 1: Distributions & Correlation Heatmap
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

# Histograms of the 7 variables
for i, var in enumerate(fwi_vars):
    ax = axes[i]
    # Sample 100,000 for fast plotting
    sample_vals = df[var].sample(n=min(100000, len(df)), random_state=42)
    ax.hist(sample_vals, bins=50, color="#d95f02", edgecolor="none", alpha=0.85)
    ax.set_title(f"{var.upper()}", fontsize=11, fontweight="bold")
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle=":", alpha=0.6)

# Heatmap on the 8th subplot
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=axes[7])
axes[7].set_title("Spearman Correlation", fontsize=11, fontweight="bold")

fig.suptitle("CEMS Fire Weather Indices: Distributions & Collinearity (BC 2018–2023)", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "cems_distributions_and_corr.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT / 'cems_distributions_and_corr.png'}")

# ---------------------------------------------------------
# 6. Plot 2: Seasonality & Year-Over-Year Trends
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# A. Monthly Seasonal Trend
monthly_fwi = df.groupby(["year", "month"])["fwinx"].median().reset_index()
for yr, grp in monthly_fwi.groupby("year"):
    ax1.plot(grp["month"], grp["fwinx"], marker="o", label=str(yr), linewidth=2)

ax1.set_xticks(range(5, 10))
ax1.set_xticklabels(["May", "Jun", "Jul", "Aug", "Sep"])
ax1.set_title("Median FWI by Month & Year", fontsize=12, fontweight="bold")
ax1.set_xlabel("Month")
ax1.set_ylabel("Median FWI")
ax1.legend()
ax1.grid(True, linestyle=":", alpha=0.6)

# B. Annual High Fire Danger Days (FWI > 21.3)
high_danger = (
    df[df["fwinx"] >= 21.3]
    .groupby("year")
    .size()
    .divide(df.groupby("year").size())
    * 100
)
bars = ax2.bar(high_danger.index, high_danger.values, color="#b30000", alpha=0.85)
ax2.set_title("% of Grid Observations with High/Extreme Danger (FWI ≥ 21.3)", fontsize=12, fontweight="bold")
ax2.set_xlabel("Year")
ax2.set_ylabel("Percentage (%)")
ax2.grid(True, linestyle=":", alpha=0.6, axis="y")

for bar in bars:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2, yval + 0.3, f"{yval:.1f}%", ha="center", va="bottom", fontsize=10)

fig.tight_layout()
fig.savefig(OUT / "cems_trends_and_extremes.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT / 'cems_trends_and_extremes.png'}")

print(f"\nEDA Completed successfully! All outputs saved in {OUT}")