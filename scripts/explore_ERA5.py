"""
Exploratory Data Analysis (EDA) on ERA5-Land Meteorology, Hydrology, Radiation & Land-Surface
Data for British Columbia (2018-2024).

Mirrors `scripts/explore_CEMS.py` while adding ERA5-specific physical unit conversions and
wildfire meteorology diagnostics:
  - Kelvin -> Celsius (`t2m_c`, `d2m_c`, `skt_c`)
  - Saturation & Actual Vapor Pressure -> Relative Humidity (`rh_pct`) and Vapor Pressure Deficit (`vpd_kpa`)
  - Zonal (`u10`) & Meridional (`v10`) Wind Vectors -> 10m Wind Speed (`wind_speed_10m`) & Direction (`wind_dir_deg`)
  - Total Precipitation meters -> millimeters (`tp_mm`)
  - Geopotential (`z`) -> Terrain Elevation in meters (`elevation_m = z / 9.80665`)
  - Soil Moisture layers (`swvl1`, `swvl2`, `swvl3`), Radiation (`ssrd`, `ssr`, `str`, `slhf`),
    Evaporation (`evabs`, `evatc`, `evavt`, `pev`), Snow (`sde`, `snowc`), and Vegetation (`lai_hv`, `lai_lv`, `cvh`, `cvl`)

Saves summary CSV tables and publication-ready figures to `data/exploration/`.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
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
# 1. Load Cleaned BC ERA5 Dataset (Prefer Parquet for Speed, Fallback to CSV)
# ---------------------------------------------------------
CANDIDATE_PATHS = [
    DATASETS_DIR / "dataset_ERA5_18_23.parquet",
    DATASETS_DIR / "dataset_ERA5_18_24.parquet",
    DATASETS_DIR / "dataset_ERA5_18_23.csv",
    DATASETS_DIR / "dataset_ERA5_18_24.csv",
]
data_path = next((p for p in CANDIDATE_PATHS if p.exists()), None)
if data_path is None:
    raise FileNotFoundError(
        f"Could not find cleaned ERA5 dataset in {DATASETS_DIR}. "
        "Please run `python scripts/check_ERA5_coverage.py` first."
    )

print(f"Loading {data_path}...")
if data_path.suffix == ".parquet":
    df = pd.read_parquet(data_path)
else:
    df = pd.read_csv(data_path)

df["valid_time"] = pd.to_datetime(df["valid_time"])
df["year"] = df["valid_time"].dt.year.astype(np.int16)
df["month"] = df["valid_time"].dt.month.astype(np.int8)

# Strictly restrict exploratory feature selection to 2018-2023 development years if 2024 is present
if (df["year"] == 2024).any():
    print("Filtering to 2018-2023 development years (keeping 2024 holdout untouched for feature selection)...")
    df = df[df["year"] <= 2023].copy()

print(f"Total ERA5 BC Land Rows (2018-2023): {len(df):,}")

# ---------------------------------------------------------
# 2. ERA5-Specific Physical Unit Conversions & Derived Wildfire Variables
# ---------------------------------------------------------
print("Computing physical unit conversions and derived wildfire meteorology features...")

# Temperatures: Kelvin -> Celsius
df["t2m_c"] = (df["t2m"] - 273.15).astype(np.float32)
df["d2m_c"] = (df["d2m"] - 273.15).astype(np.float32)
df["skt_c"] = (df["skt"] - 273.15).astype(np.float32)

# Saturation vapor pressure (es) & actual vapor pressure (ea) in kPa via Magnus-Tetens formula
es_kpa = 0.61078 * np.exp((17.27 * df["t2m_c"]) / (df["t2m_c"] + 237.3))
ea_kpa = 0.61078 * np.exp((17.27 * df["d2m_c"]) / (df["d2m_c"] + 237.3))

# Vapor Pressure Deficit (VPD, kPa) & Relative Humidity (RH, %)
df["vpd_kpa"] = np.maximum(0.0, es_kpa - ea_kpa).astype(np.float32)
df["rh_pct"] = np.clip(100.0 * (ea_kpa / np.maximum(1e-6, es_kpa)), 0.0, 100.0).astype(np.float32)

# 10m Wind Speed (m/s) and Meteorological Wind Direction (degrees, 0=N, 90=E, 180=S, 270=W)
df["wind_speed_10m"] = np.hypot(df["u10"], df["v10"]).astype(np.float32)
df["wind_dir_deg"] = ((270.0 - np.rad2deg(np.arctan2(df["v10"], df["u10"]))) % 360.0).astype(np.float32)

# Total Precipitation: meters -> millimeters
df["tp_mm"] = np.maximum(0.0, df["tp"] * 1000.0).astype(np.float32)

# Surface Pressure: Pa -> hPa
df["sp_hpa"] = (df["sp"] / 100.0).astype(np.float32)

# Terrain Elevation from Geopotential: z (m^2/s^2) / g0 (9.80665 m/s^2)
if "z" in df.columns:
    df["elevation_m"] = (df["z"] / 9.80665).astype(np.float32)

RAW_ERA5_VARS = [
    c for c in [
        "t2m", "d2m", "skt", "u10", "v10", "sp", "tp",
        "swvl1", "swvl2", "swvl3",
        "ssrd", "ssr", "str", "slhf",
        "evabs", "evatc", "evavt", "pev",
        "sde", "snowc",
        "lai_hv", "lai_lv", "cvh", "cvl", "z", "lsm",
    ]
    if c in df.columns
]

DERIVED_ERA5_VARS = [
    c for c in [
        "t2m_c", "d2m_c", "skt_c", "vpd_kpa", "rh_pct",
        "wind_speed_10m", "wind_dir_deg", "tp_mm", "sp_hpa", "elevation_m",
    ]
    if c in df.columns
]

# Sample 250,000 rows for fast exact quantile/skew/correlation calculations on 12M+ rows
sample_df = df.sample(n=min(250_000, len(df)), random_state=42)

# ---------------------------------------------------------
# 3. Statistical Summary (Percentiles & Skewness)
# ---------------------------------------------------------
print("\n=== 1. ERA5 NUMERIC PERCENTILES & SKEWNESS ===")
all_eval_vars = DERIVED_ERA5_VARS + RAW_ERA5_VARS
summary = sample_df[all_eval_vars].describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T
summary["skew"] = sample_df[all_eval_vars].skew()
summary["missing_pct"] = (df[all_eval_vars].isnull().mean() * 100.0).round(4)
summary.to_csv(OUT / "era5_numeric_summary.csv")
print(summary[["mean", "50%", "95%", "max", "skew", "missing_pct"]].round(2))

# ---------------------------------------------------------
# 4. Atmospheric Fire Stress / VPD Danger Classes by Year
# ---------------------------------------------------------
# Vapor Pressure Deficit (VPD) classes (standard wildfire atmospheric aridity thresholds in kPa)
vpd_bins = [-np.inf, 0.4, 0.8, 1.2, 1.8, np.inf]
vpd_labels = ["Very Moist (<0.4)", "Low (0.4-0.8)", "Moderate (0.8-1.2)", "High (1.2-1.8)", "Extreme (>=1.8 kPa)"]
df["vpd_class"] = pd.cut(df["vpd_kpa"], bins=vpd_bins, labels=vpd_labels)

vpd_summary = (
    pd.crosstab(df["year"], df["vpd_class"], normalize="index") * 100.0
).round(2)
vpd_summary.to_csv(OUT / "era5_vpd_classes_by_year.csv")
print("\n=== 2. ERA5 VPD ATMOSPHERIC ARIDITY BREAKDOWN (%) BY YEAR ===")
print(vpd_summary)

# ---------------------------------------------------------
# 5. Correlation Matrix & Collinearity Audit
# ---------------------------------------------------------
CORR_VARS = [
    c for c in [
        "t2m_c", "d2m_c", "skt_c", "vpd_kpa", "rh_pct",
        "wind_speed_10m", "u10", "v10", "tp_mm", "sp_hpa",
        "swvl1", "swvl2", "swvl3",
        "ssrd", "ssr", "str", "slhf",
        "evabs", "evatc", "evavt", "pev",
        "sde", "snowc",
        "lai_hv", "lai_lv", "cvh", "cvl", "elevation_m",
    ]
    if c in sample_df.columns
]
corr = sample_df[CORR_VARS].corr(method="spearman")
corr.to_csv(OUT / "era5_correlation_matrix.csv")

# Extract high-collinearity variable pairs (|r| >= 0.80) to guide feature selection
high_corr_pairs = []
for i in range(len(CORR_VARS)):
    for j in range(i + 1, len(CORR_VARS)):
        r_val = float(corr.iloc[i, j])
        if abs(r_val) >= 0.80:
            high_corr_pairs.append({
                "var_1": CORR_VARS[i],
                "var_2": CORR_VARS[j],
                "spearman_r": round(r_val, 4),
                "abs_r": round(abs(r_val), 4),
            })
high_corr_df = pd.DataFrame(high_corr_pairs).sort_values("abs_r", ascending=False)
high_corr_df.to_csv(OUT / "era5_high_collinearity_pairs.csv", index=False)
print("\n=== 3. HIGHLY COLLINEAR ERA5 PAIRS (|Spearman r| >= 0.80) ===")
print(high_corr_df.to_string(index=False))

# ---------------------------------------------------------
# 6. Plot 1: Distributions of 12 Key ERA5 Wildfire Variables
# ---------------------------------------------------------
PLOT_VARS = [
    ("t2m_c", "2m Temp (°C)", "#d95f02"),
    ("d2m_c", "2m Dewpoint (°C)", "#e6ab02"),
    ("vpd_kpa", "Vapor Pressure Deficit (kPa)", "#b30000"),
    ("rh_pct", "Relative Humidity (%)", "#1f78b4"),
    ("wind_speed_10m", "10m Wind Speed (m/s)", "#7570b3"),
    ("tp_mm", "Daily Precip (mm)", "#386cb0"),
    ("swvl1", "Topsoil Moisture L1 (m³/m³)", "#1b9e77"),
    ("swvl2", "Soil Moisture L2 (m³/m³)", "#66a61e"),
    ("ssrd", "Solar Radiation Down (J/m²)", "#e7298a"),
    ("pev", "Potential Evap (m)", "#a6761d"),
    ("lai_hv", "Leaf Area Index High Veg", "#2ca25f"),
    ("elevation_m", "Terrain Elevation (m)", "#666666"),
]
PLOT_VARS = [(v, lbl, col) for v, lbl, col in PLOT_VARS if v in sample_df.columns]

fig, axes = plt.subplots(3, 4, figsize=(18, 11))
axes = axes.flatten()
for i, (var, label, color) in enumerate(PLOT_VARS):
    ax = axes[i]
    vals = sample_df[var].dropna()
    ax.hist(vals, bins=50, color=color, edgecolor="none", alpha=0.85)
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle=":", alpha=0.6)

fig.suptitle("ERA5-Land Key Wildfire Variables: Distributions Across BC Land Grid (2018–2023)", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "era5_distributions.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {OUT / 'era5_distributions.png'}")

# ---------------------------------------------------------
# 7. Plot 2: Spearman Correlation Heatmap
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 13))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1.0,
    vmax=1.0,
    annot_kws={"size": 7.5},
    cbar=True,
    ax=ax,
)
ax.set_title("ERA5-Land Spearman Correlation Matrix (BC Land Grid, 2018–2023)", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "era5_correlation_heatmap.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT / 'era5_correlation_heatmap.png'}")

# ---------------------------------------------------------
# 8. Plot 3: Seasonality, Soil Drying & Extreme VPD Fire-Weather Days
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

# A. Monthly Median VPD (kPa) by Year
monthly_vpd = df.groupby(["year", "month"])["vpd_kpa"].median().reset_index()
for yr, grp in monthly_vpd.groupby("year"):
    axes[0].plot(grp["month"], grp["vpd_kpa"], marker="o", label=str(yr), linewidth=2)
axes[0].set_xticks(range(5, 10))
axes[0].set_xticklabels(["May", "Jun", "Jul", "Aug", "Sep"])
axes[0].set_title("Median Vapor Pressure Deficit (VPD) by Month", fontsize=11.5, fontweight="bold")
axes[0].set_xlabel("Month")
axes[0].set_ylabel("Median VPD (kPa)")
axes[0].legend(fontsize=8.5)
axes[0].grid(True, linestyle=":", alpha=0.6)

# B. Monthly Median Topsoil Moisture (swvl1) by Year
monthly_soil = df.groupby(["year", "month"])["swvl1"].median().reset_index()
for yr, grp in monthly_soil.groupby("year"):
    axes[1].plot(grp["month"], grp["swvl1"], marker="s", label=str(yr), linewidth=2)
axes[1].set_xticks(range(5, 10))
axes[1].set_xticklabels(["May", "Jun", "Jul", "Aug", "Sep"])
axes[1].set_title("Median Topsoil Moisture (swvl1) Drawdown", fontsize=11.5, fontweight="bold")
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Median Volumetric Soil Water (m³/m³)")
axes[1].legend(fontsize=8.5)
axes[1].grid(True, linestyle=":", alpha=0.6)

# C. Annual % of BC Grid Observations with High Atmospheric Fire Stress (VPD >= 1.2 kPa & Topsoil swvl1 < 0.25)
hot_dry_mask = (df["vpd_kpa"] >= 1.2) & (df["swvl1"] < 0.25)
annual_stress = (
    df[hot_dry_mask]
    .groupby("year")
    .size()
    .reindex(sorted(df["year"].unique()), fill_value=0)
    .divide(df.groupby("year").size())
    * 100.0
)
bars = axes[2].bar(annual_stress.index.astype(str), annual_stress.values, color="#b30000", alpha=0.85)
axes[2].set_title("% of BC Grid-Days with Compound Dry-Arid Stress\n(VPD ≥ 1.2 kPa & Soil Moisture < 0.25)", fontsize=11, fontweight="bold")
axes[2].set_xlabel("Year")
axes[2].set_ylabel("Percentage of Cell-Days (%)")
axes[2].grid(True, linestyle=":", alpha=0.6, axis="y")

for bar in bars:
    yval = bar.get_height()
    axes[2].text(bar.get_x() + bar.get_width() / 2, yval + 0.2, f"{yval:.1f}%", ha="center", va="bottom", fontsize=9.5)

fig.tight_layout()
fig.savefig(OUT / "era5_trends_and_extremes.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT / 'era5_trends_and_extremes.png'}")

print(f"\nERA5 EDA Completed successfully! All outputs saved in {OUT}")

