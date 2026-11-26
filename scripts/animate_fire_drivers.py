"""Animate fire, wind, precipitation and vegetation across BC for one month.

Four panels, one frame per day:
  1. Fire: S-NPP VIIRS detections on the day (FIRMS, from datasets/raw/bc_firms_summer_2018_2024.csv).
  2. Wind: ERA5 afternoon wind (mean of the 20 UTC and next-day 00 UTC snapshots, 13:00-17:00 PDT), from
     data/era5_wind_h3r7/wind_<year>.parquet. Arrows are drawn on every n-th res-7 cell (--wind-stride).
  3. Precipitation: ERA5 total precipitation (tp, in mm). Caution: the interim ERA5 table is built from
     the five-snapshot average, so these values are a proxy, not true daily totals (see experiment 30).
  4. Vegetation: ERA5 leaf area index (lai_lv + lai_hv) for each day. It is a smooth seasonal field, so it changes
     only slightly from day to day, and it does not register burn scars.

Inputs: datasets/interim/era5_h3_res6_cell_date_18_24.parquet (res 6, all BC cells), and the FIRMS CSV.
Output: an animated GIF (default) or MP4 (needs ffmpeg).

usage: python animate_fire_drivers.py REPO_ROOT --year 2018 --month 7 --out out.gif [--fps 2] [--wind-stride 150]
Needs the h3 package in addition to pandas, pyarrow, matplotlib and Pillow.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import h3  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from matplotlib import animation  # noqa: E402

ERA5_COLS = ["h3_cell", "valid_time", "h3_lat", "h3_lon", "u10", "v10", "tp", "lai_lv", "lai_hv"]


def region_bbox(root: Path, slug: str, pad: float = 0.3):
    """Lon/lat box around one Fire Centre's grid cells (experiment 02 region grids), with a small margin."""
    g = pd.read_parquet(root / f"experiments/02_hotspot_persistence_bc/regions/{slug}/data/h3_grid.parquet",
                        columns=["h3_cell"])
    ll = np.array([h3.cell_to_latlng(c) for c in g.h3_cell.unique()])
    lat0, lat1 = ll[:, 0].min() - pad, ll[:, 0].max() + pad
    lon0, lon1 = ll[:, 1].min() - pad, ll[:, 1].max() + pad
    return lon0, lon1, lat0, lat1


def load_afternoon_wind(root: Path, year: int, month: int, stride: int, bbox=None):
    """Afternoon wind (wind_pm_u, wind_pm_v) for a sample of res-7 cells, one row per cell and day.
    If bbox (lon0, lon1, lat0, lat1) is given, only cells inside it are kept, before the stride sampling."""
    start = pd.Timestamp(year, month, 1)
    end = start + pd.offsets.MonthBegin(1)
    t = pq.read_table(root / f"data/era5_wind_h3r7/wind_{year}.parquet",
                      columns=["h3_cell", "date", "wind_pm_u", "wind_pm_v"],
                      filters=[("date", ">=", start), ("date", "<", end)]).to_pandas()
    cells = np.array(sorted(t.h3_cell.unique()))
    ll = np.array([h3.cell_to_latlng(c) for c in cells])
    if bbox is not None:
        lon0, lon1, lat0, lat1 = bbox
        inside = (ll[:, 1] >= lon0) & (ll[:, 1] <= lon1) & (ll[:, 0] >= lat0) & (ll[:, 0] <= lat1)
        cells, ll = cells[inside], ll[inside]
    keep = np.arange(len(cells))[::stride]
    cells, ll = cells[keep], ll[keep]
    t = t[t.h3_cell.isin(set(cells))]
    return cells, ll[:, 1], ll[:, 0], t


def load_era5(root: Path, year: int, month: int) -> pd.DataFrame:
    src = root / "datasets/interim/era5_h3_res6_cell_date_18_24.parquet"
    t = pq.read_table(src, columns=ERA5_COLS, filters=[("year", "=", year), ("month", "=", month)]).to_pandas()
    t["date"] = pd.to_datetime(t.valid_time).dt.normalize()
    t["tp_mm"] = t.tp.clip(lower=0) * 1000.0
    t["lai"] = t.lai_lv + t.lai_hv
    return t


def load_fire(root: Path, year: int, month: int) -> pd.DataFrame:
    f = pd.read_csv(root / "datasets/raw/bc_firms_summer_2018_2024.csv",
                    usecols=["latitude", "longitude", "acq_date", "type"], parse_dates=["acq_date"])
    f = f[(f.acq_date.dt.year == year) & (f.acq_date.dt.month == month) & (f.type == 0)]
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="repository root")
    ap.add_argument("--year", type=int, default=2018)
    ap.add_argument("--month", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--wind-stride", type=int, default=150,
                    help="draw an afternoon wind arrow on every n-th res-7 cell (about 1,500 arrows at 150)")
    ap.add_argument("--centre", default=None,
                    help="zoom to one Fire Centre, e.g. prince_george, kamloops, southeast, cariboo, coastal, northwest")
    args = ap.parse_args()
    root = args.root.resolve()

    bbox = region_bbox(root, args.centre) if args.centre else None
    era = load_era5(root, args.year, args.month)
    fire = load_fire(root, args.year, args.month)
    if bbox is not None:
        lon0, lon1, lat0, lat1 = bbox
        era = era[era.h3_lon.between(lon0, lon1) & era.h3_lat.between(lat0, lat1)]
        fire = fire[fire.longitude.between(lon0, lon1) & fire.latitude.between(lat0, lat1)]
    days = sorted(era.date.unique())
    wstride = args.wind_stride if bbox is None else max(1, args.wind_stride // 8)
    wcells, wlon, wlat, wind = load_afternoon_wind(root, args.year, args.month, wstride, bbox)

    # Static cell grid (same cells each day).
    cells = era[era.date == days[0]][["h3_cell", "h3_lat", "h3_lon", "lai"]].reset_index(drop=True)
    lon, lat = cells.h3_lon.to_numpy(), cells.h3_lat.to_numpy()
    lai = cells.lai.to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    ax_fire, ax_wind, ax_rain, ax_veg = axes.ravel()
    for ax, title in zip((ax_fire, ax_wind, ax_rain, ax_veg),
                         ("Fire: S-NPP detections", "Wind: ERA5 afternoon (13-17 PDT)",
                          "Precipitation (ERA5, mm, proxy)",
                          "Vegetation: leaf area index")):
        ax.set_title(title)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_aspect(1 / np.cos(np.radians(55)))

    # Panel 4: vegetation background, updated each day from the ERA5 leaf area index for that day.
    # Colour scale fixed to this month's own range, so small seasonal changes are visible.
    lai_month = era.lai.dropna()
    lai_lo, lai_hi = float(lai_month.quantile(0.02)), float(lai_month.quantile(0.98))
    sc_veg = ax_veg.scatter(lon, lat, c=lai, s=2, cmap="YlGn", vmin=lai_lo, vmax=lai_hi)
    fig.colorbar(sc_veg, ax=ax_veg, shrink=0.7, label=f"LAI (m²/m²), {lai_lo:.1f}–{lai_hi:.1f} this month")

    # Panels 1-3 are redrawn each frame.
    rain_max = max(float(era.tp_mm.quantile(0.99)), 1.0)
    ax_fire.scatter(lon, lat, s=2, c="lightgrey")  # BC land background
    fire_dots = ax_fire.scatter([], [], s=6, c="crimson")
    ax_fire.set_xlim(lon.min(), lon.max())
    ax_fire.set_ylim(lat.min(), lat.max())
    for ax in (ax_wind, ax_rain):
        ax.set_xlim(lon.min(), lon.max())
        ax.set_ylim(lat.min(), lat.max())
    rain_sc = ax_rain.scatter(lon, lat, c=np.zeros_like(lon), s=2, cmap="Blues", vmin=0, vmax=rain_max)
    fig.colorbar(rain_sc, ax=ax_rain, shrink=0.7, label="mm")
    ax_wind.scatter(lon, lat, s=2, c="lightgrey")  # BC land background
    quiv = ax_wind.quiver(wlon, wlat, np.zeros(len(wlon)), np.zeros(len(wlon)), color="navy",
                          scale=25, scale_units="inches", width=0.003, headwidth=3)
    suptitle = fig.suptitle("")

    def frame(i):
        day = days[i]
        e = era[era.date == day].set_index("h3_cell").reindex(cells.h3_cell)
        f = fire[fire.acq_date == day]
        fire_dots.set_offsets(np.c_[f.longitude.to_numpy(), f.latitude.to_numpy()] if len(f) else np.empty((0, 2)))
        ax_fire.set_title(f"Fire: S-NPP detections ({len(f)} on this day)")
        rain_sc.set_array(np.nan_to_num(e.tp_mm.to_numpy(), nan=0.0))
        sc_veg.set_array(np.nan_to_num(e.lai.to_numpy(), nan=0.0))
        w = wind[wind.date == day].set_index("h3_cell").reindex(wcells)
        u = np.nan_to_num(w.wind_pm_u.to_numpy(), nan=0.0)
        v = np.nan_to_num(w.wind_pm_v.to_numpy(), nan=0.0)
        quiv.set_UVC(u, v)
        suptitle.set_text(f"BC, {day:%Y-%m-%d}")
        return fire_dots, rain_sc, sc_veg, quiv, suptitle

    anim = animation.FuncAnimation(fig, frame, frames=len(days), interval=1000 / args.fps, blit=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix.lower() == ".gif":
        anim.save(args.out, writer=animation.PillowWriter(fps=args.fps))
    else:
        anim.save(args.out, writer=animation.FFMpegWriter(fps=args.fps))
    print(f"{len(days)} frames -> {args.out}")


if __name__ == "__main__":
    main()
