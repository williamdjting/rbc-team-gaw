"""Interpolate BCWS station fire weather to dataset cells (inverse-distance weighting).

Input: data/bcws_weather/bcws_daily_2018_2023.parquet + stations (scripts/download_bcws_weather.py).
Output: data/ignition_dataset/res<r>/weather_<year>.parquet with h3_cell, date and w_* columns.
Row order matches ignition_<year>.parquet (cell-major, then date), so files can be
concatenated column-wise; h3_cell/date are included for checked merges.

Every w_* value for target date t uses the station daily value of t-1 (BCWS daily codes are
computed from the 12:00 LST observation, i.e. available during day t-1). Per variable and day,
the IDW uses the K nearest stations that reported that variable, within MAX_KM.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
WX = ROOT / "data/bcws_weather"
VARS = ["ffmc", "dmc", "dc", "isi", "bui", "fwi", "temp_max", "rh_min", "precip_mm", "wind_max"]
K, POWER, MAX_KM, EARTH_KM = 4, 2.0, 150.0, 6371.0088
YEARS = range(2018, 2024)


def xyz(lat, lon):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def idw(cell_xyz, st_xyz, values):
    ok = ~np.isnan(values)
    out = np.full(len(cell_xyz), np.nan, dtype=np.float32)
    near = np.full(len(cell_xyz), np.nan, dtype=np.float32)
    if ok.sum() == 0:
        return out, near
    tree = cKDTree(st_xyz[ok])
    k = min(K, ok.sum())
    d, i = tree.query(cell_xyz, k=k)
    d, i = d.reshape(len(cell_xyz), k), i.reshape(len(cell_xyz), k)
    km = 2 * EARTH_KM * np.arcsin(np.clip(d / 2, 0, 1))
    v = values[ok][i]
    w = np.where(km <= MAX_KM, 1.0 / np.maximum(km, 0.5) ** POWER, 0.0)
    s = w.sum(axis=1)
    has = s > 0
    out[has] = (w[has] * v[has]).sum(axis=1) / s[has]
    near[:] = km[:, 0]
    return out, near


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resolution", type=int, default=6)
    a = p.parse_args()
    d = ROOT / f"data/ignition_dataset/res{a.resolution}"
    grid = pd.read_parquet(d / "grid.parquet")
    cxyz = xyz(grid["lat"].to_numpy(float), grid["lon"].to_numpy(float))
    daily = pd.read_parquet(WX / "bcws_daily_2018_2023.parquet")
    st = pd.read_csv(WX / "stations_2018_2023.csv")
    st = st.dropna(subset=["LATITUDE", "LONGITUDE"]).rename(columns={"STATION_CODE": "station_code"})
    daily = daily.merge(st[["station_code", "LATITUDE", "LONGITUDE"]], on="station_code")
    for year in YEARS:
        dates = pd.date_range(f"{year}-05-08", f"{year}-09-30")
        cols = {v: np.full((len(grid), len(dates)), np.nan, dtype=np.float32) for v in VARS}
        cols["fwi_station_km"] = np.full((len(grid), len(dates)), np.nan, dtype=np.float32)
        for j, t in enumerate(dates):
            day = daily[daily["date"] == t - pd.Timedelta(days=1)]
            if day.empty:
                continue
            sxyz = xyz(day["LATITUDE"].to_numpy(float), day["LONGITUDE"].to_numpy(float))
            for v in VARS:
                vals, near = idw(cxyz, sxyz, day[v].to_numpy(float))
                cols[v][:, j] = vals
                if v == "fwi":
                    cols["fwi_station_km"][:, j] = near
        n = len(dates)
        out = pd.DataFrame({"h3_cell": np.repeat(grid["cell"].to_numpy(), n), "date": np.tile(dates.values, len(grid))})
        for v, arr in cols.items():
            out[f"w_{v}"] = arr.ravel()
        out["h3_cell"] = out["h3_cell"].astype("category")
        out.to_parquet(d / f"weather_{year}.parquet", index=False)
        print(f"{year}: missing share " + ", ".join(f"{v}={np.isnan(cols[v]).mean():.1%}" for v in ("fwi", "temp_max", "precip_mm")),
              f"median nearest FWI station {np.nanmedian(cols['fwi_station_km']):.0f} km", flush=True)


if __name__ == "__main__":
    main()
