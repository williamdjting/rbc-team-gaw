"""Download BC Wildfire Service station weather (BCWS Data Mart) and reduce it to daily rows.

Source: https://www.for.gov.bc.ca/ftp/HPR/external/!publish/BCWS_DATA_MART/<year>/
  <year>_BCWS_WX_OBS.csv       hourly observations; daily FWI-system codes on the noon row
  <year>_BCWS_WX_STATIONS.csv  station locations

Output (data/bcws_weather/):
  bcws_daily_<start>_<end>.parquet  one row per station and local-standard date:
      ffmc, dmc, dc, isi, bui, fwi, danger_rating (daily values, computed from the 12:00 LST obs),
      precip_mm (sum of hourly), temp_max, rh_min, wind_max, temp_noon, rh_noon, wind_noon
  stations_<start>_<end>.csv        union of station tables (last listing wins per code)
Raw yearly files are deleted after reduction unless --keep-raw is given.
Daily values for date d are available at about 13:00 local on day d; use d-1 for day-ahead features.
"""
from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://www.for.gov.bc.ca/ftp/HPR/external/!publish/BCWS_DATA_MART"
ROOT = Path(__file__).resolve().parents[1]
DAILY = {"FINE_FUEL_MOISTURE_CODE": "ffmc", "DUFF_MOISTURE_CODE": "dmc", "DROUGHT_CODE": "dc",
         "INITIAL_SPREAD_INDEX": "isi", "BUILDUP_INDEX": "bui", "FIRE_WEATHER_INDEX": "fwi",
         "DANGER_RATING": "danger_rating"}
HOURLY = ["HOURLY_PRECIPITATION", "HOURLY_TEMPERATURE", "HOURLY_RELATIVE_HUMIDITY", "HOURLY_WIND_SPEED"]


def fetch(url, path):
    if path.exists():
        return
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=600) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.rename(path)


def reduce_year(obs_path, months):
    cols = ["STATION_CODE", "DATE_TIME"] + HOURLY + list(DAILY)
    parts = []
    for chunk in pd.read_csv(obs_path, usecols=cols, chunksize=1_000_000, low_memory=False):
        dt = chunk["DATE_TIME"].astype("int64").astype(str)
        chunk["date"] = pd.to_datetime(dt.str[:8], format="%Y%m%d")
        chunk["hour"] = dt.str[8:10].astype(int)
        parts.append(chunk[chunk["date"].dt.month.isin(months)])
    h = pd.concat(parts, ignore_index=True)
    for c in HOURLY + [c for c in DAILY if c != "DANGER_RATING"]:
        h[c] = pd.to_numeric(h[c], errors="coerce")
    g = h.groupby(["STATION_CODE", "date"])
    day = g.agg(precip_mm=("HOURLY_PRECIPITATION", "sum"), temp_max=("HOURLY_TEMPERATURE", "max"),
                rh_min=("HOURLY_RELATIVE_HUMIDITY", "min"), wind_max=("HOURLY_WIND_SPEED", "max"),
                n_hours=("hour", "size"))
    noon = h[h["hour"] == 12].drop_duplicates(["STATION_CODE", "date"]).set_index(["STATION_CODE", "date"])
    day = day.join(noon[["HOURLY_TEMPERATURE", "HOURLY_RELATIVE_HUMIDITY", "HOURLY_WIND_SPEED"]].rename(
        columns={"HOURLY_TEMPERATURE": "temp_noon", "HOURLY_RELATIVE_HUMIDITY": "rh_noon",
                 "HOURLY_WIND_SPEED": "wind_noon"}))
    daily = h.dropna(subset=["FINE_FUEL_MOISTURE_CODE"]).drop_duplicates(["STATION_CODE", "date"], keep="last")
    day = day.join(daily.set_index(["STATION_CODE", "date"])[list(DAILY)].rename(columns=DAILY))
    return day.reset_index().rename(columns={"STATION_CODE": "station_code"})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start-year", type=int, default=2018)
    p.add_argument("--end-year", type=int, default=2023)
    p.add_argument("--months", default="4,5,6,7,8,9,10")
    p.add_argument("--output-dir", type=Path, default=ROOT / "data/bcws_weather")
    p.add_argument("--keep-raw", action="store_true")
    a = p.parse_args()
    months = [int(m) for m in a.months.split(",")]
    raw = a.output_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    days, stations = [], []
    for y in range(a.start_year, a.end_year + 1):
        obs = raw / f"{y}_BCWS_WX_OBS.csv"
        st = raw / f"{y}_BCWS_WX_STATIONS.csv"
        fetch(f"{BASE}/{y}/{obs.name}", obs)
        fetch(f"{BASE}/{y}/{st.name}", st)
        d = reduce_year(obs, months)
        days.append(d)
        s = pd.read_csv(st, encoding="utf-8-sig")
        s["listing_year"] = y
        stations.append(s)
        print(f"{y}: {len(d):,} station-days, {d['station_code'].nunique()} stations, "
              f"{d['ffmc'].notna().mean():.1%} with daily FWI codes", flush=True)
        if not a.keep_raw:
            obs.unlink()
    out = pd.concat(days, ignore_index=True)
    out["station_code"] = out["station_code"].astype(int)
    stem = f"{a.start_year}_{a.end_year}"
    out.to_parquet(a.output_dir / f"bcws_daily_{stem}.parquet", index=False)
    s = pd.concat(stations, ignore_index=True).sort_values("listing_year").drop_duplicates("STATION_CODE", keep="last")
    s.to_csv(a.output_dir / f"stations_{stem}.csv", index=False)
    print(f"Saved {len(out):,} station-days and {len(s)} stations to {a.output_dir}")


if __name__ == "__main__":
    main()
