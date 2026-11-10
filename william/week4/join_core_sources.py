#!/usr/bin/env python3
"""
This file does the following:
Joins the three core sources into one table for ignition prediction.

Put this file in the same folder as the data (your week4 folder) and run:
    python join_core_sources.py

Inputs it looks for (in this folder or any subfolder):
    *.shp                                BCWS fire incident points (labels)
    cg_flashes_3hr_0.1-deg_YYYY.nc       CanCPLD lightning grid (trigger)
    YYYY_BCWS_WX_STATIONS.csv            weather station locations
    YYYY_BCWS_WX_OBS.csv                 weather station observations (dryness)

Output:
    joined_ignition_dataset.csv          one row per (0.1 degree cell, day)

Row design: every cell-day that had at least one fire (ignition = 1), plus a
random sample of cell-days with no fire (ignition = 0).
"""
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

# ----------------------------------------------------------------------------
# Settings you may want to change
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
YEARS = [2021, 2022, 2023, 2024]
BBOX = dict(lat_min=48.0, lat_max=60.1, lon_min=-139.1, lon_max=-114.0)  # BC
SEASON_MONTHS = list(range(4, 11))   # April to October
NEG_PER_POS = 20                     # no-fire rows sampled per fire row
MAX_STATION_KM = 100                 # drop cells farther than this from any station
K_STATIONS = 3                       # fall back to 2nd/3rd nearest if 1st has no data
PST_OFFSET_HOURS = 8                 # lightning is UTC; fires and weather are PST
SEED = 42
OUT = HERE / "joined_ignition_dataset.csv"

WX_HOURLY = {
    "HOURLY_TEMPERATURE": ("temp_max_c", "max"),
    "HOURLY_RELATIVE_HUMIDITY": ("rh_min_pct", "min"),
    "HOURLY_WIND_SPEED": ("wind_max_kmh", "max"),
    "HOURLY_PRECIPITATION": ("precip_sum_mm", "sum"),
}
WX_DAILY = {   # reported once a day (noon), so "first" picks the one value
    "FINE_FUEL_MOISTURE_CODE": "ffmc",
    "DUFF_MOISTURE_CODE": "dmc",
    "DROUGHT_CODE": "dc",
    "INITIAL_SPREAD_INDEX": "isi",
    "BUILDUP_INDEX": "bui",
    "FIRE_WEATHER_INDEX": "fwi",
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def find_col(cols, *exact, contains=None, required=True, what="column"):
    """Find a column by exact name first, then by substring. Case-insensitive."""
    upper = {str(c).upper(): c for c in cols}
    for name in exact:
        if name in upper:
            return upper[name]
    if contains:
        for u, original in upper.items():
            if all(part in u for part in contains):
                return original
    if required:
        sys.exit(f"ERROR: could not find the {what} column. Columns present: {list(cols)}")
    return None


def parse_dates(s):
    """Return day-resolution timestamps from datetime, 'YYYYMMDD...' or free-form text."""
    if pd.api.types.is_datetime64_any_dtype(s):
        out = pd.to_datetime(s)
    else:
        text = s.astype(str).str.strip()
        if text.str.fullmatch(r"\d{8,14}").mean() > 0.5:
            out = pd.to_datetime(text.str[:8], format="%Y%m%d", errors="coerce")
        else:
            out = pd.to_datetime(text, errors="coerce")
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out.dt.normalize()


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


# ----------------------------------------------------------------------------
# 1. Fire incidents (labels)
# ----------------------------------------------------------------------------
def load_fires():
    shps = sorted(HERE.rglob("*.shp"))
    if not shps:
        sys.exit("ERROR: no .shp file found under " + str(HERE))
    print(f"[fires] reading {shps[0].relative_to(HERE)}")
    gdf = gpd.read_file(shps[0]).to_crs(4326)      # BC Albers -> lat/lon
    cols = [c for c in gdf.columns if c != "geometry"]
    print(f"[fires] columns: {cols}")

    c_date = find_col(cols, "IGNITION_DATE", "IGN_DATE", contains=["IGN"], what="ignition date")
    c_cause = find_col(cols, "FIRE_CAUSE", contains=["CAUSE"], what="fire cause")
    c_type = find_col(cols, "FIRE_TYPE", contains=["FIRE", "TYP"], required=False)
    c_size = find_col(cols, "CURRENT_SIZE", "SIZE_HA", contains=["SIZE"], required=False)

    df = pd.DataFrame({
        "lat": gdf.geometry.y.values,
        "lon": gdf.geometry.x.values,
        "date": parse_dates(gdf[c_date]).values,
        "cause": gdf[c_cause].astype(str).str.strip().str.lower().values,
        "size_ha": pd.to_numeric(gdf[c_size], errors="coerce").values if c_size else np.nan,
    })
    if c_type:
        ftype = gdf[c_type].astype(str).str.strip()
        print(f"[fires] {c_type} values:\n{ftype.value_counts().to_string()}")
        is_fire = (ftype.str.lower() == "fire").values
        if is_fire.any():
            df = df[is_fire]
        else:
            print("[fires] WARNING: no rows with type 'Fire'; keeping every incident type")

    df = df.dropna(subset=["date", "lat", "lon"])
    df = df[df["date"].dt.year.isin(YEARS) & df["date"].dt.month.isin(SEASON_MONTHS)]
    print(f"[fires] {len(df)} real fires in {YEARS[0]}-{YEARS[-1]}, months {SEASON_MONTHS[0]}-{SEASON_MONTHS[-1]}")
    print(f"[fires] by cause:\n{df['cause'].value_counts().to_string()}")
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. Lightning grid -> daily flash counts per cell, in PST days
# ----------------------------------------------------------------------------
def load_lightning():
    days = pd.date_range(f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-31", freq="D")
    daily = lats = lons = None
    for year in YEARS:
        path = HERE / f"cg_flashes_3hr_0.1-deg_{year}.nc"
        if not path.exists():
            sys.exit(f"ERROR: missing {path.name}")
        ds = xr.open_dataset(path)
        var = "cg_flashes" if "cg_flashes" in ds.data_vars else next(
            v for v in ds.data_vars if ds[v].ndim == 3)
        da = ds[var]
        lat_n = next(d for d in da.dims if d.lower().startswith("lat"))
        lon_n = next(d for d in da.dims if d.lower().startswith("lon"))
        t_n = next(d for d in da.dims if d not in (lat_n, lon_n))

        # Cut out BC with index slices so only that block is read from disk.
        lat_all = da[lat_n].values
        lon_all = da[lon_n].values
        lon_all = np.where(lon_all > 180, lon_all - 360, lon_all)
        iy = np.where((lat_all >= BBOX["lat_min"]) & (lat_all <= BBOX["lat_max"]))[0]
        ix = np.where((lon_all >= BBOX["lon_min"]) & (lon_all <= BBOX["lon_max"]))[0]
        da = da.isel({lat_n: slice(iy.min(), iy.max() + 1), lon_n: slice(ix.min(), ix.max() + 1)})
        da = da.transpose(t_n, lat_n, lon_n)
        y_lats = lat_all[iy.min(): iy.max() + 1]
        y_lons = lon_all[ix.min(): ix.max() + 1]
        flip_lat = y_lats[0] > y_lats[-1]
        flip_lon = y_lons[0] > y_lons[-1]
        if flip_lat:
            y_lats = y_lats[::-1]
        if flip_lon:
            y_lons = y_lons[::-1]

        if daily is None:
            lats, lons = y_lats, y_lons
            daily = np.zeros((len(days), len(lats), len(lons)), dtype="float32")
        elif not (np.allclose(lats, y_lats) and np.allclose(lons, y_lons)):
            sys.exit(f"ERROR: the lightning grid in {path.name} differs from earlier years")

        # Shift UTC to PST, then add each 3-hour slice into its PST day.
        times = pd.to_datetime(da[t_n].values) - pd.Timedelta(hours=PST_OFFSET_HOURS)
        day_idx = np.asarray((times.normalize() - days[0]).days)
        step = 240                                   # 30 days of 3-hour slices at a time
        for start in range(0, len(times), step):
            block = da.isel({t_n: slice(start, start + step)}).values.astype("float32")
            block = np.nan_to_num(block)             # masked (no coverage) -> 0
            if flip_lat:
                block = block[:, ::-1, :]
            if flip_lon:
                block = block[:, :, ::-1]
            idx = day_idx[start:start + step]
            ok = (idx >= 0) & (idx < len(days))
            np.add.at(daily, idx[ok], block[ok])
        ds.close()
        y0 = days.get_loc(pd.Timestamp(f"{year}-01-01"))
        y1 = days.get_loc(pd.Timestamp(f"{year}-12-31"))
        print(f"[lightning] {year}: {daily[y0:y1 + 1].sum():,.0f} cloud-to-ground flashes in the BC box")
    print(f"[lightning] grid: {len(lats)} x {len(lons)} cells, {len(days)} days")
    return daily, lats, lons, days


def neighbourhood_sum(daily):
    """Sum of each cell and its 8 neighbours (about a 30 km square)."""
    padded = np.pad(daily, ((0, 0), (1, 1), (1, 1)))
    out = np.zeros_like(daily)
    ny, nx = daily.shape[1:]
    for dy in range(3):
        for dx in range(3):
            out += padded[:, dy:dy + ny, dx:dx + nx]
    return out


def window_sum(arr, t, iy, ix, lags):
    """Sum arr[t - lag, iy, ix] over the given lags; days before the start count as 0."""
    total = np.zeros(len(t), dtype="float32")
    for lag in lags:
        tt = t - lag
        ok = tt >= 0
        total[ok] += arr[tt[ok], iy[ok], ix[ok]]
    return total


# ----------------------------------------------------------------------------
# 3. Weather stations and observations
# ----------------------------------------------------------------------------
def load_stations():
    files = sorted(HERE.glob("*_BCWS_WX_STATIONS.csv"))
    if not files:
        sys.exit("ERROR: no *_BCWS_WX_STATIONS.csv files found")
    st = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    c_code = find_col(st.columns, "STATION_CODE", contains=["CODE"], what="station code")
    c_lat = find_col(st.columns, "LATITUDE", contains=["LAT"], what="station latitude")
    c_lon = find_col(st.columns, "LONGITUDE", contains=["LON"], what="station longitude")
    c_elev = find_col(st.columns, "ELEVATION", "ELEVATION_M", contains=["ELEV"], required=False)
    out = pd.DataFrame({
        "STATION_CODE": pd.to_numeric(st[c_code], errors="coerce"),
        "st_lat": pd.to_numeric(st[c_lat], errors="coerce"),
        "st_lon": pd.to_numeric(st[c_lon], errors="coerce"),
        "station_elev_m": pd.to_numeric(st[c_elev], errors="coerce") if c_elev else np.nan,
    }).dropna(subset=["STATION_CODE", "st_lat", "st_lon"])
    out["STATION_CODE"] = out["STATION_CODE"].astype("int64")
    if out["st_lon"].median() > 0:               # west longitudes stored as positive
        out["st_lon"] = -out["st_lon"]
    out = out.drop_duplicates("STATION_CODE", keep="last").reset_index(drop=True)
    print(f"[stations] {len(out)} unique stations")
    return out


def load_weather():
    missing = [y for y in YEARS if not (HERE / f"{y}_BCWS_WX_OBS.csv").exists()]
    if missing:
        base = "https://www.for.gov.bc.ca/ftp/HPR/external/!publish/BCWS_DATA_MART"
        urls = "\n".join(f"    {base}/{y}/{y}_BCWS_WX_OBS.csv" for y in missing)
        sys.exit("ERROR: weather observation files are missing. The STATIONS files only say "
                 "where stations are; the readings are in the OBS files.\nDownload these into "
                 f"{HERE}:\n{urls}")
    want = {"STATION_CODE", "DATE_TIME", *WX_HOURLY, *WX_DAILY}
    frames = []
    for year in YEARS:
        df = pd.read_csv(HERE / f"{year}_BCWS_WX_OBS.csv", usecols=lambda c: c in want, low_memory=False)
        df["date"] = pd.to_datetime(df["DATE_TIME"].astype(str).str[:8], format="%Y%m%d", errors="coerce")
        df["STATION_CODE"] = pd.to_numeric(df["STATION_CODE"], errors="coerce")
        df = df.dropna(subset=["date", "STATION_CODE"])
        df = df[df["date"].dt.month.isin(SEASON_MONTHS)]
        agg = {new: (old, how) for old, (new, how) in WX_HOURLY.items() if old in df.columns}
        agg.update({new: (old, "first") for old, new in WX_DAILY.items() if old in df.columns})
        daily = df.groupby(["STATION_CODE", "date"], as_index=False).agg(**agg)
        frames.append(daily)
        print(f"[weather] {year}: {len(daily):,} station-days")
    wx = pd.concat(frames, ignore_index=True)
    wx["STATION_CODE"] = wx["STATION_CODE"].astype("int64")
    return wx


# ----------------------------------------------------------------------------
# Main: build rows, attach features, write the file
# ----------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    fires = load_fires()
    stations = load_stations()
    weather = load_weather()
    daily, lats, lons, days = load_lightning()
    ny, nx = len(lats), len(lons)
    dlat, dlon = lats[1] - lats[0], lons[1] - lons[0]

    # Nearest K stations for every grid cell.
    cell_lat, cell_lon = np.meshgrid(lats, lons, indexing="ij")
    dist = haversine_km(cell_lat.reshape(-1, 1), cell_lon.reshape(-1, 1),
                        stations["st_lat"].values[None, :], stations["st_lon"].values[None, :])
    k = min(K_STATIONS, len(stations))
    near_idx = np.argsort(dist, axis=1)[:, :k]
    near_km = np.take_along_axis(dist, near_idx, axis=1)
    usable = near_km[:, 0] <= MAX_STATION_KM          # flat index = iy * nx + ix
    print(f"[grid] {usable.sum():,} of {ny * nx:,} cells are within {MAX_STATION_KM} km of a station")

    # Positives: snap each fire to its cell and day.
    fires["iy"] = np.rint((fires["lat"] - lats[0]) / dlat).astype(int)
    fires["ix"] = np.rint((fires["lon"] - lons[0]) / dlon).astype(int)
    inside = fires["iy"].between(0, ny - 1) & fires["ix"].between(0, nx - 1)
    fires = fires[inside].copy()
    fires["cell"] = fires["iy"] * nx + fires["ix"]
    before = len(fires)
    fires = fires[usable[fires["cell"].values]]
    print(f"[grid] dropped {before - len(fires)} fires farther than {MAX_STATION_KM} km from any station")
    fires["t"] = (fires["date"] - days[0]).dt.days
    fires["is_lightning"] = fires["cause"].str.contains("lightning")
    fires["is_person"] = fires["cause"].str.contains("person|human")
    pos = fires.groupby(["t", "cell"], as_index=False).agg(
        n_fires=("cause", "size"),
        n_lightning_fires=("is_lightning", "sum"),
        n_person_fires=("is_person", "sum"),
        max_fire_size_ha=("size_ha", "max"),
    )
    pos["ignition"] = 1

    # Negatives: random cell-days in season with no fire.
    season_t = np.where(days.month.isin(SEASON_MONTHS))[0]
    usable_cells = np.where(usable)[0]
    n_neg = NEG_PER_POS * len(pos)
    neg = pd.DataFrame({
        "t": rng.choice(season_t, size=int(n_neg * 1.1)),
        "cell": rng.choice(usable_cells, size=int(n_neg * 1.1)),
    }).drop_duplicates()
    neg = neg.merge(pos[["t", "cell"]], on=["t", "cell"], how="left", indicator=True)
    neg = neg[neg["_merge"] == "left_only"].drop(columns="_merge").head(n_neg)
    neg["ignition"] = 0
    for col in ("n_fires", "n_lightning_fires", "n_person_fires"):
        neg[col] = 0
    neg["max_fire_size_ha"] = np.nan

    rows = pd.concat([pos, neg], ignore_index=True)
    t = rows["t"].values
    iy, ix = np.divmod(rows["cell"].values, nx)
    rows["date"] = days[t]
    rows["cell_lat"] = np.round(lats[iy], 2)
    rows["cell_lon"] = np.round(lons[ix], 2)
    rows["year"] = rows["date"].dt.year
    rows["day_of_year"] = rows["date"].dt.dayofyear

    # Lightning features.
    nb = neighbourhood_sum(daily)
    for lag in range(4):
        rows[f"cg_d{lag}"] = window_sum(daily, t, iy, ix, [lag])
    rows["cg_sum_prev3d"] = window_sum(daily, t, iy, ix, range(1, 4))
    rows["cg_sum_prev7d"] = window_sum(daily, t, iy, ix, range(1, 8))
    rows["cg_nb_d0"] = window_sum(nb, t, iy, ix, [0])
    rows["cg_nb_sum_prev3d"] = window_sum(nb, t, iy, ix, range(1, 4))
    rows["cg_nb_sum_prev7d"] = window_sum(nb, t, iy, ix, range(1, 8))

    # Weather features: nearest station with data that day.
    wx_cols = [c for c in weather.columns if c not in ("STATION_CODE", "date")]
    st_code = stations["STATION_CODE"].values
    st_elev = stations["station_elev_m"].values
    joined = None
    for rank in range(k):
        s = near_idx[rows["cell"].values, rank]
        tmp = pd.DataFrame({
            "date": rows["date"].values,
            "STATION_CODE": st_code[s],
            "station_dist_km": np.round(near_km[rows["cell"].values, rank], 1),
            "station_elev_m": st_elev[s],
        })
        m = tmp.merge(weather, on=["STATION_CODE", "date"], how="left")
        m = m.astype({"STATION_CODE": "float64"})
        no_data = m[wx_cols].isna().all(axis=1)
        m.loc[no_data, ["STATION_CODE", "station_dist_km", "station_elev_m"]] = np.nan
        joined = m if joined is None else joined.combine_first(m)
    rows = pd.concat([rows, joined.drop(columns="date")], axis=1)

    front = ["date", "year", "day_of_year", "cell_lat", "cell_lon", "ignition",
             "n_fires", "n_lightning_fires", "n_person_fires", "max_fire_size_ha"]
    rows = rows.drop(columns=["t", "cell"])
    rows = rows[front + [c for c in rows.columns if c not in front]]
    rows = rows.sort_values(["date", "cell_lat", "cell_lon"]).reset_index(drop=True)
    rows.to_csv(OUT, index=False)

    print("\n=== done ===")
    print(f"wrote {OUT.name}: {len(rows):,} rows, {rows.shape[1]} columns")
    print(f"ignition rows: {int(rows['ignition'].sum()):,}   no-fire rows: {int((rows['ignition'] == 0).sum()):,}")
    print(f"rows with no weather match: {int(rows['STATION_CODE'].isna().sum()):,}")
    print("mean same-day neighbourhood flashes  fire rows: "
          f"{rows.loc[rows.ignition == 1, 'cg_nb_d0'].mean():.2f}   "
          f"no-fire rows: {rows.loc[rows.ignition == 0, 'cg_nb_d0'].mean():.2f}")


if __name__ == "__main__":
    main()
