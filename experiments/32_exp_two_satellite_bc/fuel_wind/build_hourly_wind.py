"""Afternoon station weather per BCWS station and local day, from the BCWS Data Mart hourly observations
(data/bcws_weather/raw_hourly/<year>_BCWS_WX_OBS.csv, DATE_TIME in local standard time, PST = UTC-8).
Window 12:00-16:00 LST = 20:00-24:00 UTC, so a local day d value is known by the end of UTC day d (Exp 25/21 rule:
features for target day t use day t-1).
Per station-day: hw_wind_mean (km/h), hw_gust_max, hw_u / hw_v (mean wind-TO vector, km/h; east / north),
hw_rh_min, hw_temp_max, hw_isi_max (hourly initial spread index), hw_ffmc_max (hourly fine fuel moisture code), hw_hours.
Writes data/bcws_weather/afternoon_2018_2023.parquet. usage: python build_hourly_wind.py REPO_ROOT"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.csv as pv

ROOT = Path(sys.argv[1]); RAW = ROOT / 'data/bcws_weather/raw_hourly'
COLS = ['STATION_CODE', 'DATE_TIME', 'HOURLY_WIND_SPEED', 'HOURLY_WIND_DIRECTION', 'HOURLY_WIND_GUST', 'HOURLY_TEMPERATURE',
        'HOURLY_RELATIVE_HUMIDITY', 'HOURLY_INITIAL_SPREAD_INDEX', 'HOURLY_FINE_FUEL_MOISTURE_CODE']
out = []
for y in range(2018, 2024):
    t = pv.read_csv(RAW / f'{y}_BCWS_WX_OBS.csv', convert_options=pv.ConvertOptions(include_columns=COLS,
                    column_types={'DATE_TIME': 'string', 'STATION_CODE': 'string'})).to_pandas()
    dt = t.DATE_TIME.str
    t['month'] = dt[4:6].astype(int); t['hour'] = dt[8:10].astype(int)
    t = t[t.month.between(4, 9) & t.hour.between(12, 16)].copy()
    t['date'] = pd.to_datetime(t.DATE_TIME.str[:8], format='%Y%m%d')
    sp = pd.to_numeric(t.HOURLY_WIND_SPEED, errors='coerce'); d = np.radians(pd.to_numeric(t.HOURLY_WIND_DIRECTION, errors='coerce'))
    t['u'] = -np.sin(d) * sp; t['v'] = -np.cos(d) * sp  # direction is where wind blows FROM; u/v point where it blows TO
    t['sp'] = sp
    for c in ['HOURLY_WIND_GUST', 'HOURLY_TEMPERATURE', 'HOURLY_RELATIVE_HUMIDITY', 'HOURLY_INITIAL_SPREAD_INDEX', 'HOURLY_FINE_FUEL_MOISTURE_CODE']:
        t[c] = pd.to_numeric(t[c], errors='coerce')
    g = t.groupby(['STATION_CODE', 'date'])
    day = g.agg(hw_wind_mean=('sp', 'mean'), hw_gust_max=('HOURLY_WIND_GUST', 'max'), hw_u=('u', 'mean'), hw_v=('v', 'mean'),
                hw_rh_min=('HOURLY_RELATIVE_HUMIDITY', 'min'), hw_temp_max=('HOURLY_TEMPERATURE', 'max'),
                hw_isi_max=('HOURLY_INITIAL_SPREAD_INDEX', 'max'), hw_ffmc_max=('HOURLY_FINE_FUEL_MOISTURE_CODE', 'max'),
                hw_hours=('sp', 'count')).reset_index()
    out.append(day); print(y, len(day), 'station-days', day.STATION_CODE.nunique(), 'stations', flush=True)
D = pd.concat(out, ignore_index=True); D = D[D.hw_hours >= 3]
D.to_parquet(ROOT / 'data/bcws_weather/afternoon_2018_2023.parquet', index=False)
print(D.describe().round(2).T.to_string())
