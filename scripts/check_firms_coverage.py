"""Check local FIRMS dates and point membership against DataBC's legal BC boundary."""
import json
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'checks'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'data' / 'check-dependencies'))
from shapely import intersects_xy, prepare
from shapely.geometry import shape
from shapely.ops import unary_union
SERVICE = 'https://delivery.maps.gov.bc.ca/arcgis/rest/services/mpcm/bcgwpub/MapServer/537'
boundary_file = OUT / 'bc_legal_boundary.geojson'
if not boundary_file.exists():
    response = requests.get(SERVICE + '/query', params={
        'where': '1=1', 'outFields': '*', 'returnGeometry': 'true',
        'outSR': '4326', 'f': 'geojson',
    }, timeout=60)
    response.raise_for_status()
    boundary = response.json()
    if 'error' in boundary or not boundary.get('features') or boundary.get('exceededTransferLimit'):
        raise ValueError(f'Incomplete or invalid boundary response: {str(boundary)[:500]}')
    boundary_file.write_text(json.dumps(boundary), encoding='utf-8')
boundary = json.loads(boundary_file.read_text(encoding='utf-8'))
bc = unary_union([shape(feature['geometry']) for feature in boundary['features']])
if not bc.is_valid:
    raise ValueError('Invalid boundary geometry')
prepare(bc)
report = {'boundary_source': SERVICE, 'boundary_rule': 'point centre inside or on polygon boundary; no buffer',
          'request_dates': 'Not included in ZIP; must confirm from original FIRMS order', 'files': []}
for file in sorted((ROOT / 'DL_FIRE_SV-C2_825835').glob('*.csv')):
    frame = pd.read_csv(file)
    points = frame[['longitude', 'latitude']].to_numpy()
    inside = intersects_xy(bc, points[:, 0], points[:, 1])
    dates = pd.to_datetime(frame.acq_date, errors='raise')
    study = dates.between('2021-01-01', '2024-12-31')
    frame.loc[~inside].to_csv(OUT / (file.stem + '_outside_bc.csv'), index=False)
    monthly = frame.assign(month=dates.dt.to_period('M').astype(str), inside_bc=inside)
    monthly.groupby(['month', 'inside_bc']).size().rename('detections').to_csv(
        OUT / (file.stem + '_monthly_counts.csv'))
    item = {'file': file.name, 'rows': len(frame), 'first_detection': str(dates.min().date()),
            'last_detection': str(dates.max().date()), 'inside_bc': int(inside.sum()),
            'outside_bc': int((~inside).sum()), 'study_period_rows': int(study.sum()),
            'study_period_inside_bc': int((study & inside).sum()),
            'study_period_outside_bc': int((study & ~inside).sum())}
    report['files'].append(item)
    print(json.dumps(item), flush=True)
(OUT / 'firms_coverage_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('Outputs:', OUT)
