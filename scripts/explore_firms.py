"""Filter the archive and explore development years; never profile holdout features.

Run after check_firms_coverage.py. Requires pandas, numpy, matplotlib, shapely.
Raw observations and category codes are retained without quality exclusions.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'exploration'
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(OUT / 'matplotlib-cache'))
sys.path.insert(0, str(ROOT / 'data' / 'check-dependencies'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from shapely import intersects_xy, prepare
from shapely.geometry import shape
from shapely.ops import unary_union


def markdown(frame):
    rows = [list(frame.columns)] + frame.astype(str).values.tolist()
    lines = ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]
    lines.insert(1, '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |')
    return '\n'.join(lines)


def main():
    source = ROOT / 'DL_FIRE_SV-C2_825835' / 'fire_archive_SV-C2_825835.csv'
    boundary_path = ROOT / 'data' / 'checks' / 'bc_legal_boundary.geojson'
    boundary = json.loads(boundary_path.read_text(encoding='utf-8'))
    bc = unary_union([shape(f['geometry']) for f in boundary['features']])
    assert bc.is_valid
    prepare(bc)
    raw_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    raw = pd.read_csv(source)
    dates = pd.to_datetime(raw.acq_date, errors='raise')
    period = dates.dt.year.between(2021, 2024)
    summer = dates.dt.month.between(5, 9)
    inside = intersects_xy(bc, raw.longitude.to_numpy(), raw.latitude.to_numpy())
    selected = raw.loc[period & summer & inside].copy()
    years = pd.to_datetime(selected.acq_date).dt.year
    dev = selected.loc[years <= 2023].copy()
    holdout = selected.loc[years == 2024].copy()
    dev.to_csv(OUT / 'bc_may_sep_2021_2023.csv', index=False)
    holdout.to_csv(OUT / 'bc_may_sep_2024_holdout.csv', index=False)
    assert len(dev) + len(holdout) == len(selected)
    assert set(pd.to_datetime(dev.acq_date).dt.year) == {2021, 2022, 2023}
    assert pd.to_datetime(holdout.acq_date).dt.year.eq(2024).all()
    assert selected.columns.equals(raw.columns)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == raw_hash
    manifest = {
        'source': str(source.relative_to(ROOT)), 'source_sha256': raw_hash,
        'boundary_sha256': hashlib.sha256(boundary_path.read_bytes()).hexdigest(),
        'boundary_rule': 'point centre inside/on BC legal polygon; no buffer',
        'season': 'May 1 through September 30 inclusive, using source acq_date',
        'raw_rows': len(raw), 'study_year_rows': int(period.sum()),
        'study_summer_rows_before_boundary': int((period & summer).sum()),
        'summer_rows_outside_bc': int((period & summer & ~inside).sum()),
        'bc_study_rows_outside_summer': int((period & inside & ~summer).sum()),
        'development_rows': len(dev), 'holdout_rows': len(holdout),
        'quality_filters': 'none',
        'packages': {name: __import__(name).__version__ for name in ['pandas', 'numpy', 'matplotlib', 'shapely']},
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    dev['year'] = pd.to_datetime(dev.acq_date).dt.year
    dev['month'] = pd.to_datetime(dev.acq_date).dt.month
    monthly = dev.groupby(['year', 'month']).size().reindex(
        pd.MultiIndex.from_product([range(2021, 2024), range(5, 10)], names=['year', 'month']), fill_value=0
    ).rename('detections').reset_index()
    monthly.to_csv(OUT / 'monthly_counts.csv', index=False)
    categories = pd.concat([
        dev.groupby(['year', col]).size().rename('detections').reset_index().rename(columns={col: 'value'}).assign(field=col)
        for col in ['confidence', 'type', 'daynight']
    ], ignore_index=True)[['field', 'year', 'value', 'detections']]
    categories.to_csv(OUT / 'category_counts.csv', index=False)
    numeric = dev[['frp', 'brightness', 'bright_t31', 'scan', 'track']].describe(
        percentiles=[.01, .5, .95, .99, .999]).T
    numeric.to_csv(OUT / 'numeric_summary.csv')
    dev.nlargest(50, 'frp').to_csv(OUT / 'highest_frp_observations.csv', index=False)
    quality = {'missing_cells': int(dev.isna().sum().sum()),
               'exact_duplicate_rows': int(dev.duplicated().sum()),
               'nonfinite_numeric_cells': int((~np.isfinite(dev.select_dtypes('number'))).sum().sum()),
               'negative_frp_rows': int(dev.frp.lt(0).sum()), 'zero_frp_rows': int(dev.frp.eq(0).sum())}
    (OUT / 'quality_checks.json').write_text(json.dumps(quality, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for year, group in monthly.groupby('year'):
        axes[0].plot(group.month, group.detections, marker='o', label=str(year))
    axes[0].set(xticks=range(5, 10), xticklabels=['May', 'Jun', 'Jul', 'Aug', 'Sep'], ylabel='Hotspot observations', title='Monthly detections')
    axes[0].legend()
    axes[1].hist(np.log1p(dev.frp), bins=70, color='#b34d23')
    axes[1].set(xlabel='log(1 + source FRP)', ylabel='Observations', title='FRP distribution (all values retained)')
    for col in ['brightness', 'bright_t31']:
        axes[2].hist(dev[col], bins=70, alpha=.5, label=col)
    axes[2].set(xlabel='Source brightness value', ylabel='Observations', title='Brightness distributions')
    axes[2].legend()
    fig.suptitle('BC, May–September 2021–2023 only')
    fig.tight_layout()
    fig.savefig(OUT / 'temporal_and_numeric.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharex=True, sharey=True)
    vmax = 1
    collections = []
    for ax, (year, group) in zip(axes, dev.groupby('year')):
        for polygon in (bc.geoms if bc.geom_type == 'MultiPolygon' else [bc]):
            x, y = polygon.exterior.xy
            ax.plot(x, y, color='#777777', linewidth=.35)
        h = ax.hexbin(group.longitude, group.latitude, gridsize=65, extent=(-139, -114, 48, 60.5), mincnt=1, norm=LogNorm(), cmap='inferno')
        collections.append(h)
        vmax = max(vmax, h.get_array().max())
        ax.set(title=str(year), xlabel='Longitude', xlim=(-139, -114), ylim=(48, 60.5))
        ax.set_aspect(1 / np.cos(np.deg2rad(54)))
    for h in collections:
        h.set_norm(LogNorm(vmin=1, vmax=vmax))
    axes[0].set_ylabel('Latitude')
    fig.colorbar(collections[0], ax=list(axes), label='Observations per display hexagon (log scale)', shrink=.65)
    fig.suptitle('May–September hotspot density; display bins are not a modeling grid')
    fig.savefig(OUT / 'spatial_density.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    yearly = dev.groupby('year').size().rename('detections').reset_index()
    yearly['share_percent'] = (100 * yearly.detections / len(dev)).round(2)
    report = f'''# FIRMS summer exploration

May–September is the agreed analysis season. Raw files remain unchanged. Filter uses source acquisition dates and point centres inside/on the cached BC legal polygon. No confidence, type, FRP, or brightness exclusions have been made.

## Filter and split

- Development, 2021–2023: **{len(dev):,} observations**.
- Reserved 2024 holdout: **{len(holdout):,} observations**; only filtering and split counts checked, no feature profiling or plots.
- Summer 2021–2024 observations outside BC removed: **{manifest['summer_rows_outside_bc']:,}**.
- BC observations in 2021–2024 outside May–September omitted from these derived copies: **{manifest['bc_study_rows_outside_summer']:,}**. Preserve them in the raw source for potential lookback features.

## Development-year counts

{markdown(yearly)}

## Monthly observations

{markdown(monthly)}

![Monthly counts and numeric distributions](../data/exploration/temporal_and_numeric.png)

![Spatial density by year](../data/exploration/spatial_density.png)

The maps use the same logarithmic colour scale. Geographic display hexagons are not equal-area cells or the proposed 1 km modeling grid. Density reflects repeated observations, fire duration, and satellite visibility as well as fire activity; it is not a count of unique fires or a map of ignition risk.

## Categories and numeric checks

{markdown(categories)}

{markdown(numeric.round(3).reset_index().rename(columns={'index': 'field'}))}

Development checks: {quality['missing_cells']} missing cells, {quality['exact_duplicate_rows']} exact duplicate rows, {quality['nonfinite_numeric_cells']} nonfinite numeric cells, {quality['negative_frp_rows']} negative FRP rows, and {quality['zero_frp_rows']} zero FRP rows. These checks do not establish observation coverage. High-value observations are exported for review, not removed. Source category codes and brightness names are retained pending product-documentation verification.

## Interpretation and next decisions

- Compare years separately: pooling can let the busiest year dominate the results.
- These are hotspot observations, not unique fires or confirmed ignitions. Dates without detections do not establish negative labels.
- Choose study region using these development years plus weather coverage; do not use holdout patterns to tune the region or filters.
- Confirm the target (future hotspot activity versus new wildfire occurrence) before building labels. Grid size remains provisional.
- Retain pre-season observations for lagged features; a three-year history for 2021 needs additional pre-2021 data.

## Reproduce

Run `scripts/check_firms_coverage.py` first if the cached boundary is missing, then `scripts/explore_firms.py` with Python and pandas, numpy, matplotlib, shapely installed. The script also supports the existing local `data/check-dependencies` installation. Outputs are local and ignored under `data/exploration/`; this report's image links require those generated files. `manifest.json` records source/boundary SHA-256 hashes, package versions, filter rules, and split counts. CSV summaries and the top 50 FRP observations accompany the plots.
'''
    (ROOT / 'docs' / 'firms-summer-eda.md').write_text(report, encoding='utf-8')
    print(json.dumps(manifest, indent=2))
    print(yearly.to_string(index=False))
    print('Report: docs/firms-summer-eda.md')


if __name__ == '__main__':
    main()
