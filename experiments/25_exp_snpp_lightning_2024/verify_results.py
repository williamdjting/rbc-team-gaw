"""Independently recount saved 2024 predictions and check the frozen artifacts."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / 'results'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    frozen = json.loads((RESULTS/'frozen_before_2024.json').read_text())
    complete = json.loads((RESULTS/'test_completion.json').read_text())
    assert complete['protocol_sha256'] == digest(RESULTS/'frozen_before_2024.json')
    for path, expected in frozen['source_hashes'].items():
        assert digest(ROOT/path) == expected, path
    with (RESULTS/'metrics_2024.csv').open() as stream:
        metrics = {(r['region'],r['model']):r for r in csv.DictReader(stream)}
    assert len(metrics) == 12
    total_rows = 0
    totals = {name:np.zeros(4,np.int64) for name in ('mlp_snpp_lightning','snpp_persistence')}
    for region, threshold in frozen['thresholds'].items():
        slug = region.lower().replace(' ','_')
        path = ROOT/'data/snpp_lightning_h3r7/evaluation/2024'/f'{slug}_snpp_lightning_full_grid.parquet'
        table = pq.ParquetFile(path)
        grid = pq.ParquetFile(ROOT/'experiments/02_hotspot_persistence_bc/regions'/slug/'data/h3_grid.parquet')
        assert table.metadata.num_rows == grid.metadata.num_rows * 146
        rows = 0
        counts = {name:np.zeros(4,np.int64) for name in totals}
        for batch in table.iter_batches(batch_size=250000, columns=['target_snpp_filled','mlp_score','snpp_persistence']):
            y = batch.column(0).to_numpy(zero_copy_only=False)
            score = batch.column(1).to_numpy(zero_copy_only=False)
            baseline = batch.column(2).to_numpy(zero_copy_only=False)
            assert np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all()
            for name, flag in [('mlp_snpp_lightning', score >= threshold),('snpp_persistence',baseline)]:
                counts[name] += [np.count_nonzero(y & flag), np.count_nonzero(~y & flag),
                                 np.count_nonzero(y & ~flag), np.count_nonzero(~y & ~flag)]
            rows += len(y)
        for name in counts:
            row = metrics[region,name]
            assert rows == int(row['rows'])
            assert list(counts[name]) == [int(row[k]) for k in ('tp','fp','fn','tn')]
            if name == 'mlp_snpp_lightning':
                assert float(row['threshold']) == threshold
            totals[name] += counts[name]
        total_rows += rows
    with (RESULTS/'summary_2024.csv').open() as stream:
        for row in csv.DictReader(stream):
            assert list(totals[row['model']]) == [int(row[k]) for k in ('tp','fp','fn','tn')]
            tp,fp,fn,_ = totals[row['model']]
            assert np.isclose(float(row['f1']),2*tp/(2*tp+fp+fn))
    for year in (2022,2023):
        with (RESULTS/f'metrics_{year}.csv').open() as stream:
            for row in csv.DictReader(stream):
                if row['model']=='mlp_snpp_lightning':
                    assert float(row['threshold']) == frozen['thresholds'][row['region']]
    report = dict(status='passed',full_grid_rows=total_rows,regions=6,
                  frozen_hashes_verified=len(frozen['source_hashes']),
                  prediction_confusion_counts_recomputed=True,thresholds_unchanged_since_2022=True)
    (RESULTS/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
