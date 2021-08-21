# Authors: Florian Hofer
#
# License: BSD (3-clause)

"""Runtime and detection quality benchmarks for heartbeat detectors."""

import csv
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm
from utils import detector_dispatch, evaluate_single, reader_dispatch

if len(sys.argv) != 2:
    print('Usage: python benchmark_detectors.py <benchmark>')
    exit()

with open('config.yml') as config_file:
    cfg = yaml.safe_load(config_file)

benchmark = sys.argv[1]
try:
    cfg = cfg[benchmark]
except KeyError:
    raise ValueError(f'Invalid benchmark: {benchmark!r}, available: {list(cfg)}.') from None


if cfg.get('suppress_warnings', False):
    warnings.filterwarnings('ignore')

outfile_dir = Path(cfg['outfile_dir'])
outfile_dir.mkdir(parents=True, exist_ok=True)

timestamp = time.strftime('%Y_%m_%d__%H_%M_%S')
csv_filepath = outfile_dir / f'{benchmark}__{timestamp}.csv'
print(f'Storing results to {csv_filepath.resolve()}')

records = list(reader_dispatch(cfg['data_dir'], cfg['db_slug']))
print(f'Loaded {len(records)} records from {cfg["db_slug"]}.')


fieldnames = [
    'record_id', 'lead', 'fs', 'num_samples', 'detector', 'max_distance', 'runtime', 'TP',
    'FP', 'FN', 'error_message',
]
if cfg.get('calc_rri_similarity', False):
    fieldnames += ['pearsonr', 'spearmanr', 'rmse']

# Trigger jit compilation to make runtime benchmarks representative
if 'sleepecg-numba' in cfg['detectors']:
    detector_dispatch(records[0].ecg[:10*records[0].fs], records[0].fs, 'sleepecg-numba')


with open(csv_filepath, 'w', newline='') as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
    )
    writer.writeheader()
    for signal_len in cfg['signal_lengths']:
        print(f'==== Signal length: {signal_len} minutes ====')
        for detector in cfg['detectors']:
            detector_results = []
            for record in tqdm(records, desc=detector, leave=False, disable=True):
                if len(record.ecg) < signal_len * record.fs*60:
                    continue

                detector_results.append(
                    evaluate_single(
                        record,
                        detector,
                        signal_len,
                        cfg.get('max_distance', 0.1),
                        cfg.get('calc_rri_similarity', False),
                    ),
                )

                writer.writerow(detector_results[-1])
                csv_file.flush()
            mean_runtime = np.nanmean([x['runtime'] for x in detector_results])
            signal_mins_per_runtime_sec = signal_len / mean_runtime
            print(f'  mean runtime: {mean_runtime:8.5f}s -> {signal_mins_per_runtime_sec:4.0f} minutes analyzed per second # {detector}')  # noqa
