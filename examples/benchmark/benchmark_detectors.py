# © SleepECG developers
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

if len(sys.argv) == 1:
    print('No benchmark specified, executing "runtime" benchmark.')
    benchmark = "runtime"
elif len(sys.argv) > 2:
    print("Usage: python benchmark_detectors.py [<benchmark>]")
    exit()
else:
    benchmark = sys.argv[1]

with open("config.yml") as config_file:
    cfg = yaml.safe_load(config_file)

try:
    cfg = cfg[benchmark]
except KeyError:
    raise ValueError(f"Invalid benchmark: {benchmark!r}, available: {list(cfg)}.") from None


if cfg.get("suppress_warnings", False):
    warnings.filterwarnings("ignore")

outfile_dir = Path(cfg.get("outfile_dir", "."))
outfile_dir.mkdir(parents=True, exist_ok=True)

db_slug = cfg["db_slug"]
timestamp = time.strftime("%Y_%m_%d__%H_%M_%S")
csv_filepath = outfile_dir / f"{benchmark}__{db_slug}__{timestamp}.csv"
print(f"Storing results to {csv_filepath.resolve()}.")

data_dir = Path(cfg.get("data_dir", "~/.sleepecg/datasets")).expanduser()
records = list(reader_dispatch(db_slug, data_dir))
print(f"Loaded {len(records)} records from {db_slug}.")

if cfg.get("export_records", False):
    from sleepecg import export_ecg_record

    print(f"Exporting records to {outfile_dir.resolve()}.")
    for record in records:
        export_ecg_record(record, outfile_dir / f"{record.id}-{record.lead}.txt")

fieldnames = [
    "record_id",
    "lead",
    "fs",
    "num_samples",
    "detector",
    "max_distance",
    "runtime",
    "TP",
    "FP",
    "FN",
]
if cfg.get("calc_rri_similarity", False):
    fieldnames += ["pearsonr", "spearmanr", "rmse"]

# Trigger jit compilation to make runtime benchmarks representative
if "sleepecg-numba" in cfg["detectors"]:
    detector_dispatch(records[0].ecg[: 10 * records[0].fs], records[0].fs, "sleepecg-numba")


with open(csv_filepath, "w", newline="") as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
    )
    writer.writeheader()
    for signal_len in cfg["signal_lengths"]:
        print(f"==== Signal length: {signal_len} minutes ====")
        for detector in cfg["detectors"]:
            detector_results = []
            for record in tqdm(records, desc=detector, leave=False, disable=True):
                if len(record.ecg) < signal_len * record.fs * 60:
                    continue

                detector_results.append(
                    evaluate_single(
                        record,
                        detector,
                        signal_len,
                        cfg.get("max_distance", 0.1),
                        cfg.get("calc_rri_similarity", False),
                    ),
                )

                writer.writerow(detector_results[-1])
                csv_file.flush()
            mean_runtime = np.nanmean([x["runtime"] for x in detector_results])
            signal_mins_per_runtime_sec = signal_len / mean_runtime
            print(
                f"  mean runtime: {mean_runtime:8.5f}s -> "
                f"{signal_mins_per_runtime_sec:4.0f} minutes analyzed per second # "
                f"{detector}"
            )
