# © SleepECG developers
#
# License: BSD (3-clause)

"""Utilities for runtime and detection quality benchmarks."""

import time
from typing import Any, Dict, Iterator

import biosppy
import ecgdetectors
import heartpy
import heartpy.exceptions
import mne
import neurokit2
import numpy as np
import wfdb.processing

import sleepecg
from sleepecg.io.ecg_readers import ECGRecord


def reader_dispatch(db_slug: str, data_dir: str) -> Iterator[ECGRecord]:
    """
    Read ECG records from mitdb, ltdb or gudb.

    Parameters
    ----------
    db_slug : str
        Short identifier of a dataset, e.g. `'mitdb'`.
    data_dir : str
        Directory where all datasets are stored.

    Yields
    ------
    ECGRecord
        Each element in the generator is of type `ECGRecord` and contains the ECG signal
        (`.ecg`), sampling frequency (`.fs`), annotated beat indices (`.annotations`),
        `.lead`, and `.id`.
    """
    readers = {
        "gudb": sleepecg.read_gudb,
        "ltdb": sleepecg.read_ltdb,
        "mitdb": sleepecg.read_mitdb,
    }
    if db_slug not in readers:
        raise ValueError(f"Invalid db_slug: {db_slug}")
    yield from readers[db_slug](data_dir=data_dir)


def detector_dispatch(ecg: np.ndarray, fs: float, detector: str) -> np.ndarray:
    """
    Provide a common interface for different heartbeat detectors.

    Parameters
    ----------
    ecg : np.ndarray
        ECG signal.
    fs : float
        Sampling frequency of the ECG signal in Hz.
    detector : str
        String identifier of the detector to be used.

    Returns
    -------
    np.ndarray
        Indices of detected heartbeats.
    """
    if detector == "mne":
        detection = mne.preprocessing.ecg.qrs_detector(fs, ecg, verbose=False)
    elif detector == "wfdb-xqrs":
        detection = wfdb.processing.xqrs_detect(ecg, fs, verbose=False)
    elif detector == "pyecg-pantompkins":
        detection = ecgdetectors.Detectors(fs).pan_tompkins_detector(ecg)
    elif detector == "biosppy-hamilton":
        detection = biosppy.signals.ecg.hamilton_segmenter(ecg, fs)[0]
    elif detector == "heartpy":
        wd, m = heartpy.process(ecg, fs)
        detection = np.array(wd["peaklist"])[wd["binary_peaklist"].astype(bool)]
    elif detector == "neurokit2-nk":
        clean_ecg = neurokit2.ecg.ecg_clean(ecg, int(fs), method="neurokit")
        detection = neurokit2.ecg.ecg_findpeaks(clean_ecg, int(fs), method="neurokit")[
            "ECG_R_Peaks"
        ]
    elif detector == "neurokit2-kalidas2017":
        clean_ecg = neurokit2.ecg.ecg_clean(ecg, int(fs), method="kalidas2017")
        detection = neurokit2.ecg.ecg_findpeaks(clean_ecg, int(fs), method="kalidas2017")[
            "ECG_R_Peaks"
        ]
    elif detector == "sleepecg-c":
        detection = sleepecg.detect_heartbeats(ecg, fs, backend="c")
    elif detector == "sleepecg-numba":
        detection = sleepecg.detect_heartbeats(ecg, fs, backend="numba")
    elif detector == "sleepecg-python":
        detection = sleepecg.detect_heartbeats(ecg, fs, backend="python")
    else:
        raise ValueError(f"Unknown QRS detector: {detector}")
    return np.asarray(detection)


def evaluate_single(
    record: ECGRecord,
    detector: str,
    signal_len: int,
    max_distance: float,
    calc_rri_similarity: bool,
) -> Dict[str, Any]:
    """
    Evaluate a heartbeat detector on a given annotated ECG record.

    Optionally, similarity measures between detected and annotated RR intervals can be
    calculated. As this requires interpolation, it may take some time for long signals.

    Parameters
    ----------
    record : ECGRecord
        As received from `reader_dispatch`.
    detector : str
        String identifier of the detector to be used.
    signal_len : int
        Length to which the signal should be sliced.
    max_distance : float
        Maximum temporal distance in seconds between detected and annotated beats to count
        as a successful detection.
    calc_rri_similarity : bool
        If `True`, calculate similarity measures between detected and annotated RR intervals
        (computationally expensive for long signals).

    Returns
    -------
    dict[str, Any]
        A dictionary containing evaluation results.
    """
    signal_len_samples = int(signal_len * record.fs * 60)
    ecg = record.ecg[:signal_len_samples]
    annotation = record.annotation[record.annotation < signal_len_samples]
    fs = int(record.fs)

    try:
        start = time.perf_counter()
        detection = detector_dispatch(ecg, fs, detector)
        runtime = time.perf_counter() - start
        TP, FP, FN = sleepecg.compare_heartbeats(
            detection,
            annotation,
            int(max_distance * record.fs),
        )

        if calc_rri_similarity:
            pearsonr, spearmanr, rmse = sleepecg.rri_similarity(detection, annotation)

    except heartpy.exceptions.BadSignalWarning:
        runtime = np.nan
        TP = []
        FP = []
        FN = annotation

        if calc_rri_similarity:
            pearsonr = np.nan
            spearmanr = np.nan
            rmse = np.nan

    result = {
        "record_id": record.id,
        "lead": record.lead,
        "fs": record.fs,
        "num_samples": len(ecg),
        "detector": detector,
        "max_distance": max_distance,
        "runtime": runtime,
        "TP": len(TP),
        "FP": len(FP),
        "FN": len(FN),
    }
    if calc_rri_similarity:
        result.update(
            {
                "pearsonr": pearsonr,
                "spearmanr": spearmanr,
                "rmse": rmse,
            }
        )
    return result
