# Authors: Florian Hofer
#
# License: BSD (3-clause)

"""Functions for reading datasets containing ECG and beat annotations."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np
import requests
from tqdm import tqdm

from .physionet_api import download_physionet_records, list_physionet_records
from .utils import download_file

__all__ = [
    'read_mitbih',
    'read_gudb',
]


@dataclass
class ECGRecord:
    """
    Dataclass to store a single ECG record.

    Attributes
    ----------
    ecg : np.ndarray
        The ECG signal.
    fs : float
        The sampling frequency.
    annotation : np.ndarray
        Indices of annotated heartbeats.
    lead : str, optional
        Which ECG lead the signal was recorded from, by default `None`.
    id : str, optional
        The record's ID, by default `None`.
    """

    ecg: np.ndarray
    fs: float
    annotation: np.ndarray
    lead: Optional[str] = None
    id: Optional[str] = None


def read_mitbih(
    data_dir: Union[str, Path],
    db_slug: str,
    records_pattern: str = '*',
    offline: bool = False,
) -> Iterator[ECGRecord]:
    """
    Lazily reads records from MIT-BIH datasets (e.g. MITDB, LTDB).

    Required files are downloaded if not present in `<data_dir>/<db_slug>`.

    Parameters
    ----------
    data_dir : str | pathlib.Path
        Directory where all datasets are stored.
    db_slug : str
        Short identifier of a database, e.g. `'mitdb'`.
    records_pattern : str, optional
        Glob-like pattern to select record IDs, by default `'*'`.
    offline : bool, optional
        If `True`, only local files will be used (i.e. no files will be
        downloaded), by default `False`.

    Yields
    ------
    Iterator[ECGRecord]
        Each element in the generator is of type `ECGRecord` and contains
        the ECG signal (`.ecg`), sampling frequency (`.fs`), annotated beat
        indices (`.annotations`), `.lead`, and `.id`.
    """
    import wfdb

    # https://archive.physionet.org/physiobank/database/html/mitdbdir/intro.htm#symbols
    BEAT_ANNOTATION_SYMBOLS = set('NLRAaJSFejE/fQ|')

    data_dir = Path(data_dir)

    requested_records = list_physionet_records(data_dir, db_slug, pattern=records_pattern)

    if not offline:
        download_physionet_records(
            data_dir,
            db_slug,
            requested_records,
            extensions=['.hea', '.dat', '.atr'],
        )

    for record_id in requested_records:
        record_file = str(data_dir / db_slug / record_id)

        annotations = wfdb.rdann(record_file, 'atr')
        beat_indices = []
        for sample, symbol in zip(annotations.sample, annotations.symbol):
            if symbol in BEAT_ANNOTATION_SYMBOLS:
                beat_indices.append(sample)

        record = wfdb.rdrecord(record_file)

        for signal_index, signal_name in enumerate(record.sig_name):
            yield ECGRecord(
                ecg=record.p_signal[:, signal_index].flatten(),
                fs=record.fs,
                annotation=np.array(beat_indices, dtype=int),
                lead=signal_name,
                id=record_id,
            )


def read_gudb(data_dir: Union[str, Path], offline: bool = False) -> Iterator[ECGRecord]:
    """
    Lazily reads records from GUDB (https://berndporr.github.io/ECG-GUDB/).

    Required files are downloaded if not present in `'<data_dir>/gudb'`.

    Parameters
    ----------
    data_dir : str | pathlib.Path
        Directory where all datasets are stored.
    offline : bool, optional
        If `True`, only local files will be used (i.e. no files will be
        downloaded), by default `False`.

    Yields
    ------
    Iterator[ECGRecord]
        Each element in the generator is of type `ECGRecord` and contains
        the ECG signal (`.ecg`), sampling frequency (`.fs`), annotated beat
        indices (`.annotations`), `.lead`, and `.id`.
    """
    import pandas as pd

    DB_URL = 'https://berndporr.github.io/ECG-GUDB/experiment_data'
    EXPERIMENTS = ['sitting', 'maths', 'walking', 'hand_bike', 'jogging']
    FS = 250

    db_dir = Path(data_dir) / 'gudb'

    for subject_id in tqdm(list(range(25)), desc='Reading GUDB'):
        for experiment in EXPERIMENTS:
            experiment_subdir = f'subject_{subject_id:02}/{experiment}'
            if not offline:
                for tsv_filename in (
                    'ECG.tsv',
                    'annotation_cs.tsv',
                    'annotation_cables.tsv',
                ):
                    ecg_file_url = f'{DB_URL}/{experiment_subdir}/{tsv_filename}'
                    target_filepath = db_dir / experiment_subdir / tsv_filename
                    try:
                        download_file(ecg_file_url, target_filepath)
                    except requests.exceptions.HTTPError as error:
                        print(error)
            ecg_data = pd.read_csv(
                db_dir / experiment_subdir / 'ECG.tsv',
                sep=' ',  # contrary to what .tsv suggests, the data is space-separated
                names=['chest', 'II', 'III', 'X', 'Y', 'Z'],
            )
            annotations_chest_file = db_dir / experiment_subdir / 'annotation_cs.tsv'
            if annotations_chest_file.is_file():
                yield ECGRecord(
                    ecg=ecg_data['chest'].to_numpy(),
                    fs=FS,
                    annotation=np.loadtxt(annotations_chest_file, dtype=np.int32),
                    lead='chest',
                    id=f'{subject_id:02}_{experiment}',
                )
            annotations_chest_file = db_dir / experiment_subdir / 'annotation_cables.tsv'
            if annotations_chest_file.is_file():
                annotations = np.loadtxt(annotations_chest_file, dtype=np.int32)
                for lead in ('II', 'III'):
                    yield ECGRecord(
                        ecg=ecg_data[lead].to_numpy(),
                        fs=FS,
                        annotation=annotations,
                        lead=lead,
                        id=f'{subject_id:02}_{experiment}',
                    )
