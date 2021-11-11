# Authors: Florian Hofer
#
# License: BSD (3-clause)

"""Read datasets containing ECG data and sleep stage annotations."""

import datetime
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Iterator, NamedTuple, Optional, Union
from xml.etree import ElementTree

import numpy as np

from ..heartbeats import detect_heartbeats
from .nsrr import get_nsrr_url, list_nsrr
from .utils import download_file

__all__ = [
    'SleepRecord',
    'SleepStage',
    'read_mesa',
]


class SleepStage(IntEnum):
    """
    Mapping of AASM sleep stages to integers.

    To facilitate hypnogram plotting, values increase with wakefulness.
    """

    WAKE = 5
    REM = 4
    N1 = 3
    N2 = 2
    N3 = 1
    UNDEFINED = -1


@dataclass
class SleepRecord:
    """
    Dataclass to store a single sleep record.

    Attributes
    ----------
    heartbeat_times : np.ndarray
        Times of heartbeats after ECG signal onset in seconds.
    sleep_stages : np.ndarray
        Sleep stages according to AASM guidelines, stored as integers as
        defined by `SleepStage`.
    fs_sleep_stages : float
        Sampling frequency of attribute `sleep_stages`.
    id : Optional[str] = ''
        The record's ID, by default `''`.
    recording_start_time : datetime.time, optional
        Time at which the recording was started, by default `None`.
    """

    heartbeat_times: np.ndarray
    sleep_stages: np.ndarray
    fs_sleep_stages: float
    id: Optional[str] = ''
    recording_start_time: Optional[datetime.time] = None


class _ParseNsrrXmlResult(NamedTuple):
    sleep_stages: np.ndarray
    fs_sleep_stages: float
    recording_start_time: datetime.time


def _parse_nsrr_xml(xml_filepath: Path) -> _ParseNsrrXmlResult:
    """
    Parse NSRR xml sleep stage annotation file.

    Parameters
    ----------
    xml_filepath : pathlib.Path
        Path of the annotation file to read.

    Returns
    -------
    sleep_stages : np.ndarray
        Sleep stages according to AASM guidelines, stored as integers as
        defined by `SleepStage`.
    fs_sleep_stages : float
        Sampling frequency of `sleep_stages`.
    recording_start_time : datetime.time
        Time at which the recording was started.

    """
    STAGE_MAPPING = {
        'Wake|0': SleepStage.WAKE,
        'Stage 1 sleep|1': SleepStage.N1,
        'Stage 2 sleep|2': SleepStage.N2,
        'Stage 3 sleep|3': SleepStage.N3,
        'Stage 4 sleep|4': SleepStage.N3,
        'REM sleep|5': SleepStage.REM,
        'Unscored|9': SleepStage.UNDEFINED,
    }

    tree = ElementTree.parse(xml_filepath)
    root = tree.getroot()

    epoch_length = root.findtext('EpochLength')
    if epoch_length is None:
        raise RuntimeError(f'EpochLength not found in {xml_filepath}.')
    epoch_length = int(epoch_length)

    start_time = None
    annot_stages = []

    for event in root.find('ScoredEvents'):
        if event.find('EventConcept').text == 'Recording Start Time':
            start_time = event.find('ClockTime').text.split()[1]
            start_time = datetime.datetime.strptime(start_time, '%H.%M.%S').time()

        if event.find('EventType').text == 'Stages|Stages':
            epoch_duration = int(float(event.findtext('Duration')))
            stage = STAGE_MAPPING[event.findtext('EventConcept')]
            annot_stages.extend([stage] * int(epoch_duration / epoch_length))

    if start_time is None:
        raise RuntimeError(f'"Recording Start Time" not found in {xml_filepath}.')

    return _ParseNsrrXmlResult(
        np.array(annot_stages, dtype=np.int8),
        1 / epoch_length,
        start_time,
    )


def read_mesa(
    data_dir: Union[str, Path],
    records_pattern: str = '*',
    use_preprocessed_heartbeats: bool = True,
    offline: bool = False,
    persist_edfs: bool = False,
) -> Iterator[SleepRecord]:
    """
    Lazily reads records from MESA (https://sleepdata.org/datasets/mesa).

    Each MESA record consists of a `.edf` files containing raw
    polysomnography data and an `.xml` files containing annotated events.
    Since the entire MESA dataset requires about 385 GB of disk space,
    `.edf` files can be deleted after the heartbeat times have been
    extracted. Heartbeat times are stored to a `.npy` file in
    `<data_dir>/mesa/preprocessed/heartbeats`.

    Parameters
    ----------
    data_dir : str | pathlib.Path
        Directory where all datasets are stored.
    records_pattern : str, optional
         Glob-like pattern to select record IDs, by default `'*'`.
    use_preprocessed_heartbeats : bool, optional
        If `True`, get heartbeat times directly from the stored `.npy`
        file, so `.edf` files are only downloaded for records which have
        not yet been preprocessed. By default `True`.
    offline : bool, optional
        If `True`, search for local files only instead of using the NSRR
        API, by default `False`.
    persist_edfs : bool, optional
        If `False`, remove `.edf` after heartbeat detection, by default
        `False`.

    Yields
    ------
    SleepRecord
        Each element in the generator is a `SleepRecord`, containing
        heartbeat times, sleep stages, sleep stage sampling frequency,
        record ID and record start time.
    """
    from mne.io import read_raw_edf

    DB_SLUG = 'mesa'
    ANNOTATION_DIRNAME = 'polysomnography/annotations-events-nsrr'
    EDF_DIRNAME = 'polysomnography/edfs'
    HEARTBEATS_DIRNAME = 'preprocessed/heartbeats'

    if not offline:
        download_url = get_nsrr_url(DB_SLUG)

    db_dir = Path(data_dir) / DB_SLUG
    annotations_dir = db_dir / ANNOTATION_DIRNAME
    edf_dir = db_dir / EDF_DIRNAME
    heartbeats_dir = db_dir / HEARTBEATS_DIRNAME

    for directory in (annotations_dir, edf_dir, heartbeats_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if not offline:
        checksums = {}
        xml_files = list_nsrr(
            DB_SLUG,
            ANNOTATION_DIRNAME,
            f'mesa-sleep-{records_pattern}-nsrr.xml',
            shallow=True,
        )
        checksums.update(xml_files)
        requested_records = [Path(file).stem[:-5] for file, _ in xml_files]

        edf_files = list_nsrr(
            DB_SLUG,
            EDF_DIRNAME,
            f'mesa-sleep-{records_pattern}.edf',
            shallow=True,
        )
        checksums.update(edf_files)
    else:
        xml_files = sorted(annotations_dir.glob(f'mesa-sleep-{records_pattern}-nsrr.xml'))
        requested_records = [file.stem[:-5] for file in xml_files]
        if not use_preprocessed_heartbeats:
            edf_files = sorted(edf_dir.glob(f'mesa-sleep-{records_pattern}.edf'))

    for record_id in requested_records:
        heartbeats_file = heartbeats_dir / f'{record_id}.npy'
        if use_preprocessed_heartbeats and heartbeats_file.is_file():
            heartbeat_times = np.load(heartbeats_file)
        else:
            edf_filename = EDF_DIRNAME + f'/{record_id}.edf'
            edf_filepath = db_dir / edf_filename
            edf_was_available = edf_filepath.is_file()
            if not offline:
                download_file(
                    download_url + edf_filename,
                    edf_filepath,
                    checksums[edf_filename],
                    'md5',
                )

            rec = read_raw_edf(edf_filepath, verbose=False)
            ecg = rec.get_data('EKG').ravel()
            fs = rec.info['sfreq']
            heartbeat_indices = detect_heartbeats(ecg, fs)
            heartbeat_times = heartbeat_indices / fs
            np.save(heartbeats_file, heartbeat_times)

            if not edf_was_available and not persist_edfs:
                edf_filepath.unlink()

        xml_filename = ANNOTATION_DIRNAME + f'/{record_id}-nsrr.xml'
        xml_filepath = db_dir / xml_filename
        if not offline:
            download_file(
                download_url + xml_filename,
                xml_filepath,
                checksums[xml_filename],
                'md5',
            )

        parsed_xml = _parse_nsrr_xml(xml_filepath)

        yield SleepRecord(
            heartbeat_times,
            parsed_xml.sleep_stages,
            parsed_xml.fs_sleep_stages,
            id=record_id,
            recording_start_time=parsed_xml.recording_start_time,
        )
