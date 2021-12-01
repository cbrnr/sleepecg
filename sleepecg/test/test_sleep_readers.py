# Authors: Florian Hofer
#
# License: BSD (3-clause)

"""Tests for sleep data reader functions."""

import warnings
from pathlib import Path
from typing import List

import numpy as np
import scipy.misc
from pyedflib import highlevel

from sleepecg.io import read_mesa
from sleepecg.io.sleep_readers import _SleepStage


def _dummy_mesa_edf(filename: str, hours: float):
    ECG_FS = 360
    ecg_5_min = scipy.misc.electrocardiogram()
    seconds = int(hours * 60 * 60)
    ecg = np.tile(ecg_5_min, int(np.ceil(seconds / 300)))[np.newaxis, :seconds * ECG_FS]
    signal_headers = highlevel.make_signal_headers(['EKG'], sample_frequency=ECG_FS)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        highlevel.write_edf(filename, ecg, signal_headers)


def _dummy_mesa_xml(filename: str, hours: float, random_state: int):
    EPOCH_LENGTH = 30
    STAGES = [
        'Wake|0',
        'Stage 1 sleep|1',
        'Stage 2 sleep|2',
        'Stage 3 sleep|3',
        'Stage 4 sleep|4',
        'REM sleep|5',
        'Unscored|9',
    ]

    rng = np.random.default_rng(random_state)

    with open(filename, 'w') as xml_file:
        xml_file.write(
            '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
            '<PSGAnnotation>\n'

            f'<EpochLength>{EPOCH_LENGTH}</EpochLength>\n'

            '<ScoredEvents>\n'
            '<ScoredEvent>\n'
            '<EventType/>\n'
            '<EventConcept>Recording Start Time</EventConcept>\n'
            '<ClockTime>01.01.85 20.29.59</ClockTime>\n'
            '</ScoredEvent>\n',
        )
        record_duration = hours * 60 * 60
        start = 0
        while True:
            if start > record_duration:
                break
            epoch_duration = rng.choice(np.arange(4, 21)) * EPOCH_LENGTH
            stage = rng.choice(STAGES)
            xml_file.write(
                '<ScoredEvent>\n'
                '<EventType>Stages|Stages</EventType>\n'
                f'<EventConcept>{stage}</EventConcept>\n'
                f'<Start>{start:.1f}</Start>\n'
                f'<Duration>{epoch_duration:.1f}</Duration>\n'
                '</ScoredEvent>\n',
            )
            start += epoch_duration

        xml_file.write(
            '</ScoredEvents>\n'
            '</PSGAnnotation>\n',
        )


def _create_dummy_mesa(data_dir: str, durations: List[float], random_state: int = 42):
    DB_SLUG = 'mesa'
    ANNOTATION_DIRNAME = 'polysomnography/annotations-events-nsrr'
    EDF_DIRNAME = 'polysomnography/edfs'

    db_dir = Path(data_dir).expanduser() / DB_SLUG
    annotations_dir = db_dir / ANNOTATION_DIRNAME
    edf_dir = db_dir / EDF_DIRNAME

    for directory in (annotations_dir, edf_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for i, hours in enumerate(durations):
        record_id = f'mesa-sleep-dummy-{i:04}'
        _dummy_mesa_edf(f'{edf_dir}/{record_id}.edf', hours)
        _dummy_mesa_xml(f'{annotations_dir}/{record_id}-nsrr.xml', hours, random_state)


def test_read_mesa(tmp_path):
    """Basic sanity checks for records read via read_mesa."""
    durations = [0.1, 0.2]  # hours
    valid_stages = {int(s) for s in _SleepStage}

    _create_dummy_mesa(data_dir=tmp_path, durations=durations)
    records = list(read_mesa(data_dir=tmp_path, offline=True))

    assert len(records) == 2

    for rec in records:
        assert rec.sleep_stage_duration == 30
        assert set(rec.sleep_stages) - valid_stages == set()
