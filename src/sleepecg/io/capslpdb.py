# © SleepECG developers
#
# License: BSD (3-clause)

"""Read records from the CAP Sleep Database."""

from __future__ import annotations

import csv
import datetime
import warnings
from collections.abc import Iterator
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from edfio import Edf

from sleepecg.config import get_config_value
from sleepecg.heartbeats import detect_heartbeats
from sleepecg.io.physionet import _list_physionet, download_physionet
from sleepecg.io.sleep_readers import SleepRecord, SleepStage


class _ParseCapslpdbResult(NamedTuple):
    sleep_stages: np.ndarray
    scoring_start_offset: int
    malformed_rows: int


_CAPSLPDB_STAGE_MAPPING = {
    "SLEEP-S0": SleepStage.WAKE,
    "SLEEP-S1": SleepStage.N1,
    "SLEEP-S2": SleepStage.N2,
    "SLEEP-S3": SleepStage.N3,
    "SLEEP-S4": SleepStage.N3,
    "SLEEP-REM": SleepStage.REM,
    "SLEEP-MT": SleepStage.UNDEFINED,
}


def _clock_seconds(value: str) -> int:
    parsed = datetime.datetime.strptime(value.replace(".", ":"), "%H:%M:%S")
    return parsed.hour * 3600 + parsed.minute * 60 + parsed.second


def _time_seconds(value: datetime.time) -> int:
    return value.hour * 3600 + value.minute * 60 + value.second


def _normalise_capslpdb_header(value: str) -> str:
    return value.strip().replace("Duration [s]", "Duration[s]")


def _read_capslpdb_stage_rows(
    annotation_filepath: Path,
) -> tuple[list[tuple[int, SleepStage]], int]:
    lines = annotation_filepath.read_text(encoding="ascii").splitlines()
    try:
        header_index = next(
            index
            for index, line in enumerate(lines)
            if line.split("\t", maxsplit=1)[0].strip() == "Sleep Stage"
        )
    except StopIteration as error:
        raise RuntimeError(
            f"Sleep-stage table header not found in {annotation_filepath}."
        ) from error

    reader = csv.DictReader(lines[header_index:], delimiter="\t")
    if reader.fieldnames is None:
        raise RuntimeError(
            f"Sleep-stage table header not found in {annotation_filepath}."
        )

    rows: list[tuple[int, SleepStage]] = []
    malformed_rows = 0
    for raw_row in reader:
        raw_row.pop(None, None)
        row = {
            _normalise_capslpdb_header(key): (value or "").strip()
            for key, value in raw_row.items()
        }
        event = row.get("Event", "")
        if not event.startswith("SLEEP-"):
            if row.get("Sleep Stage", "") and not event:
                malformed_rows += 1
            continue

        try:
            clock = _clock_seconds(row.get("Time [hh:mm:ss]", ""))
            duration = float(row.get("Duration[s]", ""))
            stage = _CAPSLPDB_STAGE_MAPPING[event]
        except (KeyError, ValueError):
            malformed_rows += 1
            continue

        if duration != 30:
            malformed_rows += 1
            continue
        rows.append((clock, stage))

    if not rows:
        raise RuntimeError(f"No valid sleep-stage rows found in {annotation_filepath}.")
    return rows, malformed_rows


def _parse_capslpdb_annotation(
    annotation_filepath: Path,
    recording_start_time: datetime.time,
    recording_duration: float,
) -> _ParseCapslpdbResult:
    """Parse and align one CAPSLPDB annotation table to its EDF recording."""
    rows, malformed_rows = _read_capslpdb_stage_rows(annotation_filepath)
    recording_start = _time_seconds(recording_start_time)

    first_offset = (rows[0][0] - recording_start + 12 * 3600) % (24 * 3600)
    first_offset -= 12 * 3600

    onsets = [first_offset]
    for (previous_clock, _), (clock, _) in pairwise(rows):
        onsets.append(onsets[-1] + (clock - previous_clock) % (24 * 3600))

    aligned = [(onset, stage) for onset, (_, stage) in zip(onsets, rows) if onset >= 0]
    if not aligned:
        raise RuntimeError(
            f"All sleep-stage rows precede the recording in {annotation_filepath}."
        )

    scoring_start_offset = aligned[0][0]
    stages = [aligned[0][1]]
    previous_onset, previous_stage = aligned[0]
    for onset, stage in aligned[1:]:
        difference = onset - previous_onset
        if difference < 30 or difference % 30:
            raise RuntimeError(
                f"Sleep-stage grid changes by {difference} seconds in "
                f"{annotation_filepath}."
            )
        stages.extend([previous_stage] * (difference // 30 - 1))
        stages.append(stage)
        previous_onset = onset
        previous_stage = stage

    complete_epochs = max(
        int((recording_duration - scoring_start_offset) // 30),
        0,
    )
    stages = stages[:complete_epochs]
    if not stages:
        raise RuntimeError(
            f"No complete scored epochs overlap the recording in {annotation_filepath}."
        )

    if malformed_rows:
        warnings.warn(
            f"Skipped {malformed_rows} malformed sleep-stage row(s) in "
            f"{annotation_filepath}.",
            RuntimeWarning,
            stacklevel=2,
        )

    return _ParseCapslpdbResult(
        sleep_stages=np.asarray(stages, dtype=np.int8),
        scoring_start_offset=scoring_start_offset,
        malformed_rows=malformed_rows,
    )


def _get_capslpdb_ecg(edf: Edf) -> tuple[np.ndarray, float] | None:
    signals = {signal.label.casefold(): signal for signal in edf.signals}
    for label in ("ecg1-ecg2", "ecg", "ekg"):
        if label in signals:
            signal = signals[label]
            return np.asarray(signal.data), float(signal.sampling_frequency)

    if "ecg1" not in signals or "ecg2" not in signals:
        return None
    ecg1 = signals["ecg1"]
    ecg2 = signals["ecg2"]
    if ecg1.sampling_frequency != ecg2.sampling_frequency:
        raise RuntimeError("ECG1 and ECG2 have different sampling frequencies.")
    return (
        np.asarray(ecg1.data) - np.asarray(ecg2.data),
        float(ecg1.sampling_frequency),
    )


def _shift_time(value: datetime.time, seconds: int) -> datetime.time:
    start = datetime.datetime.combine(datetime.date.today(), value)
    return (start + datetime.timedelta(seconds=seconds)).time()


def read_capslpdb(
    records_pattern: str = "*",
    heartbeats_source: str = "ecg",
    offline: bool = False,
    keep_edfs: bool = False,
    data_dir: str | Path | None = None,
) -> Iterator[SleepRecord]:
    """
    Lazily read records from [CAPSLPDB](https://physionet.org/content/capslpdb/).

    Each record consists of an EDF file containing polysomnography signals and a text
    file containing sleep-stage annotations. Sleep stages are aligned to complete
    30-second epochs covered by the EDF recording. Missing internal annotation rows are
    filled with the preceding stage, following the reference CAPSLPDB reader.

    Parameters
    ----------
    records_pattern : str, optional
        Glob-like pattern to select record IDs, by default `'*'`.
    heartbeats_source : {'cached', 'ecg'}, optional
        If `'ecg'` (default), detect heartbeat times from the ECG and cache them. If
        `'cached'`, read previously cached heartbeat times.
    offline : bool, optional
        If `True`, search for local files only instead of downloading from PhysioNet, by
        default `False`.
    keep_edfs : bool, optional
        If `False`, remove EDF files downloaded for heartbeat detection after processing,
        by default `False`.
    data_dir : str | pathlib.Path, optional
        Directory where all datasets are stored. If `None` (default), the value will be
        taken from the configuration.

    Yields
    ------
    SleepRecord
        Each element in the generator is of type `SleepRecord`.
    """
    from edfio import read_edf

    db_slug = "capslpdb"
    if heartbeats_source not in {"cached", "ecg"}:
        raise ValueError(
            f"Invalid value for parameter `heartbeats_source`: {heartbeats_source}, "
            "possible options: {'cached', 'ecg'}"
        )
    if data_dir is None:
        data_dir = get_config_value("data_dir")

    data_dir = Path(data_dir).expanduser()
    db_dir = data_dir / db_slug
    heartbeats_dir = db_dir / "preprocessed/heartbeats"
    heartbeats_dir.mkdir(parents=True, exist_ok=True)

    requested_records = [
        Path(filename).stem
        for filename in _list_physionet(
            data_dir=data_dir,
            db_slug=db_slug,
            pattern=f"{records_pattern}.edf",
        )
    ]
    if "n16" in requested_records:
        warnings.warn(
            "Skipping n16 because its EDF does not contain an ECG channel.",
            RuntimeWarning,
            stacklevel=2,
        )
        requested_records.remove("n16")
    edf_was_available = {
        record_id: (db_dir / f"{record_id}.edf").is_file()
        for record_id in requested_records
    }
    if not offline:
        download_physionet(
            db_slug=db_slug,
            requested_records=requested_records,
            extensions=[".edf", ".txt"],
            data_dir=data_dir,
        )

    for record_id in requested_records:
        edf_filepath = db_dir / f"{record_id}.edf"
        annotation_filepath = db_dir / f"{record_id}.txt"
        heartbeats_filepath = heartbeats_dir / f"{record_id}.npy"

        if heartbeats_source == "cached" and not heartbeats_filepath.is_file():
            print(f"Skipping {record_id} due to missing cached heartbeats.")
            continue

        edf = read_edf(edf_filepath, lazy_load_data=True)
        parsed = _parse_capslpdb_annotation(
            annotation_filepath,
            recording_start_time=edf.starttime,
            recording_duration=edf.duration,
        )
        scored_duration = len(parsed.sleep_stages) * 30

        if heartbeats_source == "cached":
            heartbeat_times = np.load(heartbeats_filepath)
        else:
            ecg_data = _get_capslpdb_ecg(edf)
            if ecg_data is None:
                warnings.warn(
                    f"Skipping {record_id} because its EDF does not contain a supported "
                    "ECG channel.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                if not edf_was_available[record_id] and not keep_edfs:
                    edf_filepath.unlink()
                continue
            ecg, sampling_frequency = ecg_data
            heartbeat_times = (
                detect_heartbeats(ecg, sampling_frequency) / sampling_frequency
                - parsed.scoring_start_offset
            )
            heartbeat_times = heartbeat_times[
                (heartbeat_times >= 0) & (heartbeat_times < scored_duration)
            ]
            np.save(heartbeats_filepath, heartbeat_times)

        heartbeat_times = heartbeat_times[
            (heartbeat_times >= 0) & (heartbeat_times < scored_duration)
        ]

        if not edf_was_available[record_id] and not keep_edfs:
            edf_filepath.unlink()

        yield SleepRecord(
            sleep_stages=parsed.sleep_stages,
            sleep_stage_duration=30,
            id=record_id,
            recording_start_time=_shift_time(
                edf.starttime,
                parsed.scoring_start_offset,
            ),
            heartbeat_times=heartbeat_times,
        )
