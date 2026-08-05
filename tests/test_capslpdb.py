# © SleepECG developers
#
# License: BSD (3-clause)

"""Tests for the CAP Sleep Database reader."""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pytest
from edfio import Edf, EdfSignal

from sleepecg import SleepStage, read_capslpdb
from sleepecg.io.capslpdb import (
    _get_capslpdb_ecg,
    _parse_capslpdb_annotation,
)


def _write_annotation(
    path: Path,
    rows: list[tuple[str, str, str, str]],
    *,
    duration_header: str = "Duration[s]",
) -> None:
    lines = [
        "CAP Sleep Database test fixture",
        f"Sleep Stage\tTime [hh:mm:ss]\tEvent\t{duration_header}\tLocation",
    ]
    lines.extend("\t".join((*row, "C3")) for row in rows)
    path.write_text("\n".join(lines), encoding="ascii")


def _write_edf(
    path: Path,
    labels: tuple[str, ...] = ("ECG",),
    *,
    duration: int = 120,
    sampling_frequency: int = 100,
) -> None:
    signals = [
        EdfSignal(
            np.full(duration * sampling_frequency, index + 1.0),
            sampling_frequency,
            label=label,
        )
        for index, label in enumerate(labels)
    ]
    Edf(signals, starttime=datetime.time(22, 0)).write(path)


def test_parse_capslpdb_annotation_maps_event_fills_gaps_and_clips(tmp_path):
    """Map event labels, fill internal gaps and keep complete EDF epochs."""
    annotation = tmp_path / "record.txt"
    _write_annotation(
        annotation,
        [
            ("W", "21:59:30", "SLEEP-S0", "30"),
            ("1S3", "22:00:00", "SLEEP-S1", "30"),
            ("S2", "22:01:00", "SLEEP-S2", "30"),
            ("S3", "22:01:30", "SLEEP-S3", "30"),
            ("S4", "22:02:00", "SLEEP-S4", "30"),
            ("R", "22:02:30", "SLEEP-REM", "30"),
        ],
    )

    parsed = _parse_capslpdb_annotation(
        annotation,
        recording_start_time=datetime.time(22, 0),
        recording_duration=150,
    )

    assert parsed.sleep_stages.tolist() == [
        SleepStage.N1,
        SleepStage.N1,
        SleepStage.N2,
        SleepStage.N3,
        SleepStage.N3,
    ]
    assert parsed.scoring_start_offset == 0
    assert parsed.malformed_rows == 0


def test_parse_capslpdb_annotation_accepts_header_and_time_variants(tmp_path):
    """Accept the duration header and dotted times used by individual records."""
    annotation = tmp_path / "record.txt"
    _write_annotation(
        annotation,
        [
            ("W", "22.00.30", "SLEEP-S0", "30"),
            ("R", "22.01.00", "SLEEP-REM", "30"),
            ("MT", "22.01.30", "SLEEP-MT", "30"),
        ],
        duration_header="Duration [s]",
    )

    parsed = _parse_capslpdb_annotation(
        annotation,
        recording_start_time=datetime.time(22, 0),
        recording_duration=120,
    )

    assert parsed.sleep_stages.tolist() == [
        SleepStage.WAKE,
        SleepStage.REM,
        SleepStage.UNDEFINED,
    ]
    assert parsed.scoring_start_offset == 30


def test_parse_capslpdb_annotation_aligns_across_midnight(tmp_path):
    """Drop a pre-recording row and keep the 30-second grid across midnight."""
    annotation = tmp_path / "record.txt"
    _write_annotation(
        annotation,
        [
            ("W", "23:59:00", "SLEEP-S0", "30"),
            ("S1", "23:59:30", "SLEEP-S1", "30"),
            ("S2", "00:00:00", "SLEEP-S2", "30"),
        ],
    )

    parsed = _parse_capslpdb_annotation(
        annotation,
        recording_start_time=datetime.time(23, 59, 30),
        recording_duration=60,
    )

    assert parsed.sleep_stages.tolist() == [SleepStage.N1, SleepStage.N2]
    assert parsed.scoring_start_offset == 0


def test_parse_capslpdb_annotation_warns_about_malformed_stage_row(tmp_path):
    """Skip a truncated stage row with one record-level warning."""
    annotation = tmp_path / "record.txt"
    _write_annotation(
        annotation,
        [
            ("W", "22:00:00", "SLEEP-S0", "30"),
            ("S1", "22:00:30", "", ""),
        ],
    )

    with pytest.warns(RuntimeWarning, match="Skipped 1 malformed"):
        parsed = _parse_capslpdb_annotation(
            annotation,
            recording_start_time=datetime.time(22, 0),
            recording_duration=60,
        )

    assert parsed.sleep_stages.tolist() == [SleepStage.WAKE]
    assert parsed.malformed_rows == 1


def test_parse_capslpdb_annotation_rejects_irregular_grid(tmp_path):
    """Reject stage rows that cannot be placed on one 30-second grid."""
    annotation = tmp_path / "record.txt"
    _write_annotation(
        annotation,
        [
            ("W", "22:00:00", "SLEEP-S0", "30"),
            ("S1", "22:00:45", "SLEEP-S1", "30"),
        ],
    )

    with pytest.raises(RuntimeError, match="grid changes by 45 seconds"):
        _parse_capslpdb_annotation(
            annotation,
            recording_start_time=datetime.time(22, 0),
            recording_duration=90,
        )


def test_get_capslpdb_ecg_uses_bipolar_signal():
    """Derive a bipolar ECG when the two source channels are separate."""
    edf = Edf(
        [
            EdfSignal(np.full(100, 3.0), 100, label="ECG1"),
            EdfSignal(np.full(100, 1.0), 100, label="ECG2"),
        ]
    )

    ecg, sampling_frequency = _get_capslpdb_ecg(edf)

    np.testing.assert_array_equal(ecg, np.full(100, 2.0))
    assert sampling_frequency == 100


def test_read_capslpdb_rebases_heartbeats(monkeypatch, tmp_path):
    """Return heartbeat times relative to the first retained scored epoch."""
    db_dir = tmp_path / "capslpdb"
    db_dir.mkdir()
    _write_edf(db_dir / "record.edf")
    _write_annotation(
        db_dir / "record.txt",
        [
            ("W", "22:00:30", "SLEEP-S0", "30"),
            ("S1", "22:01:00", "SLEEP-S1", "30"),
            ("S2", "22:01:30", "SLEEP-S2", "30"),
        ],
    )
    monkeypatch.setattr(
        "sleepecg.io.capslpdb._list_physionet",
        lambda **kwargs: ["record.edf"],
    )
    monkeypatch.setattr(
        "sleepecg.io.capslpdb.detect_heartbeats",
        lambda ecg, fs: np.array([0, 3000, 6000, 9000]),
    )

    record = next(
        read_capslpdb(
            records_pattern="record",
            offline=True,
            keep_edfs=True,
            data_dir=tmp_path,
        )
    )

    assert record.id == "record"
    assert record.recording_start_time == datetime.time(22, 0, 30)
    assert record.sleep_stage_duration == 30
    assert record.sleep_stages.tolist() == [
        SleepStage.WAKE,
        SleepStage.N1,
        SleepStage.N2,
    ]
    np.testing.assert_array_equal(record.heartbeat_times, [0, 30, 60])


def test_read_capslpdb_skips_n16(monkeypatch, tmp_path):
    """Skip n16 with a warning because it has no ECG channel."""

    def _list(**kwargs):
        assert kwargs["pattern"] == "n16.edf"
        return ["n16.edf"]

    monkeypatch.setattr(
        "sleepecg.io.capslpdb._list_physionet",
        _list,
    )

    with pytest.warns(RuntimeWarning, match="Skipping n16"):
        records = list(
            read_capslpdb(records_pattern="n16", offline=True, data_dir=tmp_path)
        )

    assert records == []


def test_read_capslpdb_removes_downloaded_edf(monkeypatch, tmp_path):
    """Remove a newly downloaded EDF when `keep_edfs` is false."""
    db_dir = tmp_path / "capslpdb"

    def _download(**kwargs):
        db_dir.mkdir(exist_ok=True)
        _write_edf(db_dir / "record.edf")
        _write_annotation(
            db_dir / "record.txt",
            [("W", "22:00:00", "SLEEP-S0", "30")],
        )

    monkeypatch.setattr(
        "sleepecg.io.capslpdb._list_physionet",
        lambda **kwargs: ["record.edf"],
    )
    monkeypatch.setattr("sleepecg.io.capslpdb.download_physionet", _download)
    monkeypatch.setattr(
        "sleepecg.io.capslpdb.detect_heartbeats",
        lambda ecg, fs: np.array([100]),
    )

    next(read_capslpdb(data_dir=tmp_path))

    assert not (db_dir / "record.edf").exists()
    assert (db_dir / "record.txt").exists()
