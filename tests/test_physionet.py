# © SleepECG developers
#
# License: BSD (3-clause)

"""Tests for the PhysioNet I/O helpers."""

from hashlib import sha256
from unittest.mock import Mock

import pytest

from sleepecg import list_physionet_records, set_config
from sleepecg.io.physionet import _list_physionet


@pytest.mark.parametrize(
    ("dataset", "entries", "expected", "pattern", "filtered"),
    [
        (
            "capslpdb",
            "n1.edf\nn16.edf\nn1.txt\nbrux1.edf\n",
            ["n1", "brux1"],
            "n1",
            ["n1"],
        ),
        ("mitdb", "100\n101\n200\n", ["100", "101", "200"], "10?", ["100", "101"]),
        ("ltdb", "14046\n14134\n", ["14046", "14134"], "140*", ["14046"]),
        ("slpdb", "slp01a\nslp01b\n", ["slp01a", "slp01b"], "*a", ["slp01a"]),
    ],
)
def test_list_physionet_records(
    dataset, entries, expected, pattern, filtered, tmp_path, monkeypatch
):
    """Return reader selectors and consult only metadata, even on an empty cache."""
    records = entries.encode()
    root = f"https://physionet.org/files/{dataset}/1.0.0"
    files = {
        f"{root}/SHA256SUMS.txt": f"{sha256(records).hexdigest()}  RECORDS\n".encode(),
        f"{root}/RECORDS": records,
    }
    monkeypatch.setattr(
        "sleepecg.io.utils._get", lambda url: Mock(content=files.pop(url))
    )
    set_config(data_dir=str(tmp_path))
    assert list_physionet_records(dataset) == expected
    set_config(data_dir=str(tmp_path / "other"))
    assert list_physionet_records(dataset, pattern, data_dir=tmp_path) == filtered


def test_unknown_physionet_dataset(tmp_path):
    """A dataset without a SleepECG reader has no reader IDs to list."""
    with pytest.raises(ValueError, match="PhysioNet"):
        list_physionet_records("unknown", data_dir=tmp_path)


def test_list_physionet_expands_user_directory(monkeypatch, tmp_path):
    """Ensure PhysioNet paths expand a leading tilde."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    db_dir = tmp_path / ".sleepecg/datasets/capslpdb"
    db_dir.mkdir(parents=True)
    (db_dir / "SHA256SUMS.txt").write_text("checksum  RECORDS\n")
    (db_dir / "RECORDS").write_text("n1.edf\nn2.edf\n")

    records = _list_physionet(
        "~/.sleepecg/datasets",
        "capslpdb",
        pattern="*.edf",
    )

    assert records == ["n1.edf", "n2.edf"]
