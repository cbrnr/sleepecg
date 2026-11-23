# © SleepECG developers
#
# License: BSD (3-clause)

"""Tests for listing PhysioNet records."""

from hashlib import sha256
from unittest.mock import Mock

import pytest

from sleepecg import list_physionet_records, set_config


@pytest.mark.parametrize(
    ("dataset", "entries", "pattern", "expected"),
    [
        ("capslpdb", "n1.edf\nn16.edf\nn1.txt\nbrux1.edf\n", "n1", ["n1"]),
        ("mitdb", "100\n101\n200\n", "10?", ["100", "101"]),
        ("ltdb", "14046\n14134\n", "140*", ["14046"]),
        ("slpdb", "slp01a\nslp01b\n", "*a", ["slp01a"]),
    ],
)
def test_list_physionet_records(
    dataset, entries, pattern, expected, tmp_path, monkeypatch
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
    assert list_physionet_records(dataset, pattern) == expected
    set_config(data_dir=str(tmp_path / "other"))
    assert (
        list_physionet_records(dataset, expected[0], data_dir=tmp_path) == expected[:1]
    )


def test_unknown_physionet_dataset(tmp_path):
    """A dataset without a SleepECG reader has no reader IDs to list."""
    with pytest.raises(ValueError, match="PhysioNet"):
        list_physionet_records("unknown", data_dir=tmp_path)
