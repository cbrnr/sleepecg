# © SleepECG developers
#
# License: BSD (3-clause)

"""Tests for the PhysioNet I/O helpers."""

from sleepecg.io.physionet import _list_physionet


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
