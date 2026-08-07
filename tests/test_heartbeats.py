# © SleepECG developers
#
# License: BSD (3-clause)

"""Tests for heartbeat detection and detector evaluation."""

import os

import numpy as np
import pytest

from sleepecg import compare_heartbeats, detect_heartbeats, read_mitdb


def test_compare_heartbeats():
    """Test heartbeat comparison results."""
    detection = np.array([20, 33, 43, 53, 73])
    annotation = np.array([20, 34, 58, 75, 99])
    max_distance = 3

    TP, FP, FN = compare_heartbeats(detection, annotation, max_distance)

    assert np.all(TP == np.array([20, 33, 73]))
    assert np.all(FP == np.array([43, 53]))
    assert np.all(FN == np.array([58, 99]))


@pytest.fixture(scope="session")
def mitdb_234_MLII():
    """Fetch record for detector tests."""
    pytest.importorskip("wfdb")
    # CI caches downloaded PhysioNet files across runs in this directory, see
    # .github/workflows/cibuildwheel.yml; falls back to the configured default otherwise
    data_dir = os.environ.get("SLEEPECG_TEST_DATA_DIR")
    return next(read_mitdb(records_pattern="234", data_dir=data_dir))


@pytest.mark.parametrize("backend", ["c", "numba", "python"])
def test_detect_heartbeats(mitdb_234_MLII, backend):
    """Test heartbeat detection on mitdb:234:MLII."""
    if backend == "numba":
        pytest.importorskip("numba")
    record = mitdb_234_MLII
    detection = detect_heartbeats(record.ecg, record.fs, backend=backend)
    TP, FP, FN = compare_heartbeats(detection, record.annotation, int(record.fs / 10))

    # Changes in the heartbeat detector should not lead to worse results!
    assert len(TP) >= 2750
    assert len(FP) <= 3
    assert len(FN) == 0
