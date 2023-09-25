# © SleepECG developers
#
# License: BSD (3-clause)

"""Tests for heartbeat detection and detector evaluation."""

from sys import version_info

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
    return next(read_mitdb(records_pattern="234"))


@pytest.mark.parametrize(
    "backend",
    [
        "c",
        pytest.param(
            "numba",
            marks=pytest.mark.skipif(
                version_info >= (3, 12), reason="Numba does not support Python 3.12 yet"
            ),
        ),
        "python",
    ],
)
def test_detect_heartbeats(mitdb_234_MLII, backend):
    """Test heartbeat detection on mitdb:234:MLII."""
    record = mitdb_234_MLII
    detection = detect_heartbeats(record.ecg, record.fs, backend=backend)
    TP, FP, FN = compare_heartbeats(detection, record.annotation, int(record.fs / 10))

    # Changes in the heartbeat detector should not lead to worse results!
    assert len(TP) >= 2750
    assert len(FP) <= 3
    assert len(FN) == 0
