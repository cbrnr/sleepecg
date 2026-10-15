# © SleepECG developers
#
# License: BSD (3-clause)

"""Tests for feature extraction."""

import datetime

import numpy as np
import pytest

from sleepecg.feature_extraction import (
    _FEATURE_GROUPS,
    _hrv_frequencydomain_features,
    _hrv_timedomain_features,
    _metadata_features,
    extract_features,
)
from sleepecg.io.sleep_readers import SleepRecord, SubjectData


def test_feature_ids():
    """Compare length of feature id lists with shape of feature matrices.

    If this fails, make sure the identifiers in `feature_extraction._FEATURE_GROUPS`
    match the calculated features in the relevant function. Note that the test only
    compares lengths, so the order might still be incorrect.
    """
    heartbeat_times = np.cumsum(np.random.uniform(0.5, 1.5, 60 * 60))
    sleep_stages = np.random.randint(1, 6, int(max(heartbeat_times)) // 30)
    sleep_stage_duration = 30
    rri = np.diff(heartbeat_times)
    rri_times = heartbeat_times[1:]
    stage_times = np.arange(len(sleep_stages)) * sleep_stage_duration

    X_time = _hrv_timedomain_features(
        rri,
        rri_times,
        stage_times,
        lookback=0,
        lookforward=30,
    )
    assert X_time.shape[1] == len(_FEATURE_GROUPS["hrv-time"])

    X_frequency = _hrv_frequencydomain_features(
        rri,
        rri_times,
        stage_times,
        lookback=0,
        lookforward=30,
        fs_rri_resample=4,
        max_nans=0,
    )
    assert X_frequency.shape[1] == len(_FEATURE_GROUPS["hrv-frequency"])


@pytest.mark.parametrize(
    ["metadata", "feature_vec"],
    [
        (
            {"start_time": None, "age": None, "gender": None, "weight": None},
            [np.nan] * 4,
        ),
        (
            {
                "start_time": datetime.time(23, 15, 20),
                "age": 55,
                "gender": 1,
                "weight": 99,
            },
            [83720, 55, 1, 99],
        ),
    ],
)
def test_metadata_features(metadata, feature_vec):
    """Test metadata feature extraction."""
    num_stages = 10
    rec = SleepRecord(
        subject_data=SubjectData(
            gender=metadata["gender"],
            age=metadata["age"],
            weight=metadata["weight"],
        ),
        recording_start_time=metadata["start_time"],
    )
    X = _metadata_features(rec, num_stages)
    assert X.shape == (num_stages, 4)
    assert np.allclose(X, np.array(feature_vec), equal_nan=True)


def test_actigraphy_features():
    """Extract externally calculated activity counts."""
    activity_counts = np.array([10.0, 20.0, 30.0, 40.0])
    record = SleepRecord(
        id="test",
        sleep_stages=np.array([5, 2, 2, 4]),
        sleep_stage_duration=30,
        activity_counts=activity_counts,
    )

    features, stages, feature_ids = extract_features(
        [record], feature_selection=["actigraphy"]
    )

    assert feature_ids == ["activity_counts"]
    assert features[0].shape == (4, 1)
    assert np.array_equal(features[0][:, 0], activity_counts)
    assert np.array_equal(stages[0], record.sleep_stages)


@pytest.mark.parametrize(
    ("activity_counts", "message"),
    [
        (None, "without activity_counts"),
        (np.ones((4, 1)), "must be a one-dimensional array"),
        (np.ones(3), "contains 3 values, but 4 are required"),
    ],
)
def test_actigraphy_features_invalid(activity_counts, message):
    """Reject missing or misaligned activity counts."""
    record = SleepRecord(
        id="test",
        sleep_stages=np.array([5, 2, 2, 4]),
        sleep_stage_duration=30,
        activity_counts=activity_counts,
    )

    with pytest.raises(ValueError, match=message):
        extract_features([record], feature_selection=["actigraphy"])


def _extract_hrv_time(heartbeat_times, **kwargs):
    """Extract time domain HRV features in 30 s windows as a dict of columns."""
    record = SleepRecord(sleep_stage_duration=30, heartbeat_times=heartbeat_times)
    features, _, feature_ids = extract_features(
        [record], lookback=0, lookforward=30, feature_selection=["hrv-time"], **kwargs
    )
    return dict(zip(feature_ids, features[0].T))


def test_pnn_windows_with_different_lengths():
    """Normalize pNN50/pNN20 by the number of differences in each window (#347)."""
    rri = np.concatenate([np.tile([1.0, 1.1], 15), np.tile([0.5, 0.6], 30)])
    heartbeat_times = np.concatenate([[0], np.cumsum(rri)])
    X = _extract_hrv_time(heartbeat_times)
    assert np.array_equal(X["NN50"][:2], [27, 52])
    assert np.array_equal(X["NN20"][:2], [27, 52])
    assert np.array_equal(X["pNN50"][:2], [1, 1])
    assert np.array_equal(X["pNN20"][:2], [1, 1])


def test_pnn_invalid_rri():
    """Ignore RR intervals removed by preprocessing when calculating pNN50/pNN20."""
    rri = np.tile([1.0, 1.1], 60)
    rri[5] = 0.2
    heartbeat_times = np.concatenate([[0], np.cumsum(rri)])
    X = _extract_hrv_time(heartbeat_times, min_rri=0.3)
    assert X["NN50"][0] == 26
    assert X["pNN50"][0] == 1
    assert X["pNN20"][0] == 1


def test_pnn_empty_window():
    """Return NaN for NN50/NN20/pNN50/pNN20 in windows without heartbeats."""
    heartbeat_times = np.arange(0, 120, 0.8)
    heartbeat_times = heartbeat_times[(heartbeat_times < 30) | (heartbeat_times >= 60)]
    X = _extract_hrv_time(heartbeat_times)
    for feature_id in ("NN50", "NN20", "pNN50", "pNN20"):
        assert np.isnan(X[feature_id][1])
        assert not np.isnan(X[feature_id][0])


def test_cvsd():
    """Calculate cvSD as RMSSD divided by meanNN."""
    rng = np.random.default_rng(42)
    heartbeat_times = np.cumsum(rng.uniform(0.7, 1.1, 300))
    X = _extract_hrv_time(heartbeat_times)
    assert np.allclose(X["cvSD"], X["RMSSD"] / X["meanNN"], equal_nan=True)
