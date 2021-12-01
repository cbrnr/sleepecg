# Authors: Florian Hofer
#
# License: BSD (3-clause)

"""Tests for feature extraction."""

import numpy as np

from sleepecg.feature_extraction import (
    _FEATURE_GROUPS,
    _hrv_frequencydomain_features,
    _hrv_timedomain_features,
)


def test_feature_ids():
    """
    Compare length of feature id lists with shape of feature matrices.

    If this fails, make sure the identifiers in
    `feature_extraction._FEATURE_GROUPS` match the calculated features in
    the relevant function. Note that the test only compares lengths, so the
    order might still be incorrect.
    """
    heartbeat_times = np.cumsum(np.random.uniform(0.5, 1.5, 60*60*8))
    sleep_stages = np.random.randint(1, 6, int(max(heartbeat_times))//30)
    fs_sleep_stages = 1/30
    rri = np.diff(heartbeat_times)
    rri_times = heartbeat_times[1:]
    stage_times = np.arange(len(sleep_stages)) / fs_sleep_stages

    X_time = _hrv_timedomain_features(
        rri,
        rri_times,
        stage_times,
        lookback=0,
        lookforward=30,
    )
    assert X_time.shape[1] == len(_FEATURE_GROUPS['hrv-time'])

    # The analysis window (i.e. the sum of lookback and lookforward) must
    # be at least 3030.3 seconds long to give useful PSD estimates in all
    # frequency ranges, otherwise a warning is issued.
    X_frequency = _hrv_frequencydomain_features(
        rri,
        rri_times,
        stage_times,
        lookback=3001,
        lookforward=30,
        fs_rri_resample=4,
        max_nans=0,
        feature_ids=_FEATURE_GROUPS['hrv-frequency'],
    )
    assert X_frequency.shape[1] == len(_FEATURE_GROUPS['hrv-frequency'])
