# © SleepECG developers
#
# License: BSD (3-clause)

"""Tests for heartbeat detection."""

import sys

import numpy as np
import pytest
from scipy.misc import electrocardiogram
from scipy.signal import resample_poly

from sleepecg import compare_heartbeats, detect_heartbeats

pytestmark = pytest.mark.c_extension
ecg = electrocardiogram()
fs = 360
y_true = detect_heartbeats(ecg, fs)


def f1_score(y_pred, y_true, max_distance=5):
    """Calculate the F1-score."""
    tp, fp, fn = compare_heartbeats(y_pred, y_true, max_distance=max_distance)
    tp, fp, fn = len(tp), len(fp), len(fn)
    f1 = tp / (tp + 0.5 * (fp + fn))
    return f1


def test_resampling():
    """Test the impact of resampling on heartbeat detection."""
    for fs_new in [72, 90, 120, 180, 360, 720, 1080, 3600]:
        res = fs_new / fs
        if fs_new < fs:
            up = 1
            down = 1 / res
        else:
            down = 1
            up = res
        ecg_res = resample_poly(ecg, up, down)
        beats = detect_heartbeats(ecg_res, fs_new)
        beats = np.round(beats / res).astype(int)
        f1 = f1_score(beats, y_true)
        assert f1 > 0.95, "F1-score after resampling to {fs_new} Hz is below 0.90."


def test_rescaling():
    """Test the impact of rescaling on heartbeat detection."""
    for scale in np.logspace(-4, 4, 9):
        beats = detect_heartbeats(ecg * scale, fs)
        f1 = f1_score(beats, y_true)
        assert f1 == 1, "F1-score after rescaling by {scale} is not 1."


def test_squared_moving_integration_args():
    """Test squared moving window integration argument parsing."""
    from sleepecg._heartbeat_detection import _squared_moving_integration

    x = np.array([0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0])
    window_length = 10

    _squared_moving_integration(x, window_length)
    _squared_moving_integration(x=x, window_length=window_length)
    _squared_moving_integration(x, window_length=window_length)
    _squared_moving_integration(window_length=window_length, x=x)


@pytest.mark.parametrize("window_length", [1, 4, 5, 20])
def test_squared_moving_integration(window_length):
    """Test squared moving window integration calculation."""
    from sleepecg._heartbeat_detection import _squared_moving_integration

    x = np.array([0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0])

    ref = np.convolve(x**2, np.ones(window_length), mode="same")
    res = _squared_moving_integration(x, window_length)
    assert np.allclose(ref, res)


@pytest.mark.parametrize(
    ["x", "window_length"],
    [
        ("thisisastring", 10),
        (np.arange(20), 15.0),
        (np.arange(20), 15.5),
    ],
)
def test_squared_moving_integration_typechecks(x, window_length):
    """Test squared moving window integration typechecks."""
    from sleepecg._heartbeat_detection import _squared_moving_integration

    with pytest.raises(TypeError):
        _squared_moving_integration(x, window_length)


@pytest.mark.parametrize(
    ["x", "window_length"],
    [
        (np.arange(10), -5),
        (np.arange(10), 0),
        (np.arange(10), 21),
        (np.array(5), 3),
        (np.zeros((5, 5)), 3),
    ],
)
def test_squared_moving_integration_valuechecks(x, window_length):
    """Test squared moving window integration valuechecks."""
    from sleepecg._heartbeat_detection import _squared_moving_integration

    with pytest.raises(ValueError):
        _squared_moving_integration(x, window_length)


def test_thresholding_args():
    """Test thresholding argument parsing."""
    from sleepecg._heartbeat_detection import _thresholding

    filtered_ecg = np.arange(100)
    integrated_ecg = np.arange(100)
    fs = 10

    _thresholding(filtered_ecg, integrated_ecg, fs)
    _thresholding(filtered_ecg, integrated_ecg, fs=fs)
    _thresholding(filtered_ecg, integrated_ecg=integrated_ecg, fs=fs)
    _thresholding(filtered_ecg=filtered_ecg, integrated_ecg=integrated_ecg, fs=fs)


@pytest.mark.parametrize(
    ["filtered_ecg", "integrated_ecg", "fs"],
    [
        ("thisisastring", np.arange(5), 10),
        (np.arange(5), "thisisastring", 10),
        (np.arange(5), np.arange(5), "thisisastring"),
    ],
)
def test_thresholding_typechecks(filtered_ecg, integrated_ecg, fs):
    """Test thresholding typechecks."""
    from sleepecg._heartbeat_detection import _thresholding

    filtered_ecg_refcount = sys.getrefcount(filtered_ecg)
    integrated_ecg_refcount = sys.getrefcount(integrated_ecg)

    with pytest.raises(TypeError):
        _thresholding(filtered_ecg, integrated_ecg, fs)

    assert sys.getrefcount(filtered_ecg) == filtered_ecg_refcount
    assert sys.getrefcount(integrated_ecg) == integrated_ecg_refcount


@pytest.mark.parametrize(
    ["filtered_ecg", "integrated_ecg", "fs"],
    [
        (np.zeros((5, 5)), np.arange(5), 10),
        (np.array(5), np.arange(5), 10),
        (np.arange(5), np.zeros((5, 5)), 10),
        (np.arange(5), np.array(5), 10),
        (np.arange(5), np.arange(5), -5),
        (np.arange(5), np.arange(5), 0),
    ],
)
def test_thresholding_valuechecks(filtered_ecg, integrated_ecg, fs):
    """Test thresholding valuechecks."""
    from sleepecg._heartbeat_detection import _thresholding

    filtered_ecg_refcount = sys.getrefcount(filtered_ecg)
    integrated_ecg_refcount = sys.getrefcount(integrated_ecg)

    with pytest.raises(ValueError):
        _thresholding(filtered_ecg, integrated_ecg, fs)

    assert sys.getrefcount(filtered_ecg) == filtered_ecg_refcount
    assert sys.getrefcount(integrated_ecg) == integrated_ecg_refcount
