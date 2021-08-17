# Authors: Florian Hofer
#
# License: BSD (3-clause)

"""Tests for heartbeat detection C extension."""

import sys

import numpy as np
import pytest

pytestmark = pytest.mark.c_extension


def test_squared_moving_integration_args():
    """Test squared moving window integration argument parsing."""
    from sleepecg._heartbeat_detection import _squared_moving_integration
    x = np.array([0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0])
    window_length = 10

    _squared_moving_integration(x, window_length)
    _squared_moving_integration(x=x, window_length=window_length)
    _squared_moving_integration(x, window_length=window_length)
    _squared_moving_integration(window_length=window_length, x=x)


@pytest.mark.parametrize('window_length', [1, 4, 5, 20])
def test_squared_moving_integration(window_length):
    """Test squared moving window integration calculation."""
    from sleepecg._heartbeat_detection import _squared_moving_integration
    x = np.array([0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0])

    ref = np.convolve(x**2, np.ones(window_length), mode='same')
    res = _squared_moving_integration(x, window_length)
    assert np.allclose(ref, res)


@pytest.mark.parametrize(
    ['x', 'window_length'],
    [
        ('thisisastring', 10),
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
    ['x', 'window_length'],
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
    ['filtered_ecg', 'integrated_ecg', 'fs'],
    [
        ('thisisastring', np.arange(5), 10),
        (np.arange(5), 'thisisastring', 10),
        (np.arange(5), np.arange(5), 'thisisastring'),
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
    ['filtered_ecg', 'integrated_ecg', 'fs'],
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
