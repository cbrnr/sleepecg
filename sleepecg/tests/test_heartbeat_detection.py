# Authors: Florian Hofer
#
# License: BSD (3-clause)

"""Tests for heartbeat detection and detector evaluation."""

import sys

import numpy as np
import pytest

import sleepecg
import sleepecg._heartbeat_detection
from sleepecg.io import read_mitbih


def test_squared_moving_integration_args():
    """Test squared moving window integration argument parsing."""
    x = np.array([0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0])
    window_length = 10

    sleepecg._heartbeat_detection._squared_moving_integration(x, window_length)
    sleepecg._heartbeat_detection._squared_moving_integration(
        x=x, window_length=window_length,
    )
    sleepecg._heartbeat_detection._squared_moving_integration(
        x, window_length=window_length,
    )
    sleepecg._heartbeat_detection._squared_moving_integration(
        window_length=window_length, x=x,
    )


@pytest.mark.parametrize('window_length', [1, 4, 5, 20])
def test_squared_moving_integration(window_length):
    """Test squared moving window integration calculation."""
    x = np.array([0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0])

    ref = np.convolve(x**2, np.ones(window_length), mode='same')
    res = sleepecg._heartbeat_detection._squared_moving_integration(x, window_length)
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
    with pytest.raises(TypeError):
        sleepecg._heartbeat_detection._squared_moving_integration(x, window_length)


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
    with pytest.raises(ValueError):
        sleepecg._heartbeat_detection._squared_moving_integration(x, window_length)


def test_thresholding_args():
    """Test thresholding argument parsing."""
    filtered_ecg = np.arange(100)
    integrated_ecg = np.arange(100)
    fs = 10

    sleepecg._heartbeat_detection._thresholding(filtered_ecg, integrated_ecg, fs)
    sleepecg._heartbeat_detection._thresholding(filtered_ecg, integrated_ecg, fs=fs)
    sleepecg._heartbeat_detection._thresholding(
        filtered_ecg, integrated_ecg=integrated_ecg, fs=fs,
    )
    sleepecg._heartbeat_detection._thresholding(
        filtered_ecg=filtered_ecg, integrated_ecg=integrated_ecg, fs=fs,
    )


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
    filtered_ecg_refcount = sys.getrefcount(filtered_ecg)
    integrated_ecg_refcount = sys.getrefcount(integrated_ecg)

    with pytest.raises(TypeError):
        sleepecg._heartbeat_detection._thresholding(filtered_ecg, integrated_ecg, fs)

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
    filtered_ecg_refcount = sys.getrefcount(filtered_ecg)
    integrated_ecg_refcount = sys.getrefcount(integrated_ecg)

    with pytest.raises(ValueError):
        sleepecg._heartbeat_detection._thresholding(filtered_ecg, integrated_ecg, fs)

    assert sys.getrefcount(filtered_ecg) == filtered_ecg_refcount
    assert sys.getrefcount(integrated_ecg) == integrated_ecg_refcount


def test_compare_heartbeats():
    """Test heartbeat comparison results."""
    detection = np.array([20, 33, 43, 53, 73])
    annotation = np.array([20, 34, 58, 75, 99])
    max_distance = 3

    TP, FP, FN = sleepecg.heartbeat_detection.compare_heartbeats(
        detection,
        annotation,
        max_distance,
    )

    assert np.all(TP == np.array([20, 33, 73]))
    assert np.all(FP == np.array([43, 53]))
    assert np.all(FN == np.array([58, 99]))


def test_detect_heartbeats(tmpdir):
    """Test heartbeat detection on mitdb:234:MLII."""
    record = next(read_mitbih(tmpdir, 'mitdb', '234'))
    detection = sleepecg.heartbeat_detection.detect_heartbeats(record.ecg, record.fs)
    TP, FP, FN = sleepecg.heartbeat_detection.compare_heartbeats(
        detection,
        record.annotation,
        int(record.fs/10),
    )
    # Changes in the heartbeat detector should not lead to worse results!
    assert len(TP) >= 2750
    assert len(FP) <= 3
    assert len(FN) == 0
