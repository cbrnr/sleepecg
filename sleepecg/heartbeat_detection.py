# Authors: Florian Hofer
#
# License: BSD (3-clause)

from typing import NamedTuple

import numpy as np
import scipy.interpolate
import scipy.signal
import scipy.stats

from ._heartbeat_detection import _squared_moving_integration, _thresholding

__all__ = [
    'compare_heartbeats',
    'detect_heartbeats',
    'rri_similarity',
]


# cache sos-filter created with scipy.signal.butter to reduce runtime
_sos_filters = {}


def detect_heartbeats(ecg: np.ndarray, fs: float) -> np.ndarray:
    """
    Detect heartbeats in an ECG signal.

    This is a modified version of the beat detection algorithm using
    adaptive thresholds described by Pan & Tompkins in 1985.

    Modifications/additions to the original algorithm described in
    https://doi.org/10.1109/TBME.1985.325532 are listed here:

    - Instead of a hardware filter adjusted to the sampling frequency of
      the MIT-BIH Arrhythmia Database, a 2nd-order bandpass with cutoff
      frequencies 5 and 30 Hz created via `scipy.signal.butter` is used.
    - A bidirectional filter is used to remove filter delay.
    - The signal might start during a peak, in which case it might have a
      relatively high initial amplitude, which will mess up threshold
      initialization. Therefore, everything until the first zero-crossing
      is set to 0.
    - The integration window is centered on the filtered signal, i.e. a
      peak in the filtered signal corresponds to a plateau in the
      integrated one, not a saddle in the rising edge. This lets the
      adaptive threshold for the integrated signal remain at a higher
      level, which is less susceptible to noise.
    - Learning phase 1 is not described in detail in the original paper.
      This implementation uses maximum and mean values inside the first 2
      seconds to initialize SPKI/SPKF/NPKI/NPKF. Details are provided in
      the C code.
    - In addition to the original searchback criterion, a searchback is
      also performed if no peak is found during the first second of the
      signal or no second peak is found 1.5s after the first one. This
      ensures correct behaviour at signal start in case an unusually large
      peak during learning phase 1 messes up threshold initialization.
    - After an unsuccessful searchback, the procedure is repeated in the
      same interval with further reduced thresholds, up to 16 times.

    Parameters
    ----------
    ecg : np.ndarray
        ECG signal.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    heartbeat_indices: np.ndarray
        Indices of detected heartbeats.
    """
    # For short signals, creating the bandpass filter makes up a large part
    # of the total runtime. Therefore the filter is cached to a global
    # variable.
    try:
        sos = _sos_filters[fs]
    except KeyError:
        sos = scipy.signal.butter(
            N=2,
            Wn=(5, 30),
            btype='bandpass',
            output='sos',
            fs=fs,
        )
        _sos_filters[fs] = sos

    # filtering bidirectionally removes filter delay
    filtered_ecg = scipy.signal.sosfiltfilt(sos, ecg)

    # Set everything until the first zero-crossing to 0. For efficiency,
    # only the first 2 seconds are checked.
    signal_start = np.where(np.diff(np.signbit(filtered_ecg[:int(2 * fs)])))[0][0] + 1
    filtered_ecg[:signal_start] = 0

    # scipy.signal.sosfilt returns an array with negative strides. Both
    # `np.correlate` and `_thresholding` require contiguity, so ensuring
    # this here once reduces total runtime.
    filtered_ecg = np.ascontiguousarray(filtered_ecg)

    # five-point derivative as described by Pan & Tompkins
    derivative = np.correlate(filtered_ecg, np.array([-1, -2, 0, 2, 1]), mode='same')

    moving_window_width = int(0.15 * fs)  # 150ms
    integrated_ecg = _squared_moving_integration(derivative, moving_window_width)

    beat_mask = _thresholding(filtered_ecg, integrated_ecg, fs)

    return np.where(beat_mask)[0]


class CompareHeartbeatsResult(NamedTuple):
    TP: np.ndarray
    FP: np.ndarray
    FN: np.ndarray


def compare_heartbeats(
    detection: np.ndarray,
    annotation: np.ndarray,
    max_distance: int = 0,
) -> CompareHeartbeatsResult:
    """
    Determine correctness of detection results.

    Determine true positives (TP), false positives (FP) and false negatives
    (FN) for an array of detected heartbeat indices based on an array of
    annotated heartbeat indices. Since neither annotations nor automated
    detectors usually hit the peak perfectly, detected peaks no further
    than `max_distance` in both directions from an annotated peak are
    considered true positives.

    Parameters
    ----------
    detection : np.ndarray
        Detected heartbeat indices.
    annotation : np.ndarray
        Annotated heartbeat indices.
    max_distance : int, optional
        Maximum distance between indices to consider as the same peak, by
        default 0.

    Returns
    -------
    TP : np.ndarray
        True positives, i.e. actual heartbeats detected as heartbeats.
    FP : np.ndarray
        False positives, i.e. non-heartbeats detected as heartbeats.
    FN : np.ndarray
        False negatives, i.e. actual heartbeats not detected as heartbeats.
    """
    if len(detection) == 0:
        return CompareHeartbeatsResult(
            TP=np.array([]),
            FP=np.array([]),
            FN=annotation.copy(),
        )

    max_len = max(np.max(detection), np.max(annotation)) + 1
    detection_mask = np.zeros(max_len, dtype=bool)
    detection_mask[detection] = 1
    annotation_mask = np.zeros(max_len, dtype=bool)
    annotation_mask[annotation] = 1

    fuzzy_filter = np.ones(max_distance * 2 + 1, dtype=bool)
    detection_mask_fuzzy = np.convolve(detection_mask, fuzzy_filter, mode='same')
    annotation_mask_fuzzy = np.convolve(annotation_mask, fuzzy_filter, mode='same')

    return CompareHeartbeatsResult(
        TP=np.where(detection_mask & annotation_mask_fuzzy)[0],
        FP=np.where(detection_mask & ~annotation_mask_fuzzy)[0],
        FN=np.where(annotation_mask & ~detection_mask_fuzzy)[0],
    )


class RRISimilarityResult(NamedTuple):
    pearsonr: float
    spearmanr: float
    rmse: float


def rri_similarity(
    detection: np.ndarray,
    annotation: np.ndarray,
    fs_resample: float = 4,
) -> RRISimilarityResult:
    """
    Calculate measures of similarity between RR intervals.

    RR intervals are calculated from detected and annotated heartbeat
    indices. The RR time series is then resampled to frequency
    `fs_resample` in the timespan common to both detection and annotation.
    Pearson's and Spearman's correlation coefficient as well as the root
    mean square error are returned.

    Parameters
    ----------
    detection : np.ndarray
        Detected heartbeat indices.
    annotation : np.ndarray
        Annotated heartbeat indices.

    Returns
    -------
    pearsonr : float
        Pearson correlation coefficient between resampled RR time series.
    spearmanr : float
        Spearman correlation coefficient between resampled RR time series.
    rmse : float
        Root mean square error between resampled RR time series.
    """
    rr_ann = np.diff(annotation)
    rr_det = np.diff(detection)

    t_ann = annotation[1:]
    t_det = detection[1:]

    start = max(min(t_ann), min(t_det))
    end = min(max(t_ann), max(t_det))
    t_new = np.arange(start, end, 1/fs_resample)

    interp_det = scipy.interpolate.interp1d(t_det, rr_det)
    interp_ann = scipy.interpolate.interp1d(t_ann, rr_ann)
    det_resampled = interp_det(t_new)
    ann_resampled = interp_ann(t_new)

    return RRISimilarityResult(
        scipy.stats.pearsonr(det_resampled, ann_resampled)[0],
        scipy.stats.spearmanr(det_resampled, ann_resampled)[0],
        np.sqrt(np.mean((det_resampled - ann_resampled)**2)),
    )
