# Authors: Florian Hofer
#
# License: BSD (3-clause)

"""Heartbeat detection and detector evaluation."""

import warnings
from typing import NamedTuple

import numpy as np
import scipy.interpolate
import scipy.signal
import scipy.stats

_all_backends = ('c', 'numba', 'python')
_available_backends = list(_all_backends)

try:
    from ._heartbeat_detection import _squared_moving_integration, _thresholding
except ImportError:
    _available_backends.remove('c')

try:
    from numba import jit
except ImportError:
    _available_backends.remove('numba')


__all__ = [
    'compare_heartbeats',
    'detect_heartbeats',
    'rri_similarity',
]


# cache sos-filter created with scipy.signal.butter to reduce runtime
_sos_filters = {}


def detect_heartbeats(ecg: np.ndarray, fs: float, backend: str = 'c') -> np.ndarray:
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
      the `_thresholding` code.
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
    backend : {'c', 'numba', 'python'}
        Which implementation of the squared moving integration and
        thresholding algorithm to use. If available, `'c'` is the fastest
        implementation, `'numba'` is about 25% slower, and `'python'` is
        about 20 times slower but provided as a fallback. By default `'c'`.

    Returns
    -------
    heartbeat_indices: np.ndarray
        Indices of detected heartbeats.
    """
    if backend not in _all_backends:
        raise ValueError(
            f'Invalid backend for heartbeat_detection: {backend!r}. '
            f'Possible options are: {_all_backends}.',
        )
    if backend not in _available_backends:
        fallback = _available_backends[0]
        warnings.warn(f'Backend {backend!r} not available, using {fallback!r} instead.')
        backend = fallback

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

    # scipy.signal.sosfiltfilt returns an array with negative strides. Both
    # `np.correlate` and `_thresholding` require contiguity, so ensuring
    # this here once reduces total runtime.
    filtered_ecg = np.ascontiguousarray(filtered_ecg)

    # five-point derivative as described by Pan & Tompkins
    derivative = np.correlate(filtered_ecg, np.array([-1, -2, 0, 2, 1]), mode='same')

    moving_window_width = int(0.15 * fs)  # 150ms

    if backend == 'c':
        integrated_ecg = _squared_moving_integration(derivative, moving_window_width)
        beat_mask = _thresholding(filtered_ecg, integrated_ecg, fs)
    elif backend == 'numba':
        integrated_ecg = _squared_moving_integration_numba(derivative, moving_window_width)
        beat_mask = _thresholding_numba(filtered_ecg, integrated_ecg, fs)
    elif backend == 'python':
        integrated_ecg = np.convolve(
            derivative**2,
            np.ones(moving_window_width),
            mode='same',
        )
        beat_mask = _thresholding_py(filtered_ecg, integrated_ecg, fs)

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
        default `0`.

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


def _squared_moving_integration_py(x: np.ndarray, window_length: int) -> np.ndarray:
    """
    Perform squaring and moving integration of an array.

    Parameters
    ----------
    x : np.ndarray
        The input array.
    window_length : int
        Length of the moving window in samples.

    Returns
    -------
    output : np.ndarray
        The squared and integrated array.
    """
    signal_len = len(x)
    if not 0 < window_length <= signal_len:
        raise ValueError('window_length has to be 0 < window_length <= len(x)')

    output = np.empty_like(x)

    # create a circular buffer to store values inside integration window
    integration_buffer = np.zeros(window_length)
    sum = 0

    # the integration window is centered on the original signal, for even
    # window_length the behaviour of np.convolve with a constant window of
    # even length is replicated (i.e the window is off-center to the left)
    window_length_half = (window_length + 1) // 2

    # during the first `window_length/2` samples, there's no output, since
    # the integration window's center would be at a negative index of the
    # input
    for i in range(window_length_half):
        square = x[i] * x[i]
        integration_buffer[i % window_length] = square
        sum += square

    for i in range(window_length_half, signal_len):
        output[i - window_length_half] = sum  # write to 'window center'
        sum -= integration_buffer[i % window_length]
        square = x[i] * x[i]
        integration_buffer[i % window_length] = square
        sum += square

    # the end of the x signal is reached, so the integration window is
    # built down and the last `window_length/2` entries of the output are
    # filled
    for i in range(signal_len, signal_len+window_length_half):
        output[i - window_length_half] = sum
        sum -= integration_buffer[i % window_length]

    return output


def _thresholding_py(
    filtered_ecg: np.ndarray,
    integrated_ecg: np.ndarray,
    fs: float,
) -> np.ndarray:
    """
    Perform adaptive thresholding based on Pan and Tompkin's algorithm.

    Parameters
    ----------
    filtered_ecg : np.ndarray
        Bandpass filtered ECG.
    integrated_ecg : np.ndarray
        Squared and moving window integrated bandpass filtered ECG.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    beat_mask : np.ndarray
        Array containing `1` for every sample in `filtered_ecg` identified
        as a heartbeat (i.e. R-peak).
    """
    signal_len = len(filtered_ecg)
    beat_mask = np.zeros_like(filtered_ecg)  # numba can't work with dtype=bool

    REFRACTORY_SAMPLES = int(0.2 * fs)  # 200ms
    T_WAVE_WINDOW = int(0.36 * fs)  # 360ms

    # --------------------------------------------------------------------
    # Learning Phase 1
    # --------------------------------------------------------------------
    # Pan & Tompkins mention a learning phase to initialize detection
    # thresholds based upon signal and noise peaks detected during the
    # first two seconds. The exact initialization process is not
    # described. The adaptive thresholds are calculated based on running
    # estimates of signal and noise peaks (`SPKF` and `NPKF` for the
    # filtered signal). Assuming constant peak amplitudes, those values
    # converge towards the signal peak amplitude and noise peak amplitude,
    # respectively. Therefore, SPKF/SPKI are assumed to be the maximum
    # values of the filtered/integrated signal during the learning phase.
    # Accordingly, NPKF/NPKI are initialized to the mean values during the
    # learning phase.
    learning_phase_samples = int(2 * fs)  # 2 seconds
    filtered_ecg_maximum = 0
    filtered_ecg_sum = 0
    integrated_ecg_maximum = 0
    integrated_ecg_sum = 0
    for i in range(learning_phase_samples):
        if (filtered_ecg[i] > filtered_ecg_maximum):
            filtered_ecg_maximum = filtered_ecg[i]
        if (integrated_ecg[i] > integrated_ecg_maximum):
            integrated_ecg_maximum = integrated_ecg[i]
        filtered_ecg_sum += filtered_ecg[i]
        integrated_ecg_sum += integrated_ecg[i]

    SPKF = filtered_ecg_maximum
    NPKF = filtered_ecg_sum / learning_phase_samples
    SPKI = integrated_ecg_maximum
    NPKI = integrated_ecg_sum / learning_phase_samples
    threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)
    threshold_F1 = NPKF + 0.25 * (SPKF - NPKF)

    # According to the original paper, `RR AVERAGE2` is the average of the
    # last 8 RR-intervals, that lie in a certain interval. In the worst
    # case, this requires going back to the very first RR-interval.
    # Therefore, all RR-intervals are stored. As the algorithm enforces a
    # refractory period, there can't be more than $signal_len /
    # refractory_samples$ heartbeats.
    RR_intervals = np.zeros(signal_len//REFRACTORY_SAMPLES)

    # tracking the number of peaks found is required to calculate the
    # average RR-intervals correctly during the first 8 beats. Also needed
    # for access to RR_intervals
    num_peaks_found = 0

    RR_missed_limit = None

    # In case a searchback was unsuccessful, no new searchback will be
    # performed until another signal peak has been found regularly.
    do_searchback = True

    # initialize so searchback before any peak was detected works
    peak_index = -REFRACTORY_SAMPLES + 1
    previous_peak_index = -REFRACTORY_SAMPLES + 1

    index = 1
    while index < signal_len - 1:

        PEAKF = None
        PEAKI = None

        signal_peak_found = False
        noise_peak_found = False
        # ----------------------------------------------------------------
        # Searchback
        # ----------------------------------------------------------------
        # During a "searchback", detection thresholds are reduced by one
        # half. The peak with highest amplitude between 200ms (i.e. the
        # refractory period) after the previous detected peak and the
        # current index is considered as a peak candidate.
        # Modifications compared to Pan&Tompkins' original method:
        # - The original paper states that a searchback-peak's amplitude
        #   has to be between the original threshold and the reduced one.
        #   The situation can arise, that this is the case for the
        #   filtered signal, but not for the integrated one (if the raw
        #   signal amplitude is suddenly considerably lower). Therefore,
        #   this implementation requires both of the signals
        #   (filtered+integrated) to be above the reduced threshold, but
        #   only one of them to be below the original threshold.
        # - No further steps are specified for the case that no peak is
        #   found during searchback. Since a searchback is triggered
        #   because (physiologically) there _has_ to be a heartbeat during
        #   the searchback interval, this implementation repeats the
        #   process with further reduced thresholds. Up to 16 searchback
        #   runs are performed, each time the thresholds are futher
        #   reduced by 1/2. Giving a hard limit of 16 runs prevents
        #   getting into an endless loop in case there's really just
        #   noise.
        # - Since the criterion for triggering a searchback is based on
        #   the average RR-interval, in the original form this could only
        #   happen after at least two detected heartbeats. An
        #   exceptionally large peak during the first learning phase can
        #   throw the initial thresholds off, so peaks at the beginning
        #   are ignored - which in turn invalidates learning phase 2.
        #   Therefore, in addition to the original searchback criterion
        #   (no peak during 1.66*"the average RR interval"), a searchback
        #   is triggered in two cases: 1) if there's no peak during the
        #   first second and 2) if there's no peak 1.5s after the first
        #   peak.
        if ((num_peaks_found > 1 and index - previous_peak_index > RR_missed_limit and do_searchback) or  # original criterion  # noqa: E501
                (num_peaks_found == 0 and index > fs) or                             # 1)
                (num_peaks_found == 1 and index - previous_peak_index > 1.5 * fs)):  # 2)

            for i in range(1, 16):
                found_a_candidate = False

                searchback_divisor = 1 << i  # 2^i
                best_searchback_index = previous_peak_index + REFRACTORY_SAMPLES
                best_candidate_amplitude = -1
                searchback_index = best_searchback_index

                while searchback_index < index:
                    PEAKF = filtered_ecg[searchback_index]
                    if PEAKF > filtered_ecg[searchback_index + 1]:
                        # next one's lower
                        if PEAKF > filtered_ecg[searchback_index - 1]:
                            # it's a peak
                            PEAKI = integrated_ecg[searchback_index]
                            # One signal is between the reduced and
                            # original threshold, the other one above the
                            # reduced threshold.
                            if ((threshold_F1 / searchback_divisor < PEAKF and PEAKF < threshold_F1 and threshold_I1 / searchback_divisor < PEAKI) or  # noqa: E501
                                    (threshold_I1 / searchback_divisor < PEAKI and PEAKI < threshold_I1 and threshold_F1 / searchback_divisor < PEAKF)):   # noqa: E501
                                if PEAKF > best_candidate_amplitude:
                                    # highest one so far
                                    best_searchback_index = searchback_index
                                    best_candidate_amplitude = filtered_ecg[searchback_index]  # noqa: E501
                                    found_a_candidate = True

                        # the amplitude of the next sample is lower -> it
                        # can't be a peak -> skip it
                        searchback_index += 1

                    searchback_index += 1

                if found_a_candidate:
                    SPKI = 0.25 * PEAKI + 0.75 * SPKI
                    SPKF = 0.25 * PEAKF + 0.75 * SPKF
                    signal_peak_found = True
                    peak_index = best_searchback_index

                    # Don't perform a searchback until the next signal
                    # peak has been found to avoid endless loops.
                    do_searchback = False
                    break

        elif filtered_ecg[index] > filtered_ecg[index + 1]:
            if filtered_ecg[index] > filtered_ecg[index - 1]:
                # a local maximum in the filtered signal was found
                PEAKF = filtered_ecg[index]
                PEAKI = integrated_ecg[index]
                if PEAKF > threshold_F1 and PEAKI > threshold_I1:
                    # Both the filtered and the integrated signal are
                    # above their respective thresholds. Thus the current
                    # peak is classified as a signal peak and the running
                    # estimates SPKF and SPKI are updated
                    SPKF = 0.125 * PEAKF + 0.875 * SPKF
                    SPKI = 0.125 * PEAKI + 0.875 * SPKI

                    signal_peak_found = True
                    peak_index = index
                else:
                    noise_peak_found = True

            # The next sample's amplitude is lower, meaning it can't be a
            # peak, so we skip it. This is why there are two separate
            # if-clauses for this block.
            index += 1

        # Calculating the RR-interval and comparing slopes only makes
        # sense, if there has already been a signal peak in the past,
        if signal_peak_found and num_peaks_found > 0:
            RR = peak_index - previous_peak_index

            # ------------------------------------------------------------
            # T-Wave Identification
            # ------------------------------------------------------------
            # "When an RI interval is less than 360 ms (it must be greater
            # than the 200 ms latency), a judgment is made to determine
            # whether the current QRS complex has been correctly
            # identified or whether it is really a T wave. If the maximal
            # slope that occurs during this waveform is less than half
            # that of the QRS waveform that preceded it, it is identified
            # to be a Twave; otherwise, it is called a QRS complex." (from
            # Pan&Tompkins, 1985)
            if RR < T_WAVE_WINDOW:
                reverse_index = peak_index
                max_slope_in_this_peak = -1
                while reverse_index > 0:
                    amplitude_here = filtered_ecg[reverse_index]
                    amplitude_before = filtered_ecg[reverse_index - 1]
                    if amplitude_before > amplitude_here:
                        break
                    slope = amplitude_here - amplitude_before
                    if slope > max_slope_in_this_peak:
                        max_slope_in_this_peak = slope
                    reverse_index -= 1

                reverse_index = previous_peak_index
                max_slope_in_previous_peak = -1
                while reverse_index > 0:
                    amplitude_here = filtered_ecg[reverse_index]
                    amplitude_before = filtered_ecg[reverse_index - 1]
                    if amplitude_before > amplitude_here:
                        break
                    slope = amplitude_here - amplitude_before
                    if (slope > max_slope_in_previous_peak):
                        max_slope_in_previous_peak = slope
                    reverse_index -= 1

                if max_slope_in_this_peak < max_slope_in_previous_peak / 2.0:
                    # based on the slope, this peak should be a T-Wave
                    signal_peak_found = False
                    noise_peak_found = True

        if signal_peak_found:
            # What we know so far: we're at a local maximum, both
            # thresholds are exceeded and it's not a T-Wave. Thus, the
            # current sample can be considered as a "signal peak" and the
            # adaptive are updated.
            num_peaks_found += 1
            beat_mask[peak_index] = 1

            threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)
            threshold_F1 = NPKF + 0.25 * (SPKF - NPKF)

            # Calculating RR-averages only makes sense once 2 peaks have
            # been found
            if num_peaks_found > 1:
                RR_intervals[num_peaks_found] = peak_index - previous_peak_index

                # --------------------------------------------------------
                # Learning phase 2
                # --------------------------------------------------------
                # "Learning phase 2 requires two heartbeats to initialize
                # RR -interval average and RR-interval limit values."
                # (from Pan&Tompkins, 1985)
                if num_peaks_found == 2:
                    RR_low_limit = 0.92 * RR_intervals[num_peaks_found]
                    RR_high_limit = 1.16 * RR_intervals[num_peaks_found]

                # --------------------------------------------------------
                # RR Average 1 / RR Average 2
                # --------------------------------------------------------
                # RR Average 2 is the average of the 8 most recent RR
                # intervals, which fell between RR_low_limit and
                # RR_high_limit. In case of a regular heart rate, this
                # equals RR Average 1 (the average over the 8 most recent
                # RR intervals, independent of any limits). Therefore, RR
                # Average 1 does not need to be calculated separately.
                RR_sum = 0
                RR_count = 0
                irregular = False
                for i in range(num_peaks_found, 1, -1):
                    RR_n = RR_intervals[i]
                    if RR_low_limit < RR_n and RR_n < RR_high_limit:
                        RR_sum += RR_n
                        RR_count += 1
                        if RR_count >= 8:
                            break
                    else:
                        irregular = True

                RR_average = RR_sum / RR_count
                RR_low_limit = 0.92 * RR_average
                RR_high_limit = 1.16 * RR_average
                RR_missed_limit = 1.66 * RR_average

                if irregular:
                    # "For irregular heart rates, the first threshold of
                    # each set is reduced by half so as to increase the
                    # detection sensitivity and to avoid missing beats."
                    threshold_F1 /= 2
                    threshold_I1 /= 2

            # A signal peak has been found, so performing a searchback
            # makes sense.
            do_searchback = True

            # previous peak index is required to calculate the RR-interval
            previous_peak_index = peak_index

            # no peak can happen during the refractory period, so skip it
            index = peak_index + REFRACTORY_SAMPLES

        elif noise_peak_found:
            NPKI = 0.125 * PEAKI + 0.875 * NPKI
            NPKF = 0.125 * PEAKF + 0.875 * NPKF
            threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)
            threshold_F1 = NPKF + 0.25 * (SPKF - NPKF)

        index += 1

    # Return an array containing `1` at each beat-position, `0` elsewhere.
    return beat_mask


if 'numba' in _available_backends:
    _squared_moving_integration_numba = jit(_squared_moving_integration_py)
    _thresholding_numba = jit(_thresholding_py)
