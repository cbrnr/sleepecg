# Authors: Florian Hofer
#
# License: BSD (3-clause)

"""Functions and utilities related to feature extraction."""

import warnings
from typing import List, Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import interp1d

__all__ = [
    'extract_hrv_features',
]


def _create_ragged_array(data: List[np.ndarray]) -> np.ndarray:
    """
    Convert an list of arrays with different lengths to a numpy array.

    Each element in `data` is a row in the resulting array. Rows shorter
    than the longest row will be padded with `np.nan`.

    Parameters
    ----------
    data : list[np.ndarray]
        A list of arrays which may have different lengths.

    Returns
    -------
    np.ndarray
        The padded rectangular array.
    """
    max_len = max(len(x) for x in data)
    ragged_array = np.full((len(data), max_len), fill_value=np.nan)
    for row_index, row in enumerate(data):
        ragged_array[row_index, :len(row)] = row
    return ragged_array


def _split_into_windows(
    data: np.ndarray,
    data_times: np.ndarray,
    window_times: np.ndarray,
    lookback: int,
    lookforward: int,
) -> List[np.ndarray]:
    """
    Split (irregularly sampled) data into windows of equal temporal length.

    Make sure `data_times`, `window_times`, `lookback` and `lookforward`
    use the same unit!

    Parameters
    ----------
    data : np.ndarray
        The data to split.
    data_times : np.ndarray
        Sampling times of `data`.
    window_times : np.ndarray
        Times at which windows should be created.
    lookback : int
        Backward extension of the window (i.e. "to the left").
    lookforward : int
        Forward extension of the window (i.e. "to the left").

    Returns
    -------
    list[np.ndarray]
        A list containing each window as an array. Note that each window
        may contain a different number of elements in case the data is
        sampled irregularly.
    """
    window_start_times = window_times - lookback
    window_end_times = window_times + lookforward

    windows = []
    for start, end in zip(window_start_times, window_end_times):
        windows.append(data[(start <= data_times) & (data_times < end)])

    return windows


def _nanpsd(x: np.ndarray, fs: float, max_nans: float = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density (PSD) along axis 1, ignoring NaNs.

    The PSD is estimated via fourier transform and scaled to match the
    output of `scipy.signal.periodogram` with `scaling='density'`. For rows
    containing a fraction of NaNs higher than `max_nans`, the output array
    `Pxx` is filled with `np.nan`.

    Parameters
    ----------
    x : np.ndarray
        2d array where each row is treated as an individual signal.
    fs : float
        Sampling frequency in Hz.
    max_nans : float, optional
        Maximum fraction of NaNs in a signal (i.e. row of `x`), for which
        the PSD computation is attempted. Should be a value between `0.0`
        and `1.0`, by default `0`.

    Returns
    -------
    f : np.ndarray
        Array of sample frequencies.
    Pxx : np.ndarray
        One-sided power spectral density of x.
    """
    fft = np.full_like(x, np.nan, dtype=np.complex128)
    nfft = x.shape[1]

    nan_fraction = np.mean(np.isnan(x), axis=1)

    # rows without any NaNs
    full_rows_mask = nan_fraction == 0
    fft[full_rows_mask] = np.fft.fft(x[full_rows_mask])

    # remaining rows with less than max_nans NaNs
    for i in np.where((nan_fraction <= max_nans) ^ full_rows_mask)[0]:
        semi_valid_window = x[i]
        valid_part = semi_valid_window[~np.isnan(semi_valid_window)]
        fft[i] = np.fft.fft(valid_part, n=nfft)

    f = np.fft.fftfreq(nfft, 1/fs)[:nfft//2]
    Pxx = np.abs(fft)[:, :nfft//2] ** 2 / (nfft*2)
    return f, Pxx


def _hrv_timedomain_features(
    rri: np.ndarray,
    rri_times: np.ndarray,
    stage_times: np.ndarray,
    lookback: int,
    lookforward: int,
) -> np.ndarray:
    """
    Calculate time domain heart rate variability (HRV) features.

    Features are implemented according to [1]_.

    Parameters
    ----------
    rri : np.ndarray
        1d-array containing RR-intervals in seconds.
    rri_times : np.ndarray
        1d-array containing sample times of `rri` in seconds.
    stage_times : np.ndarray
        1d-array containing sleep stage onset times in seconds.
    lookback : int
        Backward extension of the analysis window from each sleep stage
        time.
    lookforward : int
        Forward extension of the analysis window from each sleep stage
        time.

    Returns
    -------
    np.ndarray
        Array of shape `(len(stage_times), 9)` containing the extracted
        time domain features.

    Notes
    -----
    .. [1] Task Force of the European Society of Cardiology. (1996). Heart
       rate variability: standards of measurement, physiological
       interpretation and clinical use. circulation, 93, 1043-1065.
       https://doi.org/10.1161/01.CIR.93.5.1043
    """
    # TODO: decide on biased/unbiased calculation of SDNN/SDSD

    NN = _split_into_windows(
        rri,
        rri_times,
        stage_times,
        lookback,
        lookforward,
    )
    NN = np.ma.masked_invalid(_create_ragged_array(NN))

    meanNN = np.nanmean(NN, axis=1)
    meanHR = 60 / meanNN
    maxNN = np.max(NN, axis=1)
    minNN = np.min(NN, axis=1)
    rangeNN = maxNN - minNN
    SDNN = np.nanstd(NN, axis=1)

    SD = np.diff(NN)
    RMSSD = np.sqrt(np.nanmean(SD**2, axis=1))
    SDSD = np.nanstd(SD, axis=1)
    pNN50 = np.nanmean(np.abs(SD) > 0.05, axis=1)

    return np.vstack((meanNN, meanHR, maxNN, minNN, rangeNN, SDNN, RMSSD, SDSD, pNN50)).T


def _hrv_frequencydomain_features(
    rri: np.ndarray,
    rri_times: np.ndarray,
    stage_times: np.ndarray,
    lookback: int = 0,
    lookforward: int = 30,
    fs_rri_resample: float = 4,
    max_nans: float = 0,
) -> np.ndarray:
    """
    Calculate frequency domain heart rate variability (HRV) features.

    Features are implemented according to [1]_.

    Parameters
    ----------
    rri : np.ndarray
        1d-array containing RR-intervals in seconds.
    rri_times : np.ndarray
        1d-array containing sample times of `rri` in seconds.
    stage_times : np.ndarray
        1d-array containing sleep stage onset times in seconds. Distances
        between onsets must be regular.
    lookback : int, optional
        Backward extension of the analysis window from each sleep stage
        time, by default `0`.
    lookforward : int, optional
        Forward extension of the analysis window from each sleep stage
        time, by default `30`.
    fs_rri_resample : float, optional
        Frequency in Hz at which the RRI time series should be resampled
        before spectral analysis, by default `4`.
    max_nans : float, optional
        Maximum fraction of NaNs in an analysis window, for which frequency
        features are computed. Should be a value between `0.0` and `1.0`,
        by default `0`.

    Returns
    -------
    np.ndarray
        Array of shape `(len(stage_times), 7)` containing the extracted
        frequency domain features.

    Notes
    -----
    .. [1] Task Force of the European Society of Cardiology. (1996). Heart
       rate variability: standards of measurement, physiological
       interpretation and clinical use. circulation, 93, 1043-1065.
       https://doi.org/10.1161/01.CIR.93.5.1043
    """
    # The recording should last for at least 10 times the wavelength of the
    # lower frequency bound of the investigated component.
    window_time = lookback + lookforward
    min_window_lengths = {
        'VLF': 10 * (1 / 0.0033),
        'LF': 10 * (1 / 0.04),
        'HF': 10 * (1 / 0.15),
    }
    for name, min_window_length in min_window_lengths.items():
        if window_time < min_window_length:
            msg = (
                f'HR analysis window too short for estimating PSD in {name} range. '
                f'{min_window_length:.1f}s required, got {window_time}s'
            )
            warnings.warn(msg, category=RuntimeWarning)

    rri_interp_times = np.arange(
        start=stage_times[0] - lookback,
        stop=stage_times[-1] + lookforward,
        step=1 / fs_rri_resample,
    )
    rri_interp = interp1d(rri_times, rri, bounds_error=False)(rri_interp_times)

    # create (overlapping) windows, 1 per sleep stage
    sleep_stage_durations = np.diff(stage_times)
    if np.any(sleep_stage_durations != sleep_stage_durations[0]):
        raise ValueError('Sleep stages must be sampled regularly!')

    window_size = (lookback + lookforward) * fs_rri_resample
    window_step = int(fs_rri_resample * sleep_stage_durations[0])
    rri_windows = sliding_window_view(rri_interp, window_size)[::window_step]

    rri_no_baseline = rri_windows - np.nanmean(rri_windows, axis=1)[:, np.newaxis]
    freq, psd = _nanpsd(rri_no_baseline, fs_rri_resample, max_nans)

    total_power_mask = freq <= 0.4
    vlf_mask = (0.0033 < freq) & (freq <= 0.04)
    lf_mask = (0.04 < freq) & (freq <= 0.15)
    hf_mask = (0.15 < freq) & (freq <= 0.4)

    total_power = np.trapz(psd[:, total_power_mask], freq[total_power_mask])
    vlf = np.trapz(psd[:, vlf_mask], freq[vlf_mask])
    lf = np.trapz(psd[:, lf_mask], freq[lf_mask])
    hf = np.trapz(psd[:, hf_mask], freq[hf_mask])

    lf_norm = lf / (lf + hf) * 100
    hf_norm = hf / (lf + hf) * 100
    lf_hf_ratio = lf / hf

    return np.vstack((total_power, vlf, lf, lf_norm, hf, hf_norm, lf_hf_ratio)).T


def extract_hrv_features(
    heartbeat_times: np.ndarray,
    sleep_stages: np.ndarray,
    fs_sleep_stages: float,
    lookback: int = 0,
    lookforward: int = 30,
    fs_rri_resample: float = 4,
    feature_groups: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Calculate heart rate variability (HRV) features.

    Time and frequency domain features are calculated based on [1]_.

    Parameters
    ----------
    heartbeat_times : np.ndarray
        1d-array containing times at which heartbeats happened.
    sleep_stages : np.ndarray
        1d-array containing sleep stages.
    fs_sleep_stages : float
        Sampling frequency of the sleep stages.
    lookback : int, optional
        Backward extension of the analysis window from each sleep stage
        time, by default `0`.
    lookforward : int, optional
        Forward extension of the analysis window from each sleep stage
        time, by default `30`.
    fs_rri_resample : float, optional
        Frequency in Hz at which the RRI time series should be resampled
        before spectral analysis. Only relevant for frequency domain
        features,  by default `4`.
    feature_groups : list[str], optional
        Which feature groups to extract. Allowed: `{'hrv-timedomain',
        'hrv-frequencydomain'}`. If `None` (default), all possible features
        are extracted.

    Returns
    -------
    np.ndarray
        Array of shape `(len(sleep_stages), 16)` containing the extracted
        time and frequency domain features.

    Notes
    -----
    .. [1] Task Force of the European Society of Cardiology. (1996). Heart
       rate variability: standards of measurement, physiological
       interpretation and clinical use. circulation, 93, 1043-1065.
       https://doi.org/10.1161/01.CIR.93.5.1043
    """
    # TODO: mask nonsense heartrates
    # TODO: remove windows with too many (or only) nans (?)
    # TODO: DataFrame/dict/ndarray for X?
    # TODO: remove ectopic beats (?)

    rri = np.diff(heartbeat_times)
    rri_times = heartbeat_times[1:]
    stage_times = np.arange(len(sleep_stages)) / fs_sleep_stages

    X = []

    if feature_groups is None or 'hrv-timedomain' in feature_groups:
        X.append(
            _hrv_timedomain_features(
                rri,
                rri_times,
                stage_times,
                lookback,
                lookforward,
            ),
        )

    if feature_groups is None or 'hrv-frequencydomain' in feature_groups:
        X.append(
            _hrv_frequencydomain_features(
                rri,
                rri_times,
                stage_times,
                lookback,
                lookforward,
                fs_rri_resample,
            ),
        )

    return np.hstack(X)
