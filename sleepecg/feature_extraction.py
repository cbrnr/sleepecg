# © SleepECG developers
#
# License: BSD (3-clause)

"""Functions and utilities related to feature extraction."""

import warnings
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import interp1d
from scipy.signal import periodogram

from .io.sleep_readers import SleepRecord
from .utils import _parallel, _time_to_sec

_FEATURE_GROUPS = {
    "hrv-time": (
        "meanNN",
        "maxNN",
        "minNN",
        "rangeNN",
        "SDNN",
        "RMSSD",
        "SDSD",
        "NN50",
        "NN20",
        "pNN50",
        "pNN20",
        "medianNN",
        "madNN",
        "iqrNN",
        "cvNN",
        "cvSD",
        "meanHR",
        "maxHR",
        "minHR",
        "stdHR",
        "SD1",
        "SD2",
        "S",
        "SD1_SD2_ratio",
        "CSI",
        "CVI",
    ),
    "hrv-frequency": (
        "total_power",
        "VLF",
        "LF",
        "LF_norm",
        "HF",
        "HF_norm",
        "LF_HF_ratio",
    ),
    "metadata": ("recording_start_time", "age", "gender", "weight"),
}
_FEATURE_ID_TO_GROUP = {id: group for group, ids in _FEATURE_GROUPS.items() for id in ids}


def _create_ragged_array(data: List[np.ndarray]) -> np.ndarray:
    """
    Convert a list of arrays with different lengths to a numpy array.

    Each element in `data` is a row in the resulting array. Rows shorter than the longest
    row will be padded with `np.nan`.

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
        ragged_array[row_index, : len(row)] = row
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

    Make sure `data_times`, `window_times`, `lookback` and `lookforward` use the same units.

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
        A list containing each window as an array. Note that each window may contain a
        different number of elements in case the data is sampled irregularly.
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

    For rows containing a fraction of NaNs higher than `max_nans`, the output array `Pxx` is
    filled with `np.nan`.

    Parameters
    ----------
    x : np.ndarray
        2d array where each row is treated as an individual signal.
    fs : float
        Sampling frequency in Hz.
    max_nans : float, optional
        Maximum fraction of NaNs in a signal (i.e. row of `x`), for which the PSD
        computation is attempted. Should be a value between `0.0` and `1.0`, by default `0`.

    Returns
    -------
    f : np.ndarray
        Array of sample frequencies.
    Pxx : np.ndarray
        One-sided power spectral density of x.
    """
    nfft = x.shape[1]
    Pxx = np.full((x.shape[0], nfft // 2 + 1), np.nan)

    nan_fraction = np.mean(np.isnan(x), axis=1)

    # rows without any NaNs
    full_rows_mask = nan_fraction == 0
    f, Pxx[full_rows_mask] = periodogram(x=x[full_rows_mask], fs=fs)

    # remaining rows with less than max_nans NaNs
    empty_rows_mask = nan_fraction == 1
    for i in np.where((nan_fraction <= max_nans) & ~(full_rows_mask | empty_rows_mask))[0]:
        semi_valid_window = x[i]
        valid_part = semi_valid_window[~np.isnan(semi_valid_window)]
        _, Pxx[i] = periodogram(valid_part, fs=fs, window="hann", nfft=nfft)

    return f, Pxx


def _hrv_timedomain_features(
    rri: np.ndarray,
    rri_times: np.ndarray,
    stage_times: np.ndarray,
    lookback: int,
    lookforward: int,
) -> np.ndarray:
    """
    Calculate 26 time domain heart rate variability (HRV) features.

    Features are implemented according to [1]_, [2]_ and [3]_.

    Parameters
    ----------
    rri : np.ndarray
        1d-array containing RR-intervals in seconds.
    rri_times : np.ndarray
        1d-array containing sample times of `rri` in seconds.
    stage_times : np.ndarray
        1d-array containing sleep stage onset times in seconds.
    lookback : int
        Backward extension of the analysis window from each sleep stage time.
    lookforward : int
        Forward extension of the analysis window from each sleep stage time.

    Returns
    -------
    np.ndarray
        Array of shape `(len(stage_times), 26)` containing the extracted time domain
        features.

    Notes
    -----
    .. [1] Task Force of the European Society of Cardiology. (1996). Heart rate variability:
       standards of measurement, physiological interpretation and clinical use. Circulation,
       93, 1043-1065. https://doi.org/10.1161/01.CIR.93.5.1043
    .. [2] Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability
       metrics and norms. Frontiers in Public Health, 258.
       https://doi.org/10.3389/fpubh.2017.00258
    .. [3] Toichi, M., Sugiura, T., Murai, T., & Sengoku, A. (1997). A new method of
       assessing cardiac autonomic function and its comparison with spectral analysis and
       coefficient of variation of R–R interval. Journal of the Autonomic Nervous System,
       62(1-2), 79-84. https://doi.org/10.1016/S0165-1838(96)00112-9
    """
    NN = _split_into_windows(
        rri,
        rri_times,
        stage_times,
        lookback,
        lookforward,
    )
    NN = _create_ragged_array(NN)

    meanNN = np.nanmean(NN, axis=1)
    maxNN = np.nanmax(NN, axis=1)
    minNN = np.nanmin(NN, axis=1)
    rangeNN = maxNN - minNN
    SDNN = np.nanstd(NN, axis=1, ddof=1)

    SD = np.diff(NN)
    RMSSD = np.sqrt(np.nanmean(SD**2, axis=1))
    SDSD = np.nanstd(SD, axis=1, ddof=1)
    NN50 = np.nansum(np.abs(SD) > 0.05, axis=1)
    NN20 = np.nansum(np.abs(SD) > 0.02, axis=1)
    pNN50 = np.nanmean(np.abs(SD) > 0.05, axis=1)
    pNN20 = np.nanmean(np.abs(SD) > 0.02, axis=1)

    medianNN = np.nanmedian(NN, axis=1)
    madNN = np.nanmedian(np.abs(NN - medianNN[:, np.newaxis]), axis=1)
    iqrNN = np.nanpercentile(NN, 75, axis=1) - np.nanpercentile(NN, 25, axis=1)

    cvNN = SDNN / meanNN
    cvSD = SDSD / np.nanmean(SD, axis=1)

    meanHR = 60 / meanNN
    maxHR = 60 / minNN
    minHR = 60 / maxNN
    stdHR = np.nanstd(60 / NN, axis=1, ddof=1)

    SD1 = (SDSD**2 * 0.5) ** 0.5
    SD2 = (2 * SDNN**2 - SD1**2) ** 0.5
    S = np.pi * SD1 * SD2
    SD1_SD2_ratio = SD1 / SD2

    CSI = SD2 / SD1
    CVI = np.log10(SD1 * SD2 * 16)

    return np.vstack(
        (
            meanNN,
            maxNN,
            minNN,
            rangeNN,
            SDNN,
            RMSSD,
            SDSD,
            NN50,
            NN20,
            pNN50,
            pNN20,
            medianNN,
            madNN,
            iqrNN,
            cvNN,
            cvSD,
            meanHR,
            maxHR,
            minHR,
            stdHR,
            SD1,
            SD2,
            S,
            SD1_SD2_ratio,
            CSI,
            CVI,
        )
    ).T


def _hrv_frequencydomain_features(
    rri: np.ndarray,
    rri_times: np.ndarray,
    stage_times: np.ndarray,
    lookback: int,
    lookforward: int,
    fs_rri_resample: float,
    max_nans: float,
    feature_ids: List[str],
) -> np.ndarray:
    """
    Calculate seven frequency domain heart rate variability (HRV) features.

    Features are implemented according to [1]_.

    Parameters
    ----------
    rri : np.ndarray
        1d-array containing RR-intervals in seconds.
    rri_times : np.ndarray
        1d-array containing sample times of `rri` in seconds.
    stage_times : np.ndarray
        1d-array containing sleep stage onset times in seconds. Distances between onsets
        must be regular.
    lookback : int
        Backward extension of the analysis window from each sleep stage time.
    lookforward : int
        Forward extension of the analysis window from each sleep stage time.
    fs_rri_resample : float
        Frequency in Hz at which the RRI time series should be resampled before spectral
        analysis.
    max_nans : float
        Maximum fraction of NaNs in an analysis window, for which frequency features are
        computed. Should be a value between `0.0` and `1.0`.
    feature_ids : list[str]
        A list containing the identifiers of all features to be extracted. This does not
        change the returned array. It is only used to avoid issuing a warning about the
        analysis window being too short for some frequency range which is not requested.

    Returns
    -------
    np.ndarray
        Array of shape `(len(stage_times), 7)` containing the extracted frequency domain
        features.

    Notes
    -----
    .. [1] Task Force of the European Society of Cardiology. (1996). Heart rate variability:
       standards of measurement, physiological interpretation and clinical use. Circulation,
       93, 1043-1065. https://doi.org/10.1161/01.CIR.93.5.1043
    """
    # The recording should last for at least 10 times the wavelength of the lower frequency
    # bound of the investigated component.
    window_time = lookback + lookforward
    min_frequencies = {
        "VLF": 0.0033,
        "LF": 0.04,
        "LF_norm": 0.04,
        "HF": 0.15,
        "HF_norm": 0.04,
        "LF_HF_ratio": 0.04,
    }

    for name, min_frequency in min_frequencies.items():
        min_window_time = 10 / min_frequency
        if name not in feature_ids:
            continue
        if window_time < min_window_time:
            msg = (
                f"HR analysis window too short for estimating PSD for feature {name}. "
                f"{min_window_time:.1f}s required, got {window_time}s"
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
        raise ValueError("Sleep stages must be sampled regularly!")

    window_size = (lookback + lookforward) * fs_rri_resample
    window_step = int(fs_rri_resample * sleep_stage_durations[0])
    rri_windows = sliding_window_view(rri_interp, window_size)[::window_step]

    freq, psd = _nanpsd(rri_windows, fs_rri_resample, max_nans)

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


def _metadata_features(record: SleepRecord, num_stages: int) -> np.ndarray:
    """
    Create a feature matrix from record metadata.

    Recording start time, gender, age, and weight are used as (constant) features. In case
    of missing information (i.e. the required attribute of `SleepRecord` is `None`), the
    corresponding column is filled with `np.nan`.

    Parameters
    ----------
    record : SleepRecord
        The record from which to extract metadata features.
    num_stages : int
        The length (number of rows) of the feature matrix to be created.

    Returns
    -------
    np.ndarray
        Array of shape `(num_stages, 4)` containing the extracted features.
    """
    feature_ids = _FEATURE_GROUPS["metadata"]

    X = np.full((num_stages, len(feature_ids)), np.nan)

    # recording start time
    if record.recording_start_time is not None:
        X[:, 0] = _time_to_sec(record.recording_start_time)

    if record.subject_data is None:
        return X

    # subject data features
    if record.subject_data.age is not None:
        X[:, 1] = record.subject_data.age
    if record.subject_data.gender is not None:
        X[:, 2] = record.subject_data.gender
    if record.subject_data.weight is not None:
        X[:, 3] = record.subject_data.weight

    return X


def _parse_feature_selection(
    requested_ids: List[str],
) -> Tuple[List[str], List[str], List[int]]:
    """
    Parse a list containing feature group names and single feature IDs.

    Each feature group is expanded to all its feature identifiers as listed in
    `feature_extraction._FEATURE_GROUPS`, preserving input order. If an invalid (group) ID
    is found, a `ValueError` is raised.

    Parameters
    ----------
    requested_ids : list[str]
        A list which can contain both feature group names and single feature IDs in
        arbitrary order.

    Returns
    -------
    required_groups : list[str]
        The feature groups which have to be calculated to cover all requested features.
    feature_ids : list[str]
        The expanded list containing only single feature IDs.
    selected_cols : list[int]
        The column indices of `feature_ids` in a list of all feature IDs in all
        `required_groups`.
    """
    required_groups = set()
    feature_ids = []

    for id_ in requested_ids:
        if id_ in _FEATURE_GROUPS:
            required_groups.add(id_)
            feature_ids.extend(_FEATURE_GROUPS[id_])
        elif id_ in _FEATURE_ID_TO_GROUP:
            required_groups.add(_FEATURE_ID_TO_GROUP[id_])
            feature_ids.append(id_)
        else:
            raise ValueError(f"Invalid feature or group ID: {id_}")

    all_cols = [id for group in required_groups for id in _FEATURE_GROUPS[group]]
    selected_cols = [all_cols.index(id_) for id_ in feature_ids]

    duplicate_ids = {x for x in feature_ids if feature_ids.count(x) > 1}
    if duplicate_ids:
        warnings.warn(f"Duplicates in feature selection: {duplicate_ids}", RuntimeWarning)

    return list(required_groups), feature_ids, selected_cols


def preprocess_rri(
    rri: np.ndarray,
    min_rri: Optional[float] = None,
    max_rri: Optional[float] = None,
):
    """
    Replace invalid RRI samples with `np.nan`.

    Parameters
    ----------
    rri : np.ndarray
        An array containing consecutive RR interval lengths in seconds.
    min_rri : float, optional
        Minimum RRI in seconds to be considered valid. If `None` (default), no lower bounds
        check is performed.
    max_rri : float, optional
        Maximum RRI in seconds to be considered valid. If `None` (default), no upper bounds
        check is performed.

    Returns
    -------
    np.ndarray
        The cleaned RRI series.

    Examples
    --------
    Mask RR intervals outside the range of 0.4 to 2 s (= 30 to 150 bpm):

    >>> from sleepecg import preprocess_rri
    >>> preprocess_rri([0.5, 0.2, 0.8, 2.5, 0.6], min_rri=0.4, max_rri=2)
    array([0.5, nan, 0.8, nan, 0.6])
    """
    rri = np.array(rri, dtype=float)  # make a copy
    if min_rri is not None:
        rri[rri < min_rri] = np.nan
    if max_rri is not None:
        rri[rri > max_rri] = np.nan
    return rri


def _extract_features_single(
    record: SleepRecord,
    sleep_stage_duration: int,
    min_rri: float,
    max_rri: float,
    required_groups: List[str],
    lookback: int,
    lookforward: int,
    fs_rri_resample: float,
    max_nans: float,
    feature_ids: List[str],
    col_indices: List[int],
) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
    """
    Calculate features for a single record.

    This function is required to allow parallelizing feature extraction.

    Parameters
    ----------
    record : SleepRecord
        The record for which to calculate features.
    sleep_stage_duration : int
        Duration of a single sleep stage in the returned `stages` in seconds.
    min_rri: float, optional
        Minimum RRI value in seconds to be considered valid. Will be passed to
        :func:`preprocess_rri`.
    max_rri: float, optional
        Maximum RRI value in seconds to be considered valid. Will be passed to
        :func:`preprocess_rri`.
    required_groups : list[str]
        The feature groups which have to be calculated to cover all requested features.
    lookback : int, optional
        Backward extension of the analysis window from each sleep stage time in seconds.
    lookforward : int, optional
        Forward extension of the analysis window from each sleep stage time in seconds.
    fs_rri_resample : float
        Frequency in Hz at which the RRI time series should be resampled before spectral
        analysis. Only relevant for frequency domain features.
    max_nans : float
        Maximum fraction of NaNs in an analysis window for which frequency features are
        computed. Should be a value between `0.0` and `1.0`.
    feature_ids : list[str]
        A list containing the identifiers of all features to be extracted. This is only used
        to avoid issuing a warning about the analysis window being too short for some
        frequency range which is not requested.
    col_indices : list[int]
        The column indices of `feature_ids` in a list of all feature IDs in all
        `required_groups`. Required to select the columns corresponding to the requested
        `feature_ids` from the calculated feature matrix.

    Returns
    -------
    features : np.ndarray
        The feature matrix of shape `(len(sleep_stages), <num_features>)` containing the
        extracted features.
    stages : np.ndarray | None
        The label vector, i.e. the annotated sleep stages. For a `record` without annotated
        stages, this will be `None`.
    """
    rri_required = "hrv-time" in required_groups or "hrv-frequency" in required_groups

    if record.sleep_stages is not None and record.sleep_stage_duration is not None:
        record_duration = len(record.sleep_stages) * record.sleep_stage_duration
    elif record.heartbeat_times is not None:
        record_duration = record.heartbeat_times[-1]
    else:
        raise ValueError(f"Record duration cannot be inferred for {record.id}.")
    num_stages = int(record_duration // sleep_stage_duration)
    stage_times = np.arange(num_stages) * sleep_stage_duration

    if rri_required:
        if record.heartbeat_times is None:
            raise ValueError(
                f"Cannot extract HRV features for {record.id} without heartbeat_times."
            )
        rri = preprocess_rri(
            np.diff(record.heartbeat_times),
            min_rri=min_rri,
            max_rri=max_rri,
        )
        rri_times = record.heartbeat_times[1:]

    X = []
    for feature_group in required_groups:
        if feature_group == "hrv-time":
            X.append(
                _hrv_timedomain_features(
                    rri,
                    rri_times,
                    stage_times,
                    lookback,
                    lookforward,
                ),
            )
        elif feature_group == "hrv-frequency":
            X.append(
                _hrv_frequencydomain_features(
                    rri,
                    rri_times,
                    stage_times,
                    lookback,
                    lookforward,
                    fs_rri_resample,
                    max_nans,
                    feature_ids,
                ),
            )
        elif feature_group == "metadata":
            X.append(_metadata_features(record, num_stages))
    features = np.hstack(X)[:, col_indices]

    if record.sleep_stages is None or sleep_stage_duration == record.sleep_stage_duration:
        stages = record.sleep_stages
    else:
        stages = interp1d(
            np.arange(len(record.sleep_stages)) * record.sleep_stage_duration,
            record.sleep_stages,
            kind="nearest",
            bounds_error=False,
            fill_value=(record.sleep_stages[0], record.sleep_stages[-1]),
        )(stage_times)

    return features, stages


def extract_features(
    records: Iterable[SleepRecord],
    lookback: int = 0,
    lookforward: int = 30,
    sleep_stage_duration: int = 30,
    feature_selection: Optional[List[str]] = None,
    fs_rri_resample: float = 4,
    min_rri: Optional[float] = None,
    max_rri: Optional[float] = None,
    max_nans: float = 0,
    n_jobs: int = 1,
) -> Tuple[List[np.ndarray], List[Union[np.ndarray, None]], List[str]]:
    """
    Calculate features from sleep data (e.g. heart rate).

    Time and frequency domain heart rate variability (HRV) features are calculated based on
    [1]_, [2]_ and [3]_. :ref:`feature_extraction` lists all available features and feature
    groups.

    Parameters
    ----------
    records : Iterable[SleepRecord]
        An iterable of `SleepRecord` objects, as yielded by the various reader functions in
        SleepECG.
    lookback : int, optional
        Backward extension of the analysis window from each sleep stage time in seconds, by
        default `0`.
    lookforward : int, optional
        Forward extension of the analysis window from each sleep stage time in seconds, by
        default `30`.
    sleep_stage_duration : int, optional
        Duration of a single sleep stage in the returned `stages` in seconds, by default
        `30`.
    feature_selection : list[str], optional
        Which features to extract. Can be feature groups or single feature identifiers, as
        listed :ref:`here<feature_extraction>`. If `None` (default), all possible features
        are extracted.
    fs_rri_resample : float, optional
        Frequency in Hz at which the RRI time series should be resampled before spectral
        analysis. Only relevant for frequency domain features, by default `4`.
    min_rri: float, optional
        Minimum RRI value in seconds to be considered valid. Will be passed to
        :func:`preprocess_rri`, by default `None`.
    max_rri: float, optional
        Maximum RRI value in seconds to be considered valid. Will be passed to
        :func:`preprocess_rri`, by default `None`.
    max_nans : float, optional
        Maximum fraction of NaNs in an analysis window, for which frequency features are
        computed. Should be a value between `0.0` and `1.0`, by default `0`.
    n_jobs : int, optional
        The number of jobs to run in parallel. If `1` (default), no parallelism is used.
        `-1` means using all processors.

    Returns
    -------
    features : list[np.ndarray]
        A list containing feature matrices which are arrays of shape
        `(len(sleep_stages), <num_features>)` and contain the extracted features per record.
    stages : list[np.ndarray | None]
        A list containing label vectors, i.e. the annotated sleep stages. For any
        `SleepRecord` without annotated stages, the corresponding list entry will be `None`.
    feature_ids : list[str]
        A list containing the identifiers of the extracted features. Feature groups passed
        in `feature_selection` are expanded to all individual features they contain. The
        order matches the column order of `X`.

    Notes
    -----
    .. [1] Task Force of the European Society of Cardiology. (1996). Heart rate variability:
       standards of measurement, physiological interpretation and clinical use. Circulation,
       93, 1043-1065. https://doi.org/10.1161/01.CIR.93.5.1043
    .. [2] Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability
       metrics and norms. Frontiers in Public Health, 258.
       https://doi.org/10.3389/fpubh.2017.00258
    .. [3] Toichi, M., Sugiura, T., Murai, T., & Sengoku, A. (1997). A new method of
       assessing cardiac autonomic function and its comparison with spectral analysis and
       coefficient of variation of R–R interval. Journal of the Autonomic Nervous System,
       62(1-2), 79-84. https://doi.org/10.1016/S0165-1838(96)00112-9
    """
    if feature_selection is None:
        feature_selection = list(_FEATURE_GROUPS)

    required_groups, feature_ids, col_indices = _parse_feature_selection(feature_selection)

    # _extract_features_single has two return values, so the list returned by _parallel
    # needs to be unpacked
    Xy = _parallel(
        n_jobs,
        _extract_features_single,
        records,
        sleep_stage_duration,
        min_rri,
        max_rri,
        required_groups,
        lookback,
        lookforward,
        fs_rri_resample,
        max_nans,
        feature_ids,
        col_indices,
    )
    features = [X for X, _ in Xy]
    stages = [y for _, y in Xy]

    return features, stages, feature_ids
