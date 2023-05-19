# © SleepECG developers
#
# License: BSD (3-clause)

"""Utility functions."""

from __future__ import annotations

import datetime
import warnings
from typing import Any, Callable, Iterable, TypeVar

import numpy as np

from sleepecg.io.sleep_readers import SleepStage

# required to propagate the return type annotation through _parallel
_Returnable = TypeVar("_Returnable")


def _parallel(
    n_jobs: int,
    function: Callable[..., _Returnable],
    iterable: Iterable,
    *args: Any,
    **kwargs: Any,
) -> list[_Returnable]:
    """
    Apply a function to each element in an iterable in parallel.

    This uses joblib for parallelism. If the package is not available, it falls back to a
    pure Python loop.

    Parameters
    ----------
    n_jobs : int
        The number of jobs to run in parallel. If `1` (default), no parallelism is used.
        `-1` means using all processors.
    function : Callable
        The function to call.
    iterable : Iterable
        `function` will be applied to every element in this iterable.
    *args
        Positional arguments to be passed to `function`.
    **kwargs
        Keyword arguments to be passed to `function`.

    Returns
    -------
    list
        A list containing the return values of `function`.

    Warnings
    --------
    Note that in case the `function` is very simple, the cost for spawning workers will make
    the parallel execution slower than the standard execution.

    Examples
    --------
    To parallelize this:

    >>> def root(a, n):
    >>>     return a**(1/n)
    >>> res = [root(x, 3) for x in range(1000)]

    Write this:

    >>> n_jobs = 4
    >>> res = _parallel(n_jobs, root, range(1000), 3)
    """
    try:
        from joblib import Parallel, delayed
    except ImportError:
        if n_jobs != 1:
            warnings.warn("joblib not installed, cannot run in parallel.")
        return [function(x, *args, **kwargs) for x in iterable]

    return Parallel(n_jobs=n_jobs)(delayed(function)(x, *args, **kwargs) for x in iterable)


def _time_to_sec(time: datetime.time) -> int:
    """
    Convert a `datetime.time` to seconds.

    `00:00:00` corresponds to `0` and `23:59:59` to `86399`.

    Parameters
    ----------
    time : datetime.time
        A `datetime.time` to convert.

    Returns
    -------
    int
        Time of day in seconds.
    """
    return time.hour * 3600 + time.minute * 60 + time.second


# Classifiers don't always discriminate between all sleep stages defined by the AASM
# guidelines. This dictionary is used to create a consistent mapping from groups of AASM
# sleep stages (as defined in `SleepStage`) to integers. `SleepStage.UNDEFINED` is always
# `0` and the actual stages' values increase with wakefulness, so they map correctly to the
# y-axis in a hypnogram plot. Gaps between stage values are avoided as non-existing classes
# in a one-hot encoding leads to issues when calculating class weights and losses.

_SLEEP_STAGE_MAPPING = {
    "wake-sleep": {
        SleepStage.WAKE: 2,
        SleepStage.REM: 1,
        SleepStage.N1: 1,
        SleepStage.N2: 1,
        SleepStage.N3: 1,
    },
    "wake-rem-nrem": {
        SleepStage.WAKE: 3,
        SleepStage.REM: 2,
        SleepStage.N1: 1,
        SleepStage.N2: 1,
        SleepStage.N3: 1,
    },
    "wake-rem-light-n3": {
        SleepStage.WAKE: 4,
        SleepStage.REM: 3,
        SleepStage.N1: 2,
        SleepStage.N2: 2,
        SleepStage.N3: 1,
    },
    "wake-rem-n1-n2-n3": {
        SleepStage.WAKE: 5,
        SleepStage.REM: 4,
        SleepStage.N1: 3,
        SleepStage.N2: 2,
        SleepStage.N3: 1,
    },
}

_STAGE_NAMES = {m: m.upper().split("-")[::-1] for m in _SLEEP_STAGE_MAPPING}
_STAGE_INTS = {k: sorted(set(v.values())) for k, v in _SLEEP_STAGE_MAPPING.items()}


def _merge_sleep_stages(stages: list[np.ndarray], stages_mode: str) -> list[np.ndarray]:
    """
    Merge sleep stage labels into groups.

    Parameters
    ----------
    stages : list[np.ndarray]
        A list of 1D arrays containing AASM sleep stages as defined by `SleepStage`, e.g. as
        returned by `extract_features()`.
    stages_mode : str
        Identifier of the grouping mode. Can be any of `'wake-sleep'`, `'wake-rem-nrem'`,
        `'wake-rem-light-n3'`, `'wake-rem-n1-n2-n3'`.

    Returns
    -------
    list[np.ndarray]
        A list of 1D arrays containing merged sleep stages.
    """
    if stages_mode not in _SLEEP_STAGE_MAPPING:
        options = list(_SLEEP_STAGE_MAPPING.keys())
        raise ValueError(f"Invalid stages_mode: {stages_mode}. Possible options: {options}")

    new_stages = []
    for array in stages:
        new_array = np.full_like(array, fill_value=SleepStage.UNDEFINED)
        for source_stage, target_stage in _SLEEP_STAGE_MAPPING[stages_mode].items():
            new_array[array == source_stage] = target_stage
        new_stages.append(new_array)
    return new_stages
