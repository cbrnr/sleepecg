# © SleepECG developers
#
# License: BSD (3-clause)

"""Utility functions."""


import datetime
import warnings
from typing import Callable, Iterable, List, TypeVar

# required to propagate the return type annotation through _parallel
_Returnable = TypeVar("_Returnable")


def _parallel(
    n_jobs: int,
    function: Callable[..., _Returnable],
    iterable: Iterable,
    *args,
    **kwargs,
) -> List[_Returnable]:
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
