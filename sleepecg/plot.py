# © SleepECG developers
#
# License: BSD (3-clause)

"""Plotting functions."""

from itertools import cycle
from typing import Optional

import numpy as np


def plot_ecg(
    ecg: np.ndarray,
    fs: float,
    annotations: Optional[np.ndarray] = None,
    *args: Optional[np.ndarray],
    title: str = None,
) -> None:
    """
    Plot ECG time series with optional markers.

    Parameters
    ----------
    ecg : np.ndarray
        ECG signal.
    fs : float
        Sampling frequency of the ECG signal in Hz.
    annotations : np.ndarray, optional
        Positions of annotations (i.e. heartbeats) in samples.
    *args : np.ndarray, optional
        Additional annotations to be plotted with different markers.
    title : str, optional
        Title of the plot.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap

    t = np.arange(0, len(ecg) / fs, 1 / fs)
    fig, ax = plt.subplots()
    ax.plot(t, ecg, color="dimgray")
    ax.set_xlabel("Time (s)")
    if annotations is not None:
        ax.plot(
            t[annotations],
            ecg[annotations],
            marker="*",
            markeredgecolor="green",
            markerfacecolor="None",
            linestyle="",
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    colors = cycle(get_cmap("Set1").colors)
    markers = cycle(("o", "s", "D", "v", "<", ">", "^", "p", "X"))
    for arg, color, marker in zip(args, colors, markers):
        ax.plot(
            t[arg],
            ecg[arg],
            marker=marker,
            markeredgecolor=color,
            markerfacecolor="None",
            linestyle="",
        )
    if title is not None:
        ax.set_title(title)
