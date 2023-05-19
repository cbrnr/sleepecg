# © SleepECG developers
#
# License: BSD (3-clause)

"""Plotting functions."""

from __future__ import annotations

from itertools import cycle
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

from sleepecg.io.sleep_readers import SleepRecord, SleepStage
from sleepecg.utils import _STAGE_INTS, _STAGE_NAMES, _merge_sleep_stages, _time_to_sec


def plot_ecg(
    ecg: np.ndarray,
    fs: float,
    title: Optional[str] = None,
    **kwargs: np.ndarray,
) -> tuple["plt.Figure", "plt.Axes"]:
    """
    Plot ECG time series with optional markers.

    Parameters
    ----------
    ecg : np.ndarray
        ECG signal.
    fs : float
        Sampling frequency of the ECG signal in Hz.
    title : str, optional
        Title of the plot.
    **kwargs : np.ndarray
        Positions of annotations (i.e. heartbeats) in samples. If more than one marker
        sequence is given, the keywords will be used as labels in the plot legend.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        The axes in the figure.

    Examples
    --------
    Plot ECG without any annotations:

    >>> plot(ecg, fs)

    Plot ECG with one set of annotations and a title:

    >>> plot(ecg, fs, title="ECG", markers=annotations)

    Plot ECG with two sets of annotations:

    >>> plot(ecg, fs, marker1=annotations, marker2=heartbeats)

    The last example will create two annotation series, the first one labeled `marker1` with
    positions given by `annotations`, and the second one labeled `marker2` with positions
    given by `heartbeats`.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap

    t = np.arange(0, len(ecg) / fs, 1 / fs)
    fig, ax = plt.subplots()
    ax.plot(t, ecg, color="dimgray", label=None)
    ax.set_xlabel("Time (s)")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    colors = cycle(get_cmap("tab10").colors)
    markers = cycle(("*", "o", "s", "D", "v", "<", ">", "^", "X", "p"))
    for label, pos, color, marker in zip(kwargs.keys(), kwargs.values(), colors, markers):
        ax.plot(
            t[pos],
            ecg[pos],
            marker=marker,
            markeredgecolor=color,
            markerfacecolor="None",
            linestyle="",
            label=label,
        )
    if len(kwargs) >= 2:  # add legend if there are at least two sequences of annotations
        plt.legend()
    if title is not None:
        ax.set_title(title)

    return fig, ax


def plot_hypnogram(
    record: SleepRecord,
    stages_pred: np.ndarray,
    stages_mode: str,
    stages_pred_duration: int = 30,
    merge_annotations: bool = False,
    show_bpm: bool = False,
) -> tuple["plt.Figure", list["plt.Axes"]]:
    """
    Plot a hypnogram for a single record.

    Annotated sleep stages are included in the plot if available in `record`. If
    `stages_pred` contains probabilities, they are shown in an additional subplot.

    Parameters
    ----------
    record : SleepRecord
        A single record (i.e. night).
    stages_pred : np.ndarray
        The predicted stages, either as a 1D array of integers or a 2D array of
        probabilties.
    stages_mode : str
        Identifier of the grouping mode. Can be any of `'wake-sleep'`, `'wake-rem-nrem'`,
        `'wake-rem-light-n3'`, or `'wake-rem-n1-n2-n3'`.
    stages_pred_duration : int, optional
        Duration of the predicted sleep stages in seconds, by default `30`.
    merge_annotations : bool, optional
        If `True`, merge annotations according to `stages_mode`, otherwise plot original
        annotations. By default `False`.
    show_bpm : bool, optional
        If `True`, include a subplot of the heart rate in bpm. This can be helpful to find
        bad signal quality intervals, by default `False`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : array of matplotlib.axes.Axes
        The axes in the figure.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    if record.sleep_stage_duration is None:
        raise ValueError(f"sleep_stage_duration not available for record {record.id}")

    stages_pred_probs = None
    num_subplots = 1
    if stages_pred.ndim == 2:
        num_subplots += 1
        stages_pred_probs = stages_pred
        stages_pred = stages_pred_probs.argmax(1)
    if record.sleep_stages is not None:
        num_subplots += 1
    if show_bpm:
        num_subplots += 1

    if record.recording_start_time is None:
        start_time = 0
    else:
        start_time = _time_to_sec(record.recording_start_time)

    fig, ax = plt.subplots(num_subplots, sharex=True, figsize=(7, 4))

    # predicted stages
    t_stages_pred = np.arange(len(stages_pred)) * stages_pred_duration + start_time
    t_stages_pred = t_stages_pred.astype("datetime64[s]")
    stages_pred = stages_pred.astype(float)
    stages_pred[stages_pred == SleepStage.UNDEFINED] = np.nan
    ax[0].plot(t_stages_pred, stages_pred)
    ax[0].set_yticks(_STAGE_INTS[stages_mode])
    ax[0].set_yticklabels(_STAGE_NAMES[stages_mode])
    ax[0].set_ylabel("predicted")
    ax[0].yaxis.tick_right()

    row = 1

    # predicted stage probabilities
    if stages_pred_probs is not None:
        ax[row].stackplot(
            t_stages_pred,
            stages_pred_probs[:, 1:].T,
            labels=_STAGE_NAMES[stages_mode],
        )
        ax[row].set_ylabel("probabilities")
        legend_handles, legend_labels = ax[row].get_legend_handles_labels()
        ax[row].legend(legend_handles[::-1], legend_labels[::-1], loc=(1.01, 0))
        ax[row].set_ylim(0, 1)
        ax[row].set_yticks([])
        row += 1

    # annotated stages
    if record.sleep_stages is not None:
        stages_true = record.sleep_stages
        t_stages_true = (
            np.arange(len(stages_true)) * record.sleep_stage_duration + start_time
        )
        t_stages_true = t_stages_true.astype("datetime64[s]")
        if merge_annotations:
            stages_true = _merge_sleep_stages([stages_true], stages_mode)[0]
            stages_mode_true = stages_mode
        else:
            stages_mode_true = "wake-rem-n1-n2-n3"
        stages_true = stages_true.astype(float)
        stages_true[stages_true == SleepStage.UNDEFINED] = np.nan

        ax[row].plot(t_stages_true, stages_true)
        ax[row].set_yticks(_STAGE_INTS[stages_mode_true])
        ax[row].set_yticklabels(_STAGE_NAMES[stages_mode_true])
        ax[row].set_ylabel("annotated")
        ax[row].yaxis.tick_right()

        row += 1

    # heartrate
    if show_bpm:
        if record.heartbeat_times is None:
            raise ValueError(f"heartbeat_times not available for record {record.id}")
        t_ecg = (record.heartbeat_times[1:] + start_time).astype("datetime64[s]")
        ax[row].plot(t_ecg, 60 / np.diff(record.heartbeat_times))
        ax[row].set_ylabel("beats per minute")
        ax[row].yaxis.tick_right()

    # x axis ticks and label
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    if record.recording_start_time is None:
        ax[-1].set_xlabel("time since recording start in hours")
    else:
        ax[-1].set_xlabel("time of day in hours")
    ax[-1].set_xlim(t_stages_pred[0], t_stages_pred[-1])

    return fig, ax


def _plot_confusion_matrix(
    confmat: np.ndarray,
    stage_names: list[str],
) -> tuple["plt.Figure", "plt.Axes"]:
    """
    Create a labeled plot of a confusion matrix.

    Parameters
    ----------
    confmat : np.ndarray
        A confusion matrix, as returned by `confusion_matrix()`.
    stage_names : list[str]
        Class labels which are used as tick labels.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        The axes in the figure.
    """
    import matplotlib.pyplot as plt

    classes = np.arange(len(confmat))

    fig, ax = plt.subplots()
    ax.imshow(confmat, cmap="Blues", vmin=0, vmax=confmat[1:, 1:].max())
    for i in range(len(stage_names)):
        for j in range(len(stage_names)):
            ax.text(j, i, f"{confmat[i, j]}", ha="center", va="center", color="k")

    ax.set_ylabel("Annotated Stage")
    ax.set_xlabel("Predicted Stage")
    ax.set_xticks(classes)
    ax.set_yticks(classes)
    ax.set_xticklabels(stage_names)
    ax.set_yticklabels(stage_names)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    return fig, ax
