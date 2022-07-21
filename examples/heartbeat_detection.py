# %% Imports
import matplotlib.pyplot as plt
import numpy as np

from sleepecg import compare_heartbeats, detect_heartbeats, read_mitdb

# %% Download and read data, run detector
record = list(read_mitdb(records_pattern="234"))[1]
detection = detect_heartbeats(record.ecg, record.fs)


# %% Evaluation and visualization
TP, FP, FN = compare_heartbeats(detection, record.annotation, int(record.fs / 10))

t = np.arange(len(record.ecg)) / record.fs

fig, ax = plt.subplots(3, sharex=True, figsize=(10, 8))

ax[0].plot(t, record.ecg, color="k", zorder=1, label="ECG")
ax[0].scatter(
    record.annotation / record.fs,
    record.ecg[record.annotation],
    marker="o",
    color="g",
    s=50,
    zorder=2,
    label="annotation",
)
ax[0].set_ylabel("raw signal in mV")

ax[1].eventplot(
    detection / record.fs,
    linelength=0.5,
    linewidth=0.5,
    color="k",
    zorder=1,
    label="detection",
)
ax[1].scatter(
    FN / record.fs,
    np.ones_like(FN),
    marker="x",
    color="r",
    s=70,
    zorder=2,
    label="FN",
)
ax[1].scatter(
    FP / record.fs,
    np.ones_like(FP),
    marker="+",
    color="orange",
    s=70,
    zorder=2,
    label="FP",
)
ax[1].set_yticks([])
ax[1].set_ylabel("heartbeat events")

ax[2].plot(
    detection[1:] / record.fs,
    60 / (np.diff(detection) / record.fs),
    label="heartrate in bpm",
)
ax[2].set_ylabel("beats per minute")
ax[2].set_xlabel("time in seconds")

for ax_ in ax.flat:
    ax_.legend(loc="upper right")
    ax_.grid(axis="x")

fig.suptitle(
    f"Record ID: {record.id}, lead: {record.lead}\n"
    + f"Recall: {len(TP) / (len(TP) + len(FN)):.2%}, "
    + f"Precision: {len(TP) / (len(TP) + len(FP)):.2%}",
)

plt.show()
