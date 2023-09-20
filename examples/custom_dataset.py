# %% imports (this example requires mne and tensorflow packages)
from datetime import datetime, timezone

from mne.io import read_raw_edf

import sleepecg

# %% load dataset sleep.edf
raw = read_raw_edf("data/sleep.edf", include="ECG")
raw.set_channel_types({"ECG": "ecg"})
fs = raw.info["sfreq"]

# %% crop dataset (we only want data for the sleep duration)
start = datetime(2023, 3, 1, 23, 0, 0, tzinfo=timezone.utc)
stop = datetime(2023, 3, 2, 6, 0, 0, tzinfo=timezone.utc)
raw.crop((start - raw.info["meas_date"]).seconds, (stop - raw.info["meas_date"]).seconds)

# %% get ECG time series as 1D NumPy array
ecg = raw.get_data().squeeze()

# %% detect heartbeats
beats = sleepecg.detect_heartbeats(ecg, fs)
sleepecg.plot_ecg(ecg, fs, beats=beats)

# %% load SleepECG classifier (requires tensorflow)
clf = sleepecg.load_classifier("wrn-gru-mesa", "SleepECG")

# %% predict sleep stages
record = sleepecg.SleepRecord(
    sleep_stage_duration=30,
    recording_start_time=start,
    heartbeat_times=beats / fs,
)

stages = sleepecg.stage(clf, record, return_mode="prob")

sleepecg.plot_hypnogram(
    record,
    stages,
    stages_mode=clf.stages_mode,
    merge_annotations=True,
)
