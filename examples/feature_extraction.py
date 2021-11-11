# %%
import numpy as np

from sleepecg import extract_features
from sleepecg.io import SleepRecord

# Generate dummy data while we don't have reader functions for sleep data
recording_hours = 8
heartbeat_times = np.cumsum(np.random.uniform(0.5, 1.5, recording_hours * 3600))
sleep_stages = np.random.randint(1, 6, int(max(heartbeat_times)) // 30)
fs_sleep_stages = 1/30

rec = SleepRecord(
    heartbeat_times=heartbeat_times,
    sleep_stages=sleep_stages,
    fs_sleep_stages=fs_sleep_stages,
)

features, names = extract_features(
        [rec],
        lookback=30,
        lookforward=90,
        feature_selection=['hrv-time', 'hrv-frequency'],
)
X = features[0]
