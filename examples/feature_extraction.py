# %%
import numpy as np

from sleepecg import SleepRecord, extract_features

# Generate dummy data while we don't have reader functions for sleep data
recording_hours = 8
heartbeat_times = np.cumsum(np.random.uniform(0.5, 1.5, recording_hours * 3600))
sleep_stages = np.random.randint(1, 6, int(max(heartbeat_times)) // 30)
sleep_stage_duration = 30

rec = SleepRecord(
    sleep_stages=sleep_stages,
    sleep_stage_duration=sleep_stage_duration,
    heartbeat_times=heartbeat_times,
)

features, stages, feature_ids = extract_features(
    [rec],
    lookback=240,
    lookforward=60,
    feature_selection=["hrv-time", "LF_norm", "HF_norm", "LF_HF_ratio"],
)
X = features[0]
