# Feature extraction

## Heart rate variability features

Features are based on standards of heart rate variability (HRV) measurement and interpretation described in [Task Force of the European Society of Cardiology (1996)](https://doi.org/10.1161/01.CIR.93.5.1043) and [Shaffer & Ginsberg (2017)](https://doi.org/10.3389/fpubh.2017.00258).


### Time domain

Group identifier: `hrv-time`

All time domain HRV features are either derived from normal-to-normal (NN) intervals, from successive differences between NN intervals (SD), or from the [Poincaré plot (PP)](https://en.wikipedia.org/wiki/Poincar%C3%A9_plot).

!!! note
    
    SleepECG expects time in *seconds* to compute features. Since many features like RMSSD are typically specified in *milliseconds* in the field of heart rate variability research, the resulting features need to be manually rescaled before comparing to millisecond-based features.

|Feature|Description|Signal|
|-|-|-|
|`meanNN`|average normal-normal (NN) interval|NN|
|`maxNN`|maximum NN interval|NN|
|`minNN`|minimum NN interval|NN|
|`rangeNN`|difference between the longest and shortest NN interval|NN|
|`SDNN`|standard deviation of the NN intervals|NN|
|`RMSSD`|square root of the mean of the squares of differences between adjacent NN intervals|SD|
|`SDSD`|standard deviation of differences between adjacent NN intervals|SD|
|`NN50`|number of pairs of adjacent NN intervals differing by more than 50 ms|SD|
|`NN20`|number of pairs of adjacent NN intervals differing by more than 20 ms|SD|
|`pNN50`|percentage of pairs of adjacent NN intervals differing by more than 50 ms|SD|
|`pNN20`|percentage of pairs of adjacent NN intervals differing by more than 20 ms|SD|
|`medianNN`|median of the NN intervals|NN|
|`madNN`|median absolute deviation of the NN intervals|NN|
|`iqrNN`|interquartile range (IQR) of the NN intervals|NN|
|`cvNN`|coefficient of variation of the NN intervals|NN|
|`cvSD`|coefficient of variation of differences between adjacent NN intervals|SD|
|`meanHR`|average heart rate (HR = 60 / NN)|NN|
|`maxHR`|maximum heart rate|NN|
|`minHR`|minimum heart rate|NN|
|`stdHR`|standard deviation of the heart rate|NN|
|`SD1`|standard deviation along the short axis of the ellipse (corresponding to short term variability, equivalent to `RMSSD`)|PP|
|`SD2`|standard deviation along the long axis of the ellipse (corresponding to long term variability)|PP|
|`S`|area of the ellipse|PP|
|`SD1_SD2_ratio`|ratio of `SD1` to `SD2`|PP|
|`CSI`|cardiac sympathetic index|PP|
|`CVI`|cardiac vagal index|PP|


### Frequency domain

Group identifier: `hrv-frequency`

For calculating frequency domain HRV features, the RR time series is resampled at regular intervals, after which the power spectral density (PSD) is estimated using [Welch's method](https://en.wikipedia.org/wiki/Welch%27s_method).

|Feature|Description|Frequency range|
|-|-|-|
|`total_power`|variance of NN intervals over the temporal segment|≤ 0.4 Hz|
|`VLF`|power in very low frequency (VLF) range|[0.0033, 0.04) Hz|
|`LF`|power in low frequency (LF) range|[0.04, 0.15) Hz|
|`HF`|power in high frequency (HF) range|[0.15, 0.4) Hz|
|`LF_norm`|LF power in normalized units ($\frac{\text{LF}}{\text{LF}+\text{HF}}\cdot100$)|[0.04, 0.15) Hz|
|`HF_norm`|HF power in normalized units ($\frac{\text{HF}}{\text{LF}+\text{HF}}\cdot100$)|[0.15, 0.4) Hz|
|`LF_HF_ratio`|ratio of LF to HF|–|


## Metadata features

Group identifier: `metadata`

|Feature|Description|
|-|-|
|`recording_start_time`|time at which the recording was started in seconds (`0` corresponds to `00:00:00` and `86399` to `23:59:59`)|
|`age`|age of the subject in years|
|`gender`|`0` (female) or `1` (male)|
|`weight`|weight of the subject in kg|


## Actigraphy features

Group identifier: `actigraphy`

| Feature           | Description                                      |
|-------------------|--------------------------------------------------|
| `activity_counts` | Activity-count value for each feature-extraction epoch |

[`read_mesa()`][sleepecg.read_mesa] can load activity counts calculated by Philips
Actiware. For other datasets, externally calculated counts can be passed directly to
[`SleepRecord`][sleepecg.SleepRecord]:

```python
import numpy as np
from agcounts.extract import get_counts

from sleepecg import SleepRecord, extract_features

# raw_xyz has shape (n_samples, 3)
axis_counts = get_counts(raw_xyz, freq=sampling_frequency, epoch=30)
activity_counts = np.linalg.norm(axis_counts, axis=1)

record = SleepRecord(
    sleep_stages=sleep_stages,
    sleep_stage_duration=30,
    activity_counts=activity_counts,
)
features, stages, feature_ids = extract_features(
    [record],
    feature_selection=["actigraphy"],
)
```

[`agcounts`](https://github.com/actigraph/agcounts) is an external package and is not
installed with SleepECG.

!!! warning

    Activity counts are specific to the device and calculation method. For example,
    ActiGraph counts calculated by `agcounts` are not equivalent to the proprietary Philips
    Actiwatch values provided with MESA. Use the same method for training and prediction,
    and make sure that the count epoch matches the `sleep_stage_duration` passed to
    `extract_features()`.

The classifiers bundled with SleepECG do not use actigraphy. Using activity counts for
staging therefore requires a classifier trained with `activity_counts` in its feature
selection.
