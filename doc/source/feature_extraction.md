(feature_extraction)=
# Feature extraction

## Heart rate variability features
Features are based on standards of heart rate variability (HRV) measurement and interpretation described in [Task Force of the European Society of Cardiology (1996)](https://doi.org/10.1161/01.CIR.93.5.1043) and [Shaffer & Ginsberg (2017)](https://doi.org/10.3389/fpubh.2017.00258).

### Time domain
Group identifier: `hrv-time`

Implemented in `sleepecg.feature_extraction._hrv_timedomain_features`.
All time domain HRV features are either derived from the normal-to-normal (NN) intervals, from the successive differences between NN intervals (SD), or from the Poincaré plot (PP).

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

Implemented in `sleepecg.feature_extraction._hrv_frequencydomain_features`.
For calculating the frequency domain HRV features, the RR time series is resampled at regular intervals, after which the power spectral density (PSD) is estimated using [Welch's method](https://en.wikipedia.org/wiki/Welch%27s_method).

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
