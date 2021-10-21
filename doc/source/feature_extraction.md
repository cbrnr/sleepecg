# Feature extraction

## Heart rate variability (HRV) features
See https://doi.org/10.1161/01.CIR.93.5.1043 for standards of HRV measurement and interpretation.
### Time domain
Implemented in `sleepecg.feature_extraction._hrv_timedomain_features`.
All time domain HRV features are either derived from the normal-to-normal (NN) intervals or from the successive differences between NN intervals (SD).

|Feature identifier|Description|Signal|
|-|-|-|
|`meanNN`|average normal-normal (NN) interval|NN|
|`maxNN`|maximum NN interval|NN|
|`minNN`|minimum NN interval|NN|
|`rangeNN`|difference between the longest and shortest NN interval|NN|
|`SDNN`|standard deviation of the NN intervals|NN|
|`RMSSD`|square root of the mean of the sum of the squares of differences between adjacent NN intervals|SD|
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
|`meanHR`|average heart rate ($HR=60/NN$)|NN|
|`maxHR`|maximum heart rate|NN|
|`minHR`|minimum heart rate|NN|
|`stdHR`|standard deviation of the heart rate|NN|

### Frequency domain
Implemented in `sleepecg.feature_extraction._hrv_frequencydomain_features`.
For calculating the frequency domain HRV features, the RR time series is resampled at regular intervals, after which the power spectral density (PSD) is estimated using [Welch's method](https://en.wikipedia.org/wiki/Welch%27s_method).

|Feature identifier|Description|Frequency range|
|-|-|-|
|`total_power`|variance of NN intervals over the temporal segment|$\leq 0.4Hz$|
|`VLF`|power in very low frequency (VLF) range|$[0.0033,0.04)Hz$|
|`LF`|power in low frequency (LF) range|$[0.04,0.15)Hz$|
|`HF`|power in high frequency (HF) range|$[0.15,0.4)Hz$|
|`LF_norm`|LF power in normalized units ($\frac{LF}{LF+HF}\cdot100$)|$[0.04,0.15)Hz$|
|`HF_norm`|HF power in normalized units ($\frac{HF}{LF+HF}\cdot100$)|$[0.15,0.4)Hz$|
|`LF_HF_ratio`|ratio of LF:HF|
