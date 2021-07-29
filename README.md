![Py Version](https://img.shields.io/pypi/pyversions/sleepecg.svg?logo=python&logoColor=white)
[![PyPI Version](https://img.shields.io/pypi/v/sleepecg)](https://pypi.org/project/sleepecg/)

# SleepECG
This package provides tools for sleep stage classification when [EEG](https://en.wikipedia.org/wiki/Electroencephalography) signals are not available. Based only on [ECG](https://en.wikipedia.org/wiki/Electrocardiography) (and to a lesser extent also movement data), SleepECG provides a functional interface for
- downloading and reading open polysomnography datasets (*TODO*),
- detecting heartbeats from ECG signals, and
- classifying sleep stages (which includes the complete preprocessing, feature extraction, and classification pipeline) (*TODO*).


## Installation
SleepECG is available on PyPI and can be installed with [pip](https://pip.pypa.io/en/stable/):
```
pip install sleepecg
```


## Heartbeat detection
ECG-based sleep staging heavily relies on heartrate variability. Therefore, a reliable and efficient heartbeat detector is essential. SleepECG provides a detector based on the approach described by [Pan & Tompkins (1985)](https://doi.org/10.1109/TBME.1985.325532). We outsourced performance-critical code to a C extension, which leads to substantially faster detections as compared to implementations in pure Python.

### Usage
The function `detect_heartbeats()` finds heartbeats in an unfiltered ECG signal `ecg` (a one-dimensional NumPy array) with sampling frequency `fs` (in Hz). It returns the indices of all detected heartbeats. A complete example including visualization and performance evaluation is available in `examples/heartbeat_detection.py`.
```python
from sleepecg import detect_heartbeats

detection = detect_heartbeats(ecg, fs)
```


### Performance evaluation
We evaluated detector runtime using slices of different lengths from [LTDB](https://physionet.org/content/ltdb/1.0.0/) records with at least 20 hours duration. Error bars in the plot below correspond to the standard error of the mean. Our detector is by far the fastest implementation among all tested packages (note the y-axis is logarithmically scaled).

![LTDB runtimes](https://raw.githubusercontent.com/cbrnr/sleepecg/main/img/ltdb_runtime_logscale.svg)

We also evaluated detection performance on all [MITDB](https://physionet.org/content/mitdb/1.0.0/) records. We defined a successful detection if it was within 100ms (i.e. 36 samples) of the corresponding annotation (using a tolerance here is necessary because annotations usually do not coincide with the exact R peak locations). In terms of recall, precision, and F1 score, our detector is among the best heartbeat detectors available.

![MITDB metrics](https://raw.githubusercontent.com/cbrnr/sleepecg/main/img/mitdb_metrics.svg)

For analysis of heartrate variability, detecting the exact location of heartbeats is essential. As a measure of how accurate a detector is, we computed Pearson's correlation coefficient between resampled RRI time series deduced from annotated and detected beat locations from all [GUDB](https://github.com/berndporr/ECG-GUDB) records. Our implementation detects peaks in the bandpass-filtered ECG signal, so it produces stable RRI time series without any post-processing.

![GUDB pearson correlation](https://raw.githubusercontent.com/cbrnr/sleepecg/main/img/gudb_pearson.svg)


We used the following detectors for our benchmarks:
```python
# mne
import mne  # https://pypi.org/project/mne/
detection = mne.preprocessing.ecg.qrs_detector(fs, ecg, verbose=False)

# wfdb_xqrs
import wfdb.processing  # https://pypi.org/project/wfdb/
detection = wfdb.processing.xqrs_detect(ecg, fs, verbose=False)

# pyecg_pan_tompkins
import ecgdetectors  # https://pypi.org/project/py-ecg-detectors/
detection = ecgdetectors.Detectors(fs).pan_tompkins_detector(ecg)

# biosppy_hamilton
import biosppy  # https://pypi.org/project/biosppy/
detection = biosppy.signals.ecg.hamilton_segmenter(ecg, fs)[0]

# heartpy
import heartpy  # https://pypi.org/project/heartpy/
wd, m = heartpy.process(ecg, fs)
detection = np.array(wd['peaklist'])[wd['binary_peaklist'].astype(bool)]

# neurokit2_nk
import neurokit2  # https://pypi.org/project/neurokit2/
clean_ecg = neurokit2.ecg.ecg_clean(ecg, int(fs), method='neurokit')
peak_indices = neurokit2.ecg.ecg_findpeaks(clean_ecg, int(fs), method='neurokit')['ECG_R_Peaks']

# neurokit2_kalidas2017
import neurokit2  # https://pypi.org/project/neurokit2/
clean_ecg = neurokit2.ecg.ecg_clean(ecg, int(fs), method='kalidas2017')
peak_indices = neurokit2.ecg.ecg_findpeaks(clean_ecg, int(fs), method='kalidas2017')['ECG_R_Peaks']

# sleepecg
import sleepecg  # https://pypi.org/project/sleepecg/
detection = sleepecg.heartbeat_detection.detect_heartbeats(ecg, fs)
```
