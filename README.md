![Py Version](https://img.shields.io/pypi/pyversions/sleepecg.svg?logo=python&logoColor=white)
[![PyPI Version](https://img.shields.io/pypi/v/sleepecg)](https://pypi.org/project/sleepecg/)

# sleepecg
This package provides tools for sleep stage classification when [EEG](https://en.wikipedia.org/wiki/Electroencephalography) signals are not available. Based only on [ECG](https://en.wikipedia.org/wiki/Electrocardiography) (and to a lesser extent also movement data), it will feature a functional interface for
- downloading and reading open polysomnography datasets (TODO),
- detecting heartbeats from ECG signals, and
- classifying sleep stages (which includes the complete preprocessing, feature extraction, and classification pipeline) (TODO).


## Installation
You can install sleepecg from PyPI using pip:
```
pip install sleepecg
```


## Heartbeat detection
Since ECG-based sleep staging relies on heartrate variability, a reliable and efficient heartbeat detector is essential. This package provides a detector based on the approach described by [Pan & Tompkins (1985)](https://doi.org/10.1109/TBME.1985.325532). Outsourcing performance-critical code to a C extension (wheels are provided) leads to substantially faster detections as compared to implementations in other Python packages. Benchmarks on [MITDB](https://physionet.org/content/mitdb/1.0.0/), [LTDB](https://physionet.org/content/ltdb/1.0.0/), and [GUDB](https://github.com/berndporr/ECG-GUDB) show that our implementation produces highly reliable detections and runtime scales linearly with signal length or sampling frequency.


### Usage
The function `detect_heartbeats()` finds heartbeats in an unfiltered ECG signal `ecg` with sampling frequency `fs`. It returns a one-dimensional NumPy array containing the indices of the detected heartbeats. A complete example including visualization and performance evaluation is available in `examples/heartbeat_detection.py`.
```python
from sleepecg import detect_heartbeats
detection = detect_heartbeats(ecg, fs)
```


### Performance
We evaluated detector runtime using slices of different lengths from [LTDB](https://physionet.org/content/ltdb/1.0.0/) records which are at least 20 hours long. Error bars in the plot below correspond to the standard errors of the mean.

![LTDB runtimes](https://raw.githubusercontent.com/cbrnr/sleepecg/main/img/ltdb_runtime_logscale.svg)

For the plots below, we evaluated detectors on all [MITDB](https://physionet.org/content/mitdb/1.0.0/) records. We defined a successful detection if a detection and corresponding annotation are within 100ms (i.e. 36 samples). Using a tolerance here is necessary because annotations usually do not coincide with the exact R peak locations. In terms of recall (i.e. sensitivity), our detector is on the same level as BioSPPy's hamilton-detector, NeuroKit's neurokit-detector and WFDB's xqrs. MNE generally finds fewer peaks than other detectors, so there are fewer false positives and higher precision. Comparing F1-scores shows that sleepecg performs as well as other commonly used Python heartbeat detectors.

![MITDB metrics](https://raw.githubusercontent.com/cbrnr/sleepecg/main/img/mitdb_metrics.svg)

For analysis of heartrate variability, detecting the exact location of heartbeats is essential. As a measure of how well a detector can replicate correct RR intervals (RRI), we computed Pearson's correlation coefficient between resampled RRI time series deduced from annotated and detected beat locations from all [GUDB](https://github.com/berndporr/ECG-GUDB) records. In contrast to other databases, GUDB has annotations that are exactly at R peak locations. Our implementation detects peaks in the bandpass-filtered ECG signal, so it is able to produce stable RRI time series without any post-processing. MNE and pan_tompkins_detector from pyecgdetectors often detect an S-peak instead of the R-peak, leading to noisy RR intervals (and thus lower correlation).

![GUDB pearson correlation](https://raw.githubusercontent.com/cbrnr/sleepecg/main/img/gudb_pearson.svg)


We used the following detector calls for all benchmarks:
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
