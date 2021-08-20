![Py Version](https://img.shields.io/pypi/pyversions/sleepecg.svg?logo=python&logoColor=white)
[![PyPI Version](https://img.shields.io/pypi/v/sleepecg)](https://pypi.org/project/sleepecg/)
[![Docs](https://readthedocs.org/projects/sleepecg/badge/?version=latest)](https://sleepecg.readthedocs.io/en/latest/generated/sleepecg.html)

# SleepECG
SleepECG provides tools for sleep stage classification when [EEG](https://en.wikipedia.org/wiki/Electroencephalography) signals are not available. Based only on [ECG](https://en.wikipedia.org/wiki/Electrocardiography) (and to a lesser extent also movement data), SleepECG provides functions for
- downloading and reading open polysomnography datasets (*TODO*),
- detecting heartbeats from ECG signals, and
- classifying sleep stages (which includes the complete preprocessing, feature extraction, and classification pipeline) (*TODO*).


## Installation
SleepECG is available on PyPI and can be installed with [pip](https://pip.pypa.io/en/stable/):
```
pip install sleepecg
```


## Heartbeat detection
ECG-based sleep staging heavily relies on heartrate variability. Therefore, a reliable and efficient heartbeat detector is essential. SleepECG provides a detector based on the approach described by [Pan & Tompkins (1985)](https://doi.org/10.1109/TBME.1985.325532). We outsourced performance-critical code to a C extension, which makes the detector substantially faster than other implementations. However, we also provide Numba and pure Python backends (the Numba backend is almost as fast whereas the pure Python implementation is much slower).

### Usage
The function [`detect_heartbeats()`](https://github.com/cbrnr/sleepecg/blob/main/sleepecg/heartbeats.py#L40) finds heartbeats in an unfiltered ECG signal `ecg` with sampling frequency `fs` (in Hz). It returns the indices of all detected heartbeats. A complete example including visualization and performance evaluation is available in [`examples/heartbeat_detection.py`](https://raw.githubusercontent.com/cbrnr/sleepecg/main/examples/heartbeat_detection.py).
```python
from sleepecg import detect_heartbeats

detection = detect_heartbeats(ecg, fs)
```


### Performance evaluation
All code used for performance evaluation is available in [`examples/benchmark`](https://github.com/cbrnr/sleepecg/tree/main/examples/benchmark). The used package versions are listed in [`requirements-benchmark.txt`](https://github.com/cbrnr/sleepecg/blob/main/examples/benchmark/requirements-benchmark.py).

We evaluated detector runtime using slices of different lengths from [LTDB](https://physionet.org/content/ltdb/1.0.0/) records with at least 20 hours duration. Error bars in the plot below correspond to the standard error of the mean. The C backend of our detector is by far the fastest implementation among all tested packages (note that the *y*-axis is logarithmically scaled). Runtime evaluation was performed on an [Intel® Xeon® Prozessor E5-2440 v2](https://ark.intel.com/content/www/us/en/ark/products/75263/intel-xeon-processor-e5-2440-v2-20m-cache-1-90-ghz.html) with 32 GiB RAM. No parallelization was used.

![LTDB runtimes](https://raw.githubusercontent.com/cbrnr/sleepecg/main/img/ltdb_runtime_logscale.svg)

We also evaluated detection performance on all [MITDB](https://physionet.org/content/mitdb/1.0.0/) records. We defined a successful detection if it was within 100ms (i.e. 36 samples) of the corresponding annotation (using a tolerance here is necessary because annotations usually do not coincide with the exact R peak locations). In terms of recall, precision, and F1 score, our detector is among the best heartbeat detectors available.

![MITDB metrics](https://raw.githubusercontent.com/cbrnr/sleepecg/main/img/mitdb_metrics.svg)

For analysis of heartrate variability, detecting the exact location of heartbeats is essential. As a measure of how accurate a detector is, we computed Pearson's correlation coefficient between resampled RRI time series deduced from annotated and detected beat locations from all [GUDB](https://github.com/berndporr/ECG-GUDB) records. Our implementation detects peaks in the bandpass-filtered ECG signal, so it produces stable RRI time series without any post-processing.

![GUDB pearson correlation](https://raw.githubusercontent.com/cbrnr/sleepecg/main/img/gudb_pearson.svg)
