![Python](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fcbrnr%2Fsleepecg%2Frefs%2Fheads%2Fmain%2Fpyproject.toml&logo=python&logoColor=white)
[![PyPI](https://img.shields.io/pypi/v/sleepecg)](https://pypi.org/project/sleepecg/)
[![Docs](https://readthedocs.org/projects/sleepecg/badge/?version=latest)](https://sleepecg.readthedocs.io/en/stable/index.html)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.05411/status.svg)](https://doi.org/10.21105/joss.05411)
[![License](https://img.shields.io/github/license/cbrnr/sleepecg)](LICENSE)

## SleepECG

SleepECG provides tools for sleep stage classification when [EEG](https://en.wikipedia.org/wiki/Electroencephalography) signals are not available. Based only on [ECG](https://en.wikipedia.org/wiki/Electrocardiography), SleepECG provides functions for

- downloading and reading open polysomnography datasets,
- detecting heartbeats from ECG signals, and
- classifying sleep stages (which includes preprocessing, feature extraction, and classification).


### Documentation

Documentation for SleepECG is available on [Read the Docs](https://sleepecg.readthedocs.io/en/stable/index.html). Check out the [changelog](https://github.com/cbrnr/sleepecg/blob/main/CHANGELOG.md) to learn what we added, changed, or fixed.


### Installation

SleepECG is available on PyPI and can be installed with [pip](https://pip.pypa.io/en/stable/):

```
pip install sleepecg
```

SleepECG with all optional dependencies can be installed with the following command:

```
pip install "sleepecg[full]"
```

If you want the latest development version, use the following command:

```
pip install git+https://github.com/cbrnr/sleepecg
```


### Example

The following example detects heartbeats in a short ECG (a one-dimensional NumPy array):

```python
import numpy as np
from sleepecg import detect_heartbeats, get_toy_ecg

ecg, fs = get_toy_ecg()  # 5 min of ECG data at 360 Hz
beats = detect_heartbeats(ecg, fs)  # indices of detected heartbeats
```


### Contributing

The [contributing guide](https://github.com/cbrnr/sleepecg/blob/main/CONTRIBUTING.md) contains detailed instructions on how to contribute to SleepECG.
