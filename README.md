![Python](https://img.shields.io/pypi/pyversions/sleepecg.svg?logo=python&logoColor=white)
[![PyPI](https://img.shields.io/pypi/v/sleepecg)](https://pypi.org/project/sleepecg/)
[![conda-forge](https://img.shields.io/conda/v/conda-forge/sleepecg.svg?label=conda-forge)](https://anaconda.org/conda-forge/sleepecg)
[![Docs](https://readthedocs.org/projects/sleepecg/badge/?version=latest)](https://sleepecg.readthedocs.io/en/stable/index.html)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.05411/status.svg)](https://doi.org/10.21105/joss.05411)
[![License](https://img.shields.io/github/license/cbrnr/sleepecg)](LICENSE)

## SleepECG

SleepECG provides tools for sleep stage classification when [EEG](https://en.wikipedia.org/wiki/Electroencephalography) signals are not available. Based only on [ECG](https://en.wikipedia.org/wiki/Electrocardiography), SleepECG provides functions for

- downloading and reading open polysomnography datasets,
- detecting heartbeats from ECG signals, and
- classifying sleep stages (which includes preprocessing, feature extraction, and classification).

## HeartRate Data
Description:
I have introduced Changes to the SleepECG package, primarily focusing on enabling sleep stage classification using heart rate data when EEG signals are not available. Key features of this update include:

Heart Rate to R-Interval Conversion:

Implemented an algorithm to convert heart rate data into R-intervals. This crucial development allows the use of heart rate data, which is more readily available in many scenarios, for sleep stage classification.
Handling Data Gaps:

Added robustness to the system by integrating code that effectively handles scenarios where data might be missing for extended periods. This ensures more reliable performance and resilience in real-world applications.
Web-Based and Local Application:

Originally developed for a web-based project, these enhancements are equally applicable for local deployments, thereby broadening the usability scope of the SleepECG package.
### Documentation

Documentation for SleepECG is available on [Read the Docs](https://sleepecg.readthedocs.io/en/stable/index.html). Check out the [changelog](https://github.com/cbrnr/sleepecg/blob/main/CHANGELOG.md) to learn what we added, changed, or fixed.


### Installation

SleepECG is available on PyPI and can be installed with [pip](https://pip.pypa.io/en/stable/):

```
pip install sleepecg
```

Alternatively, an unofficial [conda](https://docs.conda.io/en/latest/) package is available:

```
conda install -c conda-forge sleepecg
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
### Example if Heart Rate data is available
data = [
    {'TimeStamp': '1690409900', 'HeartRate': '55'},
    {'TimeStamp': '1690410000', 'HeartRate': '60'},
    {'TimeStamp': '1690410120', 'HeartRate': '65'},
    {'TimeStamp': '1690410300', 'HeartRate': '62'},
    {'TimeStamp': '1690410600', 'HeartRate': '59'},
    {'TimeStamp': '1690410900', 'HeartRate': '58'},
    {'TimeStamp': '1690411200', 'HeartRate': '61'},
    {'TimeStamp': '1690411320', 'HeartRate': '56'},
    {'TimeStamp': '1690411380', 'HeartRate': '52'},
    {'TimeStamp': '1690411440', 'HeartRate': '50'},
    {'TimeStamp': '1690411500', 'HeartRate': '56'},
    {'TimeStamp': '1690411560', 'HeartRate': '58'},
    {'TimeStamp': '1690411620', 'HeartRate': '60'},
    {'TimeStamp': '1690411680', 'HeartRate': '65'},
    {'TimeStamp': '1690412740', 'HeartRate': '60'},
    {'TimeStamp': '1690412800', 'HeartRate': '62'},
    {'TimeStamp': '1690413800', 'HeartRate': '62'},
    {'TimeStamp': '1690514000', 'HeartRate': '62'}
]
final_rec , combined_stages_pred , stages_mode ,long_interval_indices = sleepanalyse(data)
plot_hypnogram(
    final_rec,
    combined_stages_pred,
    stages_mode=stages_mode )
plt.show()
![output_figure](https://github.com/MashRiza/sleepecg/assets/133785714/f22c2e56-58f2-4f25-84a1-57247ebf33e7)

### Dependencies

SleepECG requires Python ≥ 3.9 and the following packages:

- [numpy](https://numpy.org/) ≥ 1.20.0
- [requests](https://requests.readthedocs.io/en/latest/) ≥ 2.25.0
- [scipy](https://scipy.org/) ≥ 1.7.0
- [tqdm](https://tqdm.github.io/) ≥ 4.60.0
- [PyYAML](https://pyyaml.org/) ≥ 5.4.0

Optional dependencies provide additional features:

- [edfio](https://github.com/the-siesta-group/edfio/) ≥ 0.1.1 (read data from [MESA](https://sleepdata.org/datasets/mesa) and [SHHS](https://sleepdata.org/datasets/shhs))
- [joblib](https://joblib.readthedocs.io/en/latest/) ≥ 1.0.0 (parallelized feature extraction)
- [matplotlib](https://matplotlib.org/) ≥ 3.5.0 (plot ECG time courses, hypnograms, and confusion matrices)
- [numba](https://numba.pydata.org/) ≥ 0.55.0 (JIT-compiled heartbeat detector)
- [tensorflow](https://www.tensorflow.org/) ≥ 2.7.0 (sleep stage classification with Keras models)
- [wfdb](https://github.com/MIT-LCP/wfdb-python/) ≥ 3.4.0 (read data from [SLPDB](https://physionet.org/content/slpdb), [MITDB](https://physionet.org/content/mitdb), and [LTDB](https://physionet.org/content/ltdb))


### Contributing

The [contributing guide](https://github.com/cbrnr/sleepecg/blob/main/CONTRIBUTING.md) contains detailed instructions on how to contribute to SleepECG.
