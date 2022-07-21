```{toctree}
   :hidden:

   Home <self>
   Datasets <datasets>
   Heartbeat Detection <heartbeat_detection>
   Feature Extraction <feature_extraction>
   Classification <classification>
   Configuration <configuration>
   API <api>
   About <about>
```

# SleepECG
SleepECG provides tools for sleep stage classification when [EEG](https://en.wikipedia.org/wiki/Electroencephalography) signals are not available. Based only on [ECG](https://en.wikipedia.org/wiki/Electrocardiography) (and to a lesser extent also movement data), SleepECG provides functions for
- downloading and reading open polysomnography datasets,
- detecting heartbeats from ECG signals, and
- classifying sleep stages (which includes preprocessing, feature extraction, and classification).

## Overview
- [Datasets](./datasets) shows all available datasets and instructions for retrieving NSRR data.
- [Heartbeat Detection](./heartbeat_detection) demonstrates how to use the included heartbeat detector and shows benchmark results.
- [Feature Extraction](./feature_extraction) lists all implemented HRV features.
- [Classification](./classification) describes the included sleep stage classifiers and how to use the classification API.
- [Configuration](./configuration) explains configuration settings in SleepECG.
- [API](./api) has detailed information about all public functions and classes in SleepECG.

## Changelog
Check out the [changelog](https://github.com/cbrnr/sleepecg/blob/main/CHANGELOG.md) to learn what we added, changed, or fixed.

## Dependencies
SleepECG requires Python ≥ 3.8 and the following packages:
- [numpy](http://www.numpy.org/) ≥ 1.20.0
- [requests](https://requests.readthedocs.io/en/latest/) >= 2.25.0
- [scipy](https://scipy.org/) ≥ 1.7.0
- [tqdm](https://tqdm.github.io/) >= 4.60.0
- [PyYAML](https://pyyaml.org/) >= 5.4.0

Optional dependencies provide additional features:
- [joblib](https://joblib.readthedocs.io/en/latest/) ≥ 1.0.0 (parallelized feature extraction)
- [matplotlib](https://matplotlib.org/) ≥ 3.5.0 (plot hypnograms and confusion matrices)
- [mne](https://mne.tools/stable/index.html) ≥ 1.0.0 (read data from [MESA](https://sleepdata.org/datasets/mesa) and [SHHS](https://sleepdata.org/datasets/shhs))
- [numba](https://numba.pydata.org/) ≥ 0.55.0 (JIT-compiled heartbeat detector)
- [pandas](https://pandas.pydata.org/) ≥ 1.4.0 (read data from [GUDB](https://berndporr.github.io/ECG-GUDB))
- [tensorflow](https://www.tensorflow.org/) ≥ 2.7.0 (sleep stage classification with Keras models)
- [wfdb](https://github.com/MIT-LCP/wfdb-python/) ≥ 3.4.0 (read data from [SLPDB](https://physionet.org/content/slpdb), [MITDB](https://physionet.org/content/mitdb), and [LTDB](https://physionet.org/content/ltdb))

## Installation
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
pip install sleepecg[full]
```

If you want the latest development version, use the following command:

```
pip install git+https://github.com/cbrnr/sleepecg
```

## Contributing
The [contributing guide](https://github.com/cbrnr/sleepecg/blob/main/CONTRIBUTING.md) contains detailed instructions on how to contribute to SleepECG.
