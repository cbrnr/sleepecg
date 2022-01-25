```{toctree}
   :hidden:

   Home <self>
   Datasets <datasets>
   Heartbeat Detection <heartbeat_detection>
   Feature Extraction <feature_extraction>
   Classification <classification>
   Configuration <configuration>
   API <api>
   GitHub Repository <https://github.com/cbrnr/sleepecg>
```

# SleepECG
Welcome! This is the documentation for version **{{ version }}** of SleepECG, a package for sleep stage classification using ECG data.

## Installation
SleepECG is available on PyPI and can be installed with [pip](https://pip.pypa.io/en/stable/):
```
pip install sleepecg
```
Alternatively, install via [conda](https://docs.conda.io/en/latest/):
```
conda install -c conda-forge sleepecg
```

Optional dependencies provide additional features if installed:
- joblib≥1.0.0 (parallelized feature extraction)
- mne≥0.23.0 (read data from [MESA](https://sleepdata.org/datasets/mesa), [SHHS](https://sleepdata.org/datasets/shhs))
- numba≥0.53.0 (JIT-compiled heartbeat detector)
- pandas≥1.2.0 (read data from [GUDB](https://berndporr.github.io/ECG-GUDB))
- wfdb≥3.3.0 (read data from [SLPDB](https://physionet.org/content/slpdb), [MITDB](https://physionet.org/content/mitdb), [LTDB](https://physionet.org/content/ltdb))

All optional dependencies can be installed with
```
pip install sleepecg[full]
```

## Documentation overview
The [**changelog**](https://github.com/cbrnr/sleepecg/blob/main/CHANGELOG.md) and a [**contributing guide**](https://github.com/cbrnr/sleepecg/blob/main/CONTRIBUTING.md) are available in the [GitHub repo](https://github.com/cbrnr/sleepecg).

[**Datasets**](./datasets) shows all avaiable datasets and instructions about retrieving NSRR data.

[**Heartbeat Detection**](./heartbeat_detection) demonstrates how to use the included heartbeat detector and shows benchmark results.

[**Feature Extraction**](./feature_extraction) lists the implemented HRV features.

[**Classification**](./classification) describes the included sleep stage classifiers and how to use the classification API.

[**Configuration**](./configuration) explains the possible configuration settings in SleepECG.

[**API**](./api) has detailed information about all public functions and classes in SleepECG.
