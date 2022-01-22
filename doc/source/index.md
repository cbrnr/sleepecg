```{toctree}
   :hidden:

   Home <self>
   Datasets <datasets>
   Heartbeat Detection <heartbeat_detection>
   Feature Extraction <feature_extraction>
   Configuration <configuration>
   API <api>
   GitHub Repository <https://github.com/cbrnr/sleepecg>
```

# SleepECG
Welcome! This is the documentation for version **{{ version }}** of SleepECG, a package for sleep stage classification using ECG data.

SleepECG is available on PyPI and can be installed with [pip](https://pip.pypa.io/en/stable/):
```
pip install sleepecg
```
Alternatively, install via [conda](https://docs.conda.io/en/latest/):
```
conda install -c conda-forge sleepecg
```

The [**changelog**](https://github.com/cbrnr/sleepecg/blob/main/CHANGELOG.md) and a [**contributing guide**](https://github.com/cbrnr/sleepecg/blob/main/CONTRIBUTING.md) are available in the [GitHub repo](https://github.com/cbrnr/sleepecg).

[**Datasets**](./datasets) shows all avaiable datasets and instructions about retrieving NSRR data.

[**Heartbeat Detection**](./heartbeat_detection) demonstrates how to use the included heartbeat detector and shows benchmark results.

[**Feature Extraction**](./feature_extraction) lists the implemented HRV features.

[**Configuration**](./configuration) explains the possible configuration settings in SleepECG.

[**API**](./api) has detailed information about all public functions and classes in SleepECG.
