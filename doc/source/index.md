```{toctree}
   :hidden:

   Home <self>
   API <api>
   Datasets <datasets>
   Heartbeat Detection <heartbeat_detection>
   Feature Extraction <feature_extraction>
   Configuration <configuration>
```

# SleepECG
Welcome! This is the documentation for version **{{ version }}** of SleepECG, a toolbox for sleep stage classification using ECG data.

SleepECG is available on PyPI and can be installed with [pip](https://pip.pypa.io/en/stable/):
```
pip install sleepecg
```
Alternatively, install via [conda](https://docs.conda.io/en/latest/):
```
conda install -c conda-forge sleepecg
```

The [**changelog**](https://github.com/cbrnr/sleepecg/blob/main/CHANGELOG.md) and a [**contributing guide**](https://github.com/cbrnr/sleepecg/blob/main/CONTRIBUTING.md) are available in the [GitHub repo](https://github.com/cbrnr/sleepecg).

[**API**](./api) has detailed information about all public functions and classes in SleepECG.

[**Datasets**](./datasets) shows all avaiable datasets and instructions about retrieving NSRR data.

[**Heartbeat Detection**](./heartbeat_detection) demonstrates how to use the included heartbeat detector and shows benchmark results.

[**Feature Extraction**](./feature_extraction) lists the implemented HRV features.

[**Configuration**](./configuration) explains the possible configuration settings in SleepECG.
