![Py Version](https://img.shields.io/pypi/pyversions/sleepecg.svg?logo=python&logoColor=white)
[![PyPI Version](https://img.shields.io/pypi/v/sleepecg)](https://pypi.org/project/sleepecg/)
[![conda-forge version](https://img.shields.io/conda/v/conda-forge/sleepecg.svg?label=conda-forge)](https://anaconda.org/conda-forge/sleepecg)
[![Docs](https://readthedocs.org/projects/sleepecg/badge/?version=latest)](https://sleepecg.readthedocs.io/en/stable/index.html)

# SleepECG
SleepECG provides tools for sleep stage classification when [EEG](https://en.wikipedia.org/wiki/Electroencephalography) signals are not available. Based only on [ECG](https://en.wikipedia.org/wiki/Electrocardiography) (and to a lesser extent also movement data), SleepECG provides functions for
- downloading and reading open polysomnography datasets,
- detecting heartbeats from ECG signals, and
- classifying sleep stages (which includes the complete preprocessing, feature extraction, and classification pipeline) (*TODO*).


## Documentation
Documentation for SleepECG is available on [Read the Docs](https://sleepecg.readthedocs.io/en/stable/index.html).


## Installation
SleepECG is available on PyPI and can be installed with [pip](https://pip.pypa.io/en/stable/):
```
pip install sleepecg
```
Alternatively, install via [conda](https://docs.conda.io/en/latest/):
```
conda install -c conda-forge sleepecg
```


## Contributing
The [contributing guide](https://github.com/cbrnr/sleepecg/blob/main/CONTRIBUTING.md) contains detailed instructions on how to contribute to SleepECG.
