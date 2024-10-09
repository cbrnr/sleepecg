# SleepECG

SleepECG provides tools for sleep stage classification when [EEG](https://en.wikipedia.org/wiki/Electroencephalography) signals are not available. Based only on [ECG](https://en.wikipedia.org/wiki/Electrocardiography), SleepECG provides functions for

- downloading and reading open polysomnography datasets,
- detecting heartbeats from ECG signals, and
- classifying sleep stages (which includes preprocessing, feature extraction, and classification).


## Documentation

Documentation for SleepECG is available on [Read the Docs](https://sleepecg.readthedocs.io/en/stable/index.html).


## Changelog

Check out the [changelog](https://github.com/cbrnr/sleepecg/blob/main/CHANGELOG.md) to learn what we added, changed, or fixed.


## Installation

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


## Contributing

The [contributing guide](https://github.com/cbrnr/sleepecg/blob/main/CONTRIBUTING.md) contains detailed instructions on how to contribute to SleepECG.
