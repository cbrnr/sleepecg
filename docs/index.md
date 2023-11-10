# SleepECG
SleepECG provides tools for sleep stage classification when [EEG](https://en.wikipedia.org/wiki/Electroencephalography) signals are not available. Based only on [ECG](https://en.wikipedia.org/wiki/Electrocardiography), SleepECG provides functions for

- downloading and reading open polysomnography datasets,
- detecting heartbeats from ECG signals, and
- classifying sleep stages (which includes preprocessing, feature extraction, and classification).

## Documentation
Documentation for SleepECG is available on [Read the Docs](https://sleepecg.readthedocs.io/en/stable/index.html).

## Changelog
Check out the [changelog](https://github.com/cbrnr/sleepecg/blob/main/CHANGELOG.md) to learn what we added, changed, or fixed.

## Dependencies
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
pip install "sleepecg[full]"
```

If you want the latest development version, use the following command:

```
pip install git+https://github.com/cbrnr/sleepecg
```

## Contributing
The [contributing guide](https://github.com/cbrnr/sleepecg/blob/main/CONTRIBUTING.md) contains detailed instructions on how to contribute to SleepECG.
