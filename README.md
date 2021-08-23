![Py Version](https://img.shields.io/pypi/pyversions/sleepecg.svg?logo=python&logoColor=white)
[![PyPI Version](https://img.shields.io/pypi/v/sleepecg)](https://pypi.org/project/sleepecg/)
[![conda-forge version](https://img.shields.io/conda/v/conda-forge/sleepecg.svg?label=conda-forge)](https://anaconda.org/conda-forge/sleepecg)
[![Docs](https://readthedocs.org/projects/sleepecg/badge/?version=latest)](https://sleepecg.readthedocs.io/en/latest/generated/sleepecg.html)

# SleepECG
SleepECG provides tools for sleep stage classification when [EEG](https://en.wikipedia.org/wiki/Electroencephalography) signals are not available. Based only on [ECG](https://en.wikipedia.org/wiki/Electrocardiography) (and to a lesser extent also movement data), SleepECG provides a functions for
- downloading and reading open polysomnography datasets (*TODO*),
- detecting heartbeats from ECG signals, and
- classifying sleep stages (which includes the complete preprocessing, feature extraction, and classification pipeline) (*TODO*).


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


## Dataset reading
SleepECG provides a consistent functional interface for downloading and reading common polysomnography datasets. While reader functions are a [WIP](https://github.com/cbrnr/sleepecg/pull/28), SleepECG already provides an interface for downloading datasets from the _National Sleep Research Resource_ (NSRR) on [sleepdata.org](https://sleepdata.org/), which replicates the functionality of the [NSRR Ruby Gem](https://github.com/nsrr/nsrr-gem).

The example below downloads all files within [`mesa/polysomnography/edfs`](https://sleepdata.org/datasets/mesa/files/polysomnography/edfs) matching `*-00*` to a local folder `./datasets`:
```python
from sleepecg.io import download_nsrr, set_nsrr_token

set_nsrr_token('<your-download-token-here>')
download_nsrr(
    db_slug='mesa',
    subfolder='polysomnography/edfs',
    pattern='*-00*',
    data_dir='./datasets',
)
```

### ECG dataset readers
To facilitate evaluation of heartbeat detector performance, reader functions for the following annotated ECG datasets are provided:
- [GUDB](https://github.com/berndporr/ECG-GUDB): [`sleepecg.io.read_gudb`](https://sleepecg.readthedocs.io/en/stable/generated/sleepecg.io.ecg_readers.read_gudb.html)
- [MITDB](https://physionet.org/content/mitdb/1.0.0/) and [LTDB](https://physionet.org/content/ltdb/1.0.0/): [`sleepecg.io.read_mitbih`](https://sleepecg.readthedocs.io/en/stable/generated/sleepecg.io.ecg_readers.read_mitbih.html)


## Heartbeat detection
ECG-based sleep staging heavily relies on heartrate variability. Therefore, a reliable and efficient heartbeat detector is essential. SleepECG provides a detector based on the approach described by [Pan & Tompkins (1985)](https://doi.org/10.1109/TBME.1985.325532). We outsourced performance-critical code to a C extension, which makes the detector substantially faster than other implementations. However, we also provide Numba and pure Python backends (the Numba backend is almost as fast whereas the pure Python implementation is much slower).

### Usage
The function [`detect_heartbeats()`](https://github.com/cbrnr/sleepecg/blob/main/sleepecg/heartbeats.py#L40) finds heartbeats in an unfiltered ECG signal `ecg` with sampling frequency `fs` (in Hz). It returns the indices of all detected heartbeats. A complete example including visualization and performance evaluation is available in [`examples/heartbeat_detection.py`](https://raw.githubusercontent.com/cbrnr/sleepecg/main/examples/heartbeat_detection.py).
```python
from sleepecg import detect_heartbeats

detection = detect_heartbeats(ecg, fs)
```


### Performance evaluation
We evaluated detector runtime using slices of different lengths from [LTDB](https://physionet.org/content/ltdb/1.0.0/) records with at least 20 hours duration. Error bars in the plot below correspond to the standard error of the mean. Our detector (we only show the C backend) is by far the fastest implementation among all tested packages (note that the *y*-axis is logarithmically scaled).

![LTDB runtimes](https://raw.githubusercontent.com/cbrnr/sleepecg/main/img/ltdb_runtime_logscale.svg)

We also evaluated detection performance on all [MITDB](https://physionet.org/content/mitdb/1.0.0/) records. We defined a successful detection if it was within 100ms (i.e. 36 samples) of the corresponding annotation (using a tolerance here is necessary because annotations usually do not coincide with the exact R peak locations). In terms of recall, precision, and F1 score, our detector is among the best heartbeat detectors available.

![MITDB metrics](https://raw.githubusercontent.com/cbrnr/sleepecg/main/img/mitdb_metrics.svg)

For analysis of heartrate variability, detecting the exact location of heartbeats is essential. As a measure of how accurate a detector is, we computed Pearson's correlation coefficient between resampled RRI time series deduced from annotated and detected beat locations from all [GUDB](https://github.com/berndporr/ECG-GUDB) records. Our implementation detects peaks in the bandpass-filtered ECG signal, so it produces stable RRI time series without any post-processing.

![GUDB pearson correlation](https://raw.githubusercontent.com/cbrnr/sleepecg/main/img/gudb_pearson.svg)


We used the following detectors for our benchmarks:
```python
# mne
import mne  # https://pypi.org/project/mne/
detection = mne.preprocessing.ecg.qrs_detector(fs, ecg, verbose=False)

# wfdb_xqrs
import wfdb.processing  # https://pypi.org/project/wfdb/
detection = wfdb.processing.xqrs_detect(ecg, fs, verbose=False)

# pyecg_pan_tompkins
import ecgdetectors  # https://pypi.org/project/py-ecg-detectors/
detection = ecgdetectors.Detectors(fs).pan_tompkins_detector(ecg)

# biosppy_hamilton
import biosppy  # https://pypi.org/project/biosppy/
detection = biosppy.signals.ecg.hamilton_segmenter(ecg, fs)[0]

# heartpy
import heartpy  # https://pypi.org/project/heartpy/
wd, m = heartpy.process(ecg, fs)
detection = np.array(wd['peaklist'])[wd['binary_peaklist'].astype(bool)]

# neurokit2_nk
import neurokit2  # https://pypi.org/project/neurokit2/
clean_ecg = neurokit2.ecg.ecg_clean(ecg, int(fs), method='neurokit')
detection = neurokit2.ecg.ecg_findpeaks(clean_ecg, int(fs), method='neurokit')['ECG_R_Peaks']

# neurokit2_kalidas2017
import neurokit2  # https://pypi.org/project/neurokit2/
clean_ecg = neurokit2.ecg.ecg_clean(ecg, int(fs), method='kalidas2017')
detection = neurokit2.ecg.ecg_findpeaks(clean_ecg, int(fs), method='kalidas2017')['ECG_R_Peaks']

# sleepecg
import sleepecg  # https://pypi.org/project/sleepecg/
detection = sleepecg.detect_heartbeats(ecg, fs)
```
