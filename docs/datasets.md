# Datasets
SleepECG provides reader functions for various datasets. All required files will be downloaded to the location specified by the `data_dir` argument (by default `~/.sleepecg/datasets`). While all supported [PhysioNet](https://physionet.org/about/database/) datasets are publicly accessible, all [NSRR](https://sleepdata.org/datasets) datasets require [submitting a data access request](#nsrr-data-access).


## Sleep readers
|Reader|Dataset name|Annotated records|Raw data size|Access|
|-|-|-|-|-|
|[`read_mesa()`][sleepecg.read_mesa]|[Multi-Ethnic Study of Atherosclerosis](https://sleepdata.org/datasets/mesa/)|2056|385 GB|[request](https://sleepdata.org/data/requests/mesa/start)|
|[`read_shhs()`][sleepecg.read_shhs]|[Sleep Heart Health Study](https://sleepdata.org/datasets/shhs/)|8444|356 GB|[request](https://sleepdata.org/data/requests/shhs/start)|
|[`read_slpdb()`][sleepecg.read_slpdb]|[MIT-BIH Polysomnographic Database](https://physionet.org/content/slpdb)|18|632 MB|open|


## ECG readers
|Reader|Dataset name|Records|Signals|Raw data size|
|-|-|-|-|-|
|[`read_gudb()`][sleepecg.read_gudb]|[Glasgow University ECG database ](https://berndporr.github.io/ECG-GUDB/)|335|335|550 MB|
|[`read_ltdb()`][sleepecg.read_ltdb]|[MIT-BIH Long-Term ECG Database](https://physionet.org/content/ltdb)|7|15|205 MB|
|[`read_mitdb()`][sleepecg.read_mitdb]|[MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb)|48|96|98.5 MB|


## NSRR data access
To gain access to a dataset provided by the [NSRR](https://sleepdata.org), complete the following steps:

- Create an account [here](https://sleepdata.org/join).
- To create a data access request, either
    - go to the [datasets overview](https://sleepdata.org/datasets/) and click on "Request Data Access" for the desired dataset on the right side, or
    - while browsing a dataset (e.g. [MESA](https://sleepdata.org/datasets/mesa)), click on "Request Data Access" at the top of the page, or
    - follow the "request" link in [this table](#sleep-readers).
- Fill out the data access request form and wait for approval (you will be notified via email, this can take a few days).
- Once the request is approved, you can
    - download files manually from the "Files" tab on the corresponding dataset page (e.g. [MESA EDFs](https://sleepdata.org/datasets/mesa/files/polysomnography/edfs)) or
    - use your [NSRR token](https://sleepdata.org/token) to download files via the NSRR API. Your token will always stay the same and is valid for all datasets you have been granted access to.

The following code snippet shows how to read all records in the [MESA](https://sleepdata.org/datasets/mesa) dataset with SleepECG:

```python
from sleepecg import read_mesa, set_nsrr_token

set_nsrr_token("<your-download-token-here>")
mesa = read_mesa()  # note that this is a generator
```

You can also select a subset of records from a dataset. This example will download and read all records having IDs starting with `00` (i.e. records `0001`–`0099`):

```python
from sleepecg import read_mesa, set_nsrr_token

set_nsrr_token("<your-download-token-here>")
mesa = read_mesa(records_pattern="00*")  # note that this is a generator
```

!!! note
    Reader functions are generators, so they do not return the data directly. To access the data, you need to consume the generator, either by iterating over it or with subsequent calls of `next()`.

If you just want to download NSRR data (like with the [NSRR Ruby Gem](https://github.com/nsrr/nsrr-gem)), use the workflow below. The example downloads all files within [`mesa/polysomnography/edfs`](https://sleepdata.org/datasets/mesa/files/polysomnography/edfs) matching `*-00*` to a local folder `./datasets` (subfolders are automatically created to preserve the original directory structure).

```python
from sleepecg import download_nsrr, set_nsrr_token

set_nsrr_token("<your-download-token-here>")
download_nsrr(
    db_slug="mesa",
    subfolder="polysomnography/edfs",
    pattern="*-00*",
    data_dir="./datasets",
)
```
