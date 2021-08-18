# Heartbeat Detection Benchmarks
This example reproduces the benchmarks shown in the main [`README.md`](https://github.com/cbrnr/sleepecg#readme).

## Usage
To run the benchmark, create a virtual enviroment and install the requirements using
```
pip install -r requirements-benchmark.txt
```

and execute
```
python benchmark_detectors.py <benchmark>
```
where `<benchmark>` is a top-level key in `config.yml` (see section [Configuration](#configuration)). Possible values are `runtime`, `metrics`, and `rri_similarity`.

Plots can be created by executing
```
python plot_benchmark_results.py <results.csv>
```
which will save the plot to `results.svg` at the same location as `results.csv`. Plot types and labels for the provided benchmarks are selected based on the filename, so renaming may lead to errors.


## Configuration
A benchmark configuration is specified below a unique top-level key in `config.yml`. Parameters:
|Key|Type|Default|Description|
|---|----|--------|-----------|
|data_dir|`str`||Path where all datasets are stored.|
|outfile_dir|`str`||Path where the evaluation results should be stored.|
|db_slug|`str`||Which dataset to use for the evaluation. Possible values: [`mitdb`](https://physionet.org/content/mitdb/1.0.0/), [`ltdb`](https://physionet.org/content/mitdb/1.0.0/), [`gudb`](https://physionet.org/content/mitdb/1.0.0/).|
|detectors|`list[str]`||Detectors to be evaluated. For possible options, see [`utils._detector_dispatch`](https://github.com/cbrnr/sleepecg/blob/main/examples/benchmark/benchmark_detectors.py).|
|signal_lengths|`list[int]`||Length in minutes to which each ECG signal should be sliced. If a signal is too short, it is skipped.|
|max_distance|`float`|`0.1`|Maximum temporal distance in seconds between detected and annotated beats to count as a successful detection.|
|suppress_warnings|`bool`|`False`|Whether to suppress warnings during detector execution.|
|timeout|`int`|`600`|Number of seconds after which to attempt cancelling execution using SIGALRM (only on Unix).|
|max_timeouts|`int`|`3`|Number of timeouts after which a detector is skipped completely.|
|calc_rri_similarity|`bool`|`False`|Whether to calculate similarity measures between detected and annotated RR intervals (computationally expensive for long signals).|

## Known Issues
- Cancelling the `heartpy` detector for ltdb:15814:ECG2 doesn't work for `signal_lengths` larger than some value between 600 and 900 minutes. Program execution has to be terminated manually.
