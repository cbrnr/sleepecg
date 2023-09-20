# Examples

To run the provided examples, download or clone the [GitHub Repository](https://github.com/cbrnr/sleepecg) and execute the scripts in this directory or its subdirectories.

- Heartbeat detection demo:
    ```
    sleepecg/examples> python heartbeat_detection.py
    ```

- Classify sleep stages in your own dataset (starting with ECG recording for the sleep duration stored in `sleep.edf` in this example):
    ```
    sleepecg/examples> python custom_dataset.py
    ```

- Benchmark heartbeat detector runtime (more info [here](https://github.com/cbrnr/sleepecg/tree/main/examples/benchmark)):
    ```
    sleepecg/examples/benchmark> python benchmark_detectors.py runtime
    ```
