## [0.5.5] - 2023-06-01
### Changed
- Use absolute imports internally ([#170](https://github.com/cbrnr/sleepecg/pull/170) by [Clemens Brunner](https://github.com/cbrnr))

## [0.5.4] - 2023-04-13
### Changed
- Remove dependency on pandas for `read_gudb` ([#134](https://github.com/cbrnr/sleepecg/pull/134) by [Florian Hofer](https://github.com/hofaflo))
- Update SHHS encoding and missing data handling for `read_shhs` ([#139](https://github.com/cbrnr/sleepecg/pull/139) by [Andrew Gilbert](https://github.com/adgilbert))
- Ignore expected warnings during feature extraction ([#142](https://github.com/cbrnr/sleepecg/pull/142) by [Florian Hofer](https://github.com/hofaflo))

## [0.5.3] - 2022-12-05
### Added
- Do not redownload GUDB data sets if they are already stored locally ([#112](https://github.com/cbrnr/sleepecg/pull/112) by [Clemens Brunner](https://github.com/cbrnr))
- Add `sleepecg.plot_ecg` and `sleepecg.ECGRecord.plot` functions for plotting ECG with beat annotations ([#116](https://github.com/cbrnr/sleepecg/pull/116) by [Clemens Brunner](https://github.com/cbrnr))

### Changed
- The detector benchmark example now requires only those packages that are actually used in the benchmark ([#114](https://github.com/cbrnr/sleepecg/pull/114) by [Clemens Brunner](https://github.com/cbrnr))
- The Pan-Tompkins detector detects flat data at the beginning of a recording to avoid messing up its thresholds ([#87](https://github.com/cbrnr/sleepecg/pull/87) by [Raphael Vallat](https://github.com/raphaelvallat))
- Switch documentation from Sphinx to MkDocs ([#119](https://github.com/cbrnr/sleepecg/pull/119) by [Clemens Brunner](https://github.com/cbrnr))
- Remove unnecessary detection of first zero-crossing in Pan-Tompkins detector ([#120](https://github.com/cbrnr/sleepecg/pull/120) by [Clemens Brunner](https://github.com/cbrnr))
- Move `data_dir` parameter in `download_physionet` to last position ([#129](https://github.com/cbrnr/sleepecg/pull/129) by [Clemens Brunner](https://github.com/cbrnr))

### Fixed
- Fix bug where setting user config values lets `get_config` retrieve the wrong key ([#123](https://github.com/cbrnr/sleepecg/pull/123) by [Florian Hofer](https://github.com/hofaflo))

## [0.5.2] - 2022-08-02
### Fixed
- Include classifiers and tests ([#104](https://github.com/cbrnr/sleepecg/pull/104) by [Clemens Brunner](https://github.com/cbrnr))

## [0.5.1] - 2022-07-21
### Added
- Add more tests and documentation for the Pan-Tompkins detector ([#89](https://github.com/cbrnr/sleepecg/pull/89) by [Raphael Vallat](https://github.com/raphaelvallat))

### Changed
- The `preprocess_rri` function no longer operates in-place and instead returns a copy of the input array ([#91](https://github.com/cbrnr/sleepecg/pull/91) by [Raphael Vallat](https://github.com/raphaelvallat))

## [0.5.0] - 2022-03-22
### Added
- Add metadata feature extraction ([#70](https://github.com/cbrnr/sleepecg/pull/70) by [Florian Hofer](https://github.com/hofaflo))
- Add ability to read annotated heartbeats and subject data from SHHS ([#72](https://github.com/cbrnr/sleepecg/pull/72) by [Florian Hofer](https://github.com/hofaflo))
- Add function `export_ecg_record` to export ECG records to text files ([#71](https://github.com/cbrnr/sleepecg/pull/71) by [Clemens Brunner](https://github.com/cbrnr))
- Add support for Python 3.10 ([#75](https://github.com/cbrnr/sleepecg/pull/75) by [Florian Hofer](https://github.com/hofaflo))
- Add functions for classification ([#78](https://github.com/cbrnr/sleepecg/pull/78) by [Florian Hofer](https://github.com/hofaflo))
- Add a WAKE-SLEEP classifier trained on MESA ([#78](https://github.com/cbrnr/sleepecg/pull/78) by [Florian Hofer](https://github.com/hofaflo))
- Add two WAKE-REM-NREM classifiers trained on MESA ([#84](https://github.com/cbrnr/sleepecg/pull/84) by [Florian Hofer](https://github.com/hofaflo))

## [0.4.1] - 2022-01-14
### Fixed
- Fix `reader_dispatch` in `examples/benchmark/utils.py` not yielding anything  ([#68](https://github.com/cbrnr/sleepecg/pull/68) by [Florian Hofer](https://github.com/hofaflo))

## [0.4.0] - 2022-01-11
### Added
- Configuration via a YAML file ([#39](https://github.com/cbrnr/sleepecg/pull/39) by [Florian Hofer](https://github.com/hofaflo))
- Add function for reading data from the [MESA](https://sleepdata.org/datasets/mesa) datasets ([#28](https://github.com/cbrnr/sleepecg/pull/28) by [Florian Hofer](https://github.com/hofaflo))
- Add heart rate variability (HRV) feature extraction ([#36](https://github.com/cbrnr/sleepecg/pull/36) by [Florian Hofer](https://github.com/hofaflo))
- Add reader function for [SLPDB](https://physionet.org/content/slpdb) ([#47](https://github.com/cbrnr/sleepecg/pull/47)) by [Florian Hofer](https://github.com/hofaflo))
- Add reader function for [SHHS](https://sleepdata.org/datasets/shhs) ([#48](https://github.com/cbrnr/sleepecg/pull/48)) by [Florian Hofer](https://github.com/hofaflo))

### Changed
- Export members of `sleepecg.io` to main package ([#56](https://github.com/cbrnr/sleepecg/pull/56) by [Florian Hofer](https://github.com/hofaflo))
- Reader functions now have `records_pattern` as their first parameter and `data_dir` is the last one ([#65](https://github.com/cbrnr/sleepecg/pull/65) by [Florian Hofer](https://github.com/hofaflo))

## [0.3.0] - 2021-08-25
### Added
- Example containing code used for benchmarks ([#22](https://github.com/cbrnr/sleepecg/pull/22) by [Florian Hofer](https://github.com/hofaflo))

### Changed
- Shorten names of some functions and modules ([#18](https://github.com/cbrnr/sleepecg/pull/18) by [Clemens Brunner](https://github.com/cbrnr) and [Florian Hofer](https://github.com/hofaflo))
- Separate reader functions per database ([#32](https://github.com/cbrnr/sleepecg/pull/32) by [Florian Hofer](https://github.com/hofaflo))
- Use `~/.sleepecg/datasets` as default `data_dir` for readers ([#33](https://github.com/cbrnr/sleepecg/pull/33) by [Florian Hofer](https://github.com/hofaflo))

## [0.2.0] - 2021-08-11
### Added
- Add interface to download NSRR data ([#8](https://github.com/cbrnr/sleepecg/pull/8) by [Florian Hofer](https://github.com/hofaflo))
- Add pure Python and Numba implementations of heartbeat detection ([#10](https://github.com/cbrnr/sleepecg/pull/10) by [Florian Hofer](https://github.com/hofaflo))

### Fixed
- Fix MemoryError in case of invalid checksum ([#3](https://github.com/cbrnr/sleepecg/pull/3) by [Florian Hofer](https://github.com/hofaflo))

## [0.1.0] - 2021-07-29
### Added
- Initial release (by [Florian Hofer](https://github.com/hofaflo) and [Clemens Brunner](https://github.com/cbrnr))
