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
