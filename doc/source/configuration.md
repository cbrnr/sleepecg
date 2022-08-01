(configuration)=
# Configuration
SleepECG provides an interface to set and persist global configuration values. See
{func}`sleepecg.get_config` and {func}`sleepecg.set_config` for usage instructions.

This table lists the possible configuration settings and where they are used:

|Key|Default value|Description|Used in|
|-|-|-|-|
|`data_dir`|`'~/.sleepecg/datasets'`|Used as the default location to store files downloaded by reader functions. Data downloaded for tests is also stored there.|`read_ltdb`, `read_mitdb`, `read_gudb`, `read_mesa`, `read_shhs`, `read_slpdb`|
|`classifiers_dir`|`'~/.sleepecg/classifiers'`|Used as the default location to save and load `SleepClassifier` objects.|`list_classifiers`, `load_classifier`, `save_classifier`|
