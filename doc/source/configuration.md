(configuration)=
# Configuration
SleepECG provides an interface to set and persist global configuration values. See
[get_config](./generated/sleepecg.get_config) and [set_config](./generated/sleepecg.set_config) for usage instructions.

This table lists the possible configuration settings and where they are used:

|Key|Default value|Description|Used in|
|-|-|-|-|
|`data_dir`|`'~/.sleepecg/datasets'`|Used as the default location to store the files downloaded by all reader functions. Data downloaded for tests is also stored there.|`read_ltdb`, `read_mitdb`, `read_gudb`, `read_mesa`, `read_shhs`, `read_slpdb`|
