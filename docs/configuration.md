# Configuration
SleepECG provides an interface to set and persist global configuration values (see [`get_config()`][sleepecg.get_config] and [`set_config()`][sleepecg.set_config] for usage instructions).

This table lists the possible configuration settings and where they are used:

| Key               | Default value               | Description | Used in |
| ----------------- | --------------------------- | ----------- | ------- |
| `data_dir`        | `'~/.sleepecg/datasets'`    | Default location for storing files downloaded by reader functions, including data downloaded for tests. | [`read_ltdb()`][sleepecg.read_ltdb], [`read_mitdb()`][sleepecg.read_mitdb], [`read_gudb()`][sleepecg.read_gudb], [`read_mesa()`][sleepecg.read_mesa], [`read_shhs()`][sleepecg.read_shhs], [`read_slpdb()`][sleepecg.read_slpdb] |
| `classifiers_dir` | `'~/.sleepecg/classifiers'` | Default location for `SleepClassifier` objects. | [`list_classifiers()`][sleepecg.list_classifiers], [`load_classifier()`][sleepecg.load_classifier], [`save_classifier()`][sleepecg.save_classifier] |
