# © SleepECG developers
#
# License: BSD (3-clause)

"""Functions for getting and setting configuration values."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yml"
_USER_CONFIG_PATH = Path("~/.sleepecg/config.yml").expanduser()


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        with open(path) as file:
            cfg = yaml.safe_load(file)
    except FileNotFoundError:
        return {}

    # empty .yml-files are loaded as `None`
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config file at {path}")
    return cfg


def get_config(key: Optional[str] = None) -> Any:
    """
    Read SleepECG preferences from the configuration file.

    For parameters not set in the user configuration file (`~/.sleepecg/config.yml`), this
    falls back to the default values defined in `site-packages/sleepecg/config.yml`. See
    :ref:`configuration` for a list of possible settings.

    Parameters
    ----------
    key : str, optional
        The configuration key to look for. If `None`, all configuration settings are
        returned in a dictionary, by default `None`.

    Returns
    -------
    typing.Any
        The configuration value.
    """
    config = _read_yaml(_DEFAULT_CONFIG_PATH)
    user_config = _read_yaml(_USER_CONFIG_PATH)
    for key in user_config:
        if key not in config:
            raise ValueError(
                f"Invalid key found in user config at {_USER_CONFIG_PATH}: {key}"
            )
    config.update(user_config)

    if key is None:
        return config
    if key not in config:
        options = ", ".join(config)
        raise ValueError(
            f"Trying to get invalid config key: {key!r}, possible options: {options}"
        )
    return config[key]


def set_config(**kwargs):
    """
    Set SleepECG preferences and store them to the user configuration file.

    If a value is `None`, the corresponding key is deleted from the user configuration. See
    :ref:`configuration` for a list of possible settings.

    Parameters
    ----------
    **kwargs: dict, optional
        The configuration keys and values to set.

    Examples
    --------
    >>> set_config(data_dir='~/.sleepecg/datasets')
    """
    default_config = _read_yaml(_DEFAULT_CONFIG_PATH)
    user_config = _read_yaml(_USER_CONFIG_PATH)

    # validate all parameters before setting anything
    for key, value in kwargs.items():
        if key not in default_config:
            options = ", ".join(default_config)
            raise ValueError(
                f"Trying to set invalid config key: {key!r}, possible options: {options}"
            )

    for key, value in kwargs.items():
        if value is None:
            user_config.pop(key, None)
        else:
            user_config[key] = value

    _USER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_USER_CONFIG_PATH, "w") as user_config_file:
        yaml.dump(user_config, user_config_file)
