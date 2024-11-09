# © SleepECG developers
#
# License: BSD (3-clause)

"""Tests for the configuration interface."""

import pytest

import sleepecg.config
from sleepecg import get_config, get_config_value, set_config


def test_get_set_del_config():
    """Test setting, getting and deleting a setting."""
    assert get_config_value("data_dir") == "~/.sleepecg/datasets"
    assert get_config_value("classifiers_dir") == "~/.sleepecg/classifiers"
    set_config(data_dir="some_dir")
    assert get_config_value("data_dir") == "some_dir"
    assert get_config_value("classifiers_dir") == "~/.sleepecg/classifiers"
    set_config(data_dir=None)
    assert get_config_value("data_dir") == "~/.sleepecg/datasets"
    assert get_config_value("classifiers_dir") == "~/.sleepecg/classifiers"


def test_set_invalid_config():
    """Test trying to set an invalid configuration key."""
    with pytest.raises(ValueError, match="Trying to set invalid config key:"):
        set_config(invalid_key="some_value")


def test_get_invalid_config():
    """Test trying to get an invalid configuration key."""
    with pytest.raises(ValueError, match="Trying to get invalid config key:"):
        get_config_value("invalid_key")


def test_get_all_config():
    """Test trying to get all configuration values as a dict."""
    all_config = get_config()
    assert isinstance(all_config, dict)
    with pytest.raises(TypeError, match="takes 0 positional arguments"):
        get_config("key")  # type: ignore


def test_invalid_key_in_user_config_file():
    """Test writing an invalid key to the user config file."""
    with open(sleepecg.config._USER_CONFIG_PATH, "w") as user_config_file:
        user_config_file.write("invalid_key: 3")

    with pytest.raises(ValueError, match=r"Invalid key\(s\) found .* {'invalid_key'}"):
        get_config_value("data_dir")
