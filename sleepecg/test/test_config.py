# Authors: Florian Hofer
#
# License: BSD (3-clause)

"""Tests for the configuration interface."""

from pathlib import Path

import pytest

import sleepecg.config
from sleepecg import get_config, set_config

_TESTCONFIG_PATH = '~/.sleepecg/testdata/testconfig.yml'


@pytest.fixture(autouse=True)
def temp_test_config():
    """Create, use and delete a temporary user config file for testing."""
    user_config_path_backup = sleepecg.config._USER_CONFIG_PATH
    sleepecg.config._USER_CONFIG_PATH = Path(_TESTCONFIG_PATH).expanduser()

    yield

    if sleepecg.config._USER_CONFIG_PATH.is_file():
        sleepecg.config._USER_CONFIG_PATH.unlink()
    sleepecg.config._USER_CONFIG_PATH = user_config_path_backup


def test_get_set_del_config():
    """Test setting, getting and deleting a setting."""
    assert get_config('data_dir') == '~/.sleepecg/datasets'
    set_config(data_dir='some_dir')
    assert get_config('data_dir') == 'some_dir'
    set_config(data_dir=None)
    assert get_config('data_dir') == '~/.sleepecg/datasets'


def test_set_invalid_config():
    """Test trying to set an invalid configuration key."""
    with pytest.raises(ValueError):
        set_config(invalid_key='some_value')


def test_get_invalid_config():
    """Test trying to get an invalid configuration key."""
    with pytest.raises(ValueError):
        get_config('invalid_key')


def test_get_all_config():
    """Test trying to get all configuration values as a dict."""
    all_config = get_config()
    assert isinstance(all_config, dict)