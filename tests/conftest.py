"""Configuration for tests."""

import pytest

import sleepecg.config


@pytest.fixture(autouse=True)
def temp_test_config(tmp_path):
    """Create, use, and delete a temporary user config file for testing."""
    user_config_path_backup = sleepecg.config._USER_CONFIG_PATH
    sleepecg.config._USER_CONFIG_PATH = tmp_path / "testconfig.yml"
    yield
    sleepecg.config._USER_CONFIG_PATH = user_config_path_backup
