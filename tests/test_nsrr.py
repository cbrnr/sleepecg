# © SleepECG developers
#
# License: BSD (3-clause)

"""Tests for the handling of the nsrr data."""

from unittest.mock import patch

import pytest

import sleepecg.config
import sleepecg.io.nsrr
from sleepecg.io.nsrr import _get_nsrr_url


@pytest.fixture(autouse=True)
def temp_test_config(tmp_path):
    """Create, use, and delete a temporary user config file for testing."""
    user_config_path_backup = sleepecg.config._USER_CONFIG_PATH
    sleepecg.config._USER_CONFIG_PATH = tmp_path / "testconfig.yml"
    yield
    sleepecg.config._USER_CONFIG_PATH = user_config_path_backup


def test_get_nsrr_url_no_nsrr_token_set(monkeypatch):
    """Test the _get_nsrr_url method with no token set."""
    monkeypatch.delenv("nsrr_token", raising=False)

    with pytest.raises(RuntimeError, match="NSRR token not set"):
        _get_nsrr_url("mesa")


def test_get_nsrr_url_env_nsrr_token_set(monkeypatch):
    """Test the _get_nsrr_url method with token set via an environment variable."""
    monkeypatch.setenv("nsrr_token", "token")
    nsrr_url = _get_nsrr_url("mesa")
    assert nsrr_url == "https://sleepdata.org/datasets/mesa/files/a/token/m/sleepecg/"


def _patch_get_config_value_return(key):
    return "token"


def test_get_nsrr_url_config_nsrr_token_set(monkeypatch):
    """Test the _get_nsrr_url method with token set in the config file."""
    monkeypatch.delenv("nsrr_token", raising=False)
    monkeypatch.setattr("sleepecg.get_config_value", _patch_get_config_value_return)
    nsrr_url = _get_nsrr_url("mesa")
    assert nsrr_url == "https://sleepdata.org/datasets/mesa/files/a/token/m/sleepecg/"


@patch("sleepecg.io.nsrr._nsrr_token", "token")
def test_get_nsrr_url_function_nsrr_token_set(monkeypatch):
    """Test the _get_nsrr_url method with token set via the set_nsrr_token function."""
    monkeypatch.delenv("nsrr_token", raising=False)
    nsrr_url = _get_nsrr_url("mesa")
    assert nsrr_url == "https://sleepdata.org/datasets/mesa/files/a/token/m/sleepecg/"
