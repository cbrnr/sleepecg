# © SleepECG developers
#
# License: BSD (3-clause)

"""Tests for the handling of the nsrr data."""

import pytest

import sleepecg.config
import sleepecg.io.nsrr
from sleepecg.io.nsrr import _get_nsrr_url


def test_get_nsrr_url_no_nsrr_token_set(monkeypatch):
    """Test the _get_nsrr_url method with no token set."""
    monkeypatch.delenv("NSRR_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="NSRR token not set"):
        _get_nsrr_url("mesa")


def test_get_nsrr_url_env_nsrr_token_set(monkeypatch):
    """Test the _get_nsrr_url method with token set via an environment variable."""
    monkeypatch.setenv("NSRR_TOKEN", "token")
    nsrr_url = _get_nsrr_url("mesa")
    assert nsrr_url == "https://sleepdata.org/datasets/mesa/files/a/token/m/sleepecg/"


def test_get_nsrr_url_config_nsrr_token_set(monkeypatch):
    """Test the _get_nsrr_url method with token set in the config file."""
    monkeypatch.delenv("NSRR_TOKEN", raising=False)
    sleepecg.config.set_config(nsrr_token="token")
    nsrr_url = _get_nsrr_url("mesa")
    assert nsrr_url == "https://sleepdata.org/datasets/mesa/files/a/token/m/sleepecg/"


def test_get_nsrr_url_function_nsrr_token_set(monkeypatch):
    """Test the _get_nsrr_url method with token set via the set_nsrr_token function."""
    monkeypatch.delenv("NSRR_TOKEN", raising=False)
    monkeypatch.setattr("sleepecg.io.nsrr._nsrr_token", "token")
    nsrr_url = _get_nsrr_url("mesa")
    assert nsrr_url == "https://sleepdata.org/datasets/mesa/files/a/token/m/sleepecg/"
