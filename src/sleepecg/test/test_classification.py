# © SleepECG developers
#
# License: BSD (3-clause)

"""Tests for functions related to classifier training and evaluation."""

import numpy as np
import pytest

from sleepecg.classification import _merge_sleep_stages


@pytest.mark.parametrize(
    ["mode", "output"],
    [
        ("wake-sleep", [0, 1, 1, 1, 1, 2]),
        ("wake-rem-nrem", [0, 1, 1, 1, 2, 3]),
        ("wake-rem-light-n3", [0, 1, 2, 2, 3, 4]),
        ("wake-rem-n1-n2-n3", [0, 1, 2, 3, 4, 5]),
    ],
)
def test_merge_sleep_stages(mode, output):
    """Test if the sleep stage mapping works correctly."""
    stages = [np.array([0, 1, 2, 3, 4, 5])]
    assert (_merge_sleep_stages(stages, mode)[0] == np.array(output)).all()
