# © SleepECG developers
#
# License: BSD (3-clause)

"""Tests to make sure examples don't crash."""

import fnmatch
import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

EXCLUDE = [
    "*/benchmark/*",
    "*/classifiers/*",
    "*/try_ws_gru_mesa.py",
]

examples_dir = (Path(__file__).parent / "../../examples").resolve()

example_files = {str(f) for f in examples_dir.rglob("*.py")}
for pattern in EXCLUDE:
    example_files -= set(fnmatch.filter(example_files, pattern))


@pytest.mark.parametrize("script", example_files)
def test_example(script, monkeypatch):
    """Run all examples to make sure they don't crash."""
    # Keep matplotlib from showing figures
    monkeypatch.setattr(plt, "show", lambda: None)

    runpy.run_path(script)
