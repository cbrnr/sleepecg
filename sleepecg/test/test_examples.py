# Authors: Florian Hofer
#
# License: BSD (3-clause)

"""Tests to make sure examples don't crash."""

import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

EXCLUDE = [
    '*/benchmark/*',
]

examples_dir = (Path(__file__).parent / '../../examples').resolve()

files_to_test = []
for pyfile in examples_dir.rglob('*.py'):
    for pattern in EXCLUDE:
        if not pyfile.match(pattern):
            files_to_test.append(pyfile)


@pytest.mark.parametrize('script', files_to_test)
def test_example(script, monkeypatch):
    """Run all examples to make sure they don't crash."""
    # Keep matplotlib from showing figures
    monkeypatch.setattr(plt, 'show', lambda: None)

    runpy.run_path(script)
