import numpy as np
from setuptools import Extension, setup
from wheel.bdist_wheel import bdist_wheel

# Following https://github.com/joerick/python-abi3-package-sample

# Currently set to 3.10. If this is bumped for example to 3.11, the hex in the Extension
# needs to change to 0x030B0000, and the cibuildwheel `build` selector needs to change
# in pyproject.toml.

class bdist_wheel_abi3(bdist_wheel):
    def get_tag(self) -> tuple:
        python, abi, plat = super().get_tag()

        if python.startswith("cp"):
            # on CPython, our wheels are abi3 and compatible back to 3.6
            return "cp310", "abi3", plat

        return python, abi, plat


setup(
    ext_modules=[
        Extension(
            "sleepecg._heartbeat_detection",
            ["src/sleepecg/_heartbeat_detection.c"],
            include_dirs=[np.get_include()],
            define_macros=[("Py_LIMITED_API", "0x030A0000")],
            py_limited_api=True,
        ),
    ],
    cmdclass={"bdist_wheel": bdist_wheel_abi3},
)
