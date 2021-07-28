import numpy as np
from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            'sleepecg._heartbeat_detection',
            ['sleepecg/_heartbeat_detection.c'],
            include_dirs=[np.get_include()],
        ),
    ],
)
