# abi3: the extension is built against the CPython limited API (see
# tool.meson-python.limited-api in pyproject.toml), so a single wheel built
# with the oldest supported CPython (3.11) works for all newer versions too.
numpy_incdir = run_command(
    py,
    ['-c', 'import numpy; print(numpy.get_include())'],
    check: true,
).stdout().strip()

# passed as a raw compiler flag (rather than include_directories()) since
# numpy_incdir may live inside the source tree (e.g. an in-repo .venv), which
# Meson's include_directories() refuses to reference via an absolute path.
py.extension_module(
    '_heartbeat_detection',
    '_heartbeat_detection.c',
    c_args: ['-I' + numpy_incdir],
    limited_api: '3.11',
    install: true,
    subdir: 'sleepecg',
)

py.install_sources(
    '__init__.py',
    'classification.py',
    'config.py',
    'feature_extraction.py',
    'heartbeats.py',
    'plot.py',
    'utils.py',
    subdir: 'sleepecg',
)

install_data(
    '_heartbeat_detection.pyi',
    'config.yml',
    'py.typed',
    install_dir: py.get_install_dir(subdir: 'sleepecg'),
)

install_data(
    'data/ecg.npz',
    install_dir: py.get_install_dir(subdir: 'sleepecg') / 'data',
)

install_data(
    'classifiers/wrn-gru-mesa-weighted.zip',
    'classifiers/wrn-gru-mesa.zip',
    'classifiers/ws-gru-mesa.zip',
    install_dir: py.get_install_dir(subdir: 'sleepecg') / 'classifiers',
)

subdir('io')
