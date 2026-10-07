# AGENTS.md

SleepECG: Python package for sleep stage classification from ECG (dataset readers, heartbeat detection, feature extraction, classification). See [CONTRIBUTING.md](CONTRIBUTING.md) for the full human-oriented guide; where it disagrees with this file or with `pyproject.toml`/CI, the latter win.

## Commands

The environment is managed with [uv](https://docs.astral.sh/uv/). Run everything via `uv run`.

```
uv sync --locked --all-extras --all-groups   # dev environment (editable install, builds the C extension)
uv run pytest                                # tests (warnings are errors)
uv run pytest -m "not c_extension"           # skip tests that need the compiled extension
uv run pytest tests/test_config.py::test_x   # single test
uv run ruff check                            # lint (--fix for autofixes)
uv run ruff format                           # format (CI uses --check)
uv run ty check                              # type check (src/ only)
uv run mkdocs serve                          # preview docs
```

CI (`.github/workflows/cibuildwheel.yml`) runs `ruff check`, `ruff format --check`, `ty check`, then pytest on Python 3.11 and 3.13 plus cibuildwheel wheel builds. Run the first three before finishing a change.

## Layout

- `src/sleepecg/`: the package (src layout). `__init__.py` imports public names explicitly and re-exports `sleepecg.io` via `*`; `__all__` is never set.
    - `heartbeats.py`: `detect_heartbeats` and detector evaluation. The detector has three interchangeable backends (`c`, `numba`, `python`); the C one lives in `_heartbeat_detection.c` (typed by `_heartbeat_detection.pyi`) and numba is optional. Keep the backends' results consistent when changing the algorithm.
    - `feature_extraction.py`, `classification.py`: RRI preprocessing, features, Keras (Torch backend) classifiers, bundled pretrained models in `classifiers/*.zip`.
    - `io/`: dataset downloaders/readers (NSRR, PhysioNet, GUDB, CAP Sleep DB, ECG/sleep readers).
    - `config.py` + `config.yml`: user config in `~/.sleepecg` (data dir, classifiers dir, NSRR token).
    - `plot.py`, `utils.py`, `data/ecg.npz` (toy ECG).
- `tests/`: flat layout, one `test_<module>.py` per module (not mirrored subpackages).
- `docs/`: MkDocs Material site; `docs/api/*.md` lists public names via `::: sleepecg.<name>` (mkdocstrings). Config in `mkdocs.yml`.
- `examples/`: runnable scripts (classifier training, detector benchmark).

## C extension

- Built by `setup.py` as an abi3 (`Py_LIMITED_API`) extension targeting Python 3.11. Bumping the minimum Python version requires changing the `Py_LIMITED_API` hex in `setup.py`, the `bdist_wheel_abi3` tag, and the `build` selector in `[tool.cibuildwheel]`.
- After editing `_heartbeat_detection.c`, re-run `uv sync` (or `uv pip install -e .`) to rebuild; the `.so` in `src/sleepecg/` is gitignored.
- Anything new that must ship in the package (non-Python files) has to be listed in `[tool.setuptools.package-data]` in `pyproject.toml`.

## Conventions

- Every source file starts with:
    ```python
    # © SleepECG developers
    #
    # License: BSD (3-clause)
    ```
- Ruff with `D` (numpydoc convention), `C4`, `PERF`, `W`, `E501`. Docstring/comment line length max 88 (see `pyproject.toml`).
- Docstrings: numpydoc. Multi-type params use pipes (`x : int | float`); single return value states only the type; multiple returns state name and type.
- Type hints encouraged; `ty` checks `src/` only. Optional dependencies (`edfio`, `joblib`, `keras`, `matplotlib`, `numba`, `torch`, `wfdb`) are allowed unresolved imports and must be imported lazily inside functions, since the core install only depends on numpy, pyyaml, requests, scipy, tqdm.
- Non-public members are prefixed with `_`. A new public function must be exported in the relevant `__init__.py` and added to the matching `docs/api/*.md` page.
- New test dependencies go into the `cibw` extra (used by cibuildwheel) and, if needed, the dev groups.
- User-facing changes get an entry in `CHANGELOG.md` under `[UNRELEASED]`, formatted like the existing entries (PR link and author).

## Testing notes

- `tests/conftest.py` autouse fixture redirects the user config path to a temp file, so tests never touch `~/.sleepecg`. Keep it that way for any new config-related code.
- `pytest` runs with `filterwarnings = ["error"]` and `strict = true`: new warnings fail tests, and unregistered markers are errors.
- Tests that download from PhysioNet/NSRR honor `SLEEPECG_TEST_DATA_DIR` to reuse cached data (CI caches `.cache/sleepecg-test-data`). Avoid adding tests that require network access without that escape hatch; never use real NSRR tokens.

## Gotchas

- `uv.lock` is committed; use `uv lock` / `uv sync --locked`, and don't hand-edit it. Dependabot bumps it regularly.
- Releases are driven by the GitHub release + the CI `upload-pypi` job; see CONTRIBUTING "Releases". Don't change the version or tag unless asked.
