# Contributing to SleepECG
If you want to implement a new feature, fix an existing bug, or help improve SleepECG in any other way (such as adding or improving documentation), please consider submitting a [pull request](https://github.com/cbrnr/sleepecg/pulls) on GitHub. It might be a good idea to open an [issue](https://github.com/cbrnr/sleepecg/issues) beforehand and discuss your planned contributions with the developers.

Before you start working on your contribution, please make sure to follow the guidelines described in this document.


## GitHub workflow
### Setup
- Create a [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) of the [repository](https://github.com/cbrnr/sleepecg).
- Clone the fork to your machine:
    ```
    git clone https://github.com/<your-username>/sleepecg
    ```
- Make sure your [username](https://docs.github.com/en/get-started/getting-started-with-git/setting-your-username-in-git) and [email](https://docs.github.com/en/github/setting-up-and-managing-your-github-user-account/managing-email-preferences/setting-your-commit-email-address#setting-your-commit-email-address-in-git) are configured to match your GitHub account.
- Add the original repository (also called _upstream_) as a remote to your local clone:
    ```
    git remote add upstream git@github.com:cbrnr/sleepecg.git
    ```


### Add a feature or fix a bug
- Create and switch to a new branch (use a self-explanatory branch name).
- Make changes and commit them.
- Push the changes to your remote fork.
- Create a [pull request (PR)](https://github.com/cbrnr/sleepecg/pulls).
- Add an entry to `CHANGELOG.md` (section "UNRELEASED") where you mention the corresponding PR and (if you desire) your name.


### Rebasing
If another PR is merged while you are working on something, a merge conflict may arise. To resolve it, perform the following steps in your local clone:
- Fetch the upstream changes: `git fetch upstream`
- Rebase your commits: `git rebase upstream/main`
- Resolve any merge conflicts
- Push to your remote fork: `git push` (might require `git push --force`)


## Development enviroment
Make sure to use Python 3.8. You might want to [create a virtual environment](https://docs.python.org/3/library/venv.html#creating-virtual-environments) instead of working your main environment. In the root of the local clone of your fork, install SleepECG as follows:

```
pip install -e .[dev]
```

When using the flag [`-e`](https://pip.pypa.io/en/stable/cli/pip_install/#install-editable), pip does not copy the package to `site-packages/`, but creates a link to your local repository. Any changes to the source code are directly reflected in the "installed" package. Installing the optional `[dev]` dependencies makes sure all tools for style checking, testing, and building documentation are locally available.


## Code style
SleepECG adheres to [PEP 8](https://www.python.org/dev/peps/pep-0008/) and [Black](https://black.readthedocs.io/en/stable/index.html), with the following exceptions/specifications:
- Each source file contains the following header:
    ```python
    # © SleepECG developers
    #
    # License: BSD (3-clause)
    ```
- The maximum line length is `92`.
- [Type hints](https://www.python.org/dev/peps/pep-0484/) are encouraged.


## Public API
- Every non-public member (i.e. every member not intended to be accessed by an end user) is prefixed with an underscore `_`.
- Inside a (sub-)package's `__init__.py`, public module members are imported explicitly.
- `__all__` is never set.
- To add a function or class to the API reference, list its _public_ name (e.g. `sleepecg.detect_heartbeats`, not `sleepecg.heartbeat_detection.detect_heartbeats`) in `doc/source/api.rst`.


## Documentation
For docstrings, SleepECG mainly follows [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html), with the following exceptions/specifications:
- The maximum line length is `92`.
- For parameters that may take multiple types, pipe characters are used instead of the word `or`, for example `param_name : int | float`.
- For single return values, only the type is stated (no name).
- For multiple return values, both a name and a type are stated.
- Generators are treated similarly, so a type annotation `Iterator[int]` becomes just `int` in the docstring.


## Pre-commit
Coding and documentation style are checked via a CI job. To make sure your contribution passes those checks, you can use [`pre-commit`](https://pre-commit.com/) locally. To install the hooks configured in `.pre-commit-config.yml`, run

```
pre-commit install
```

inside your local clone. After that, the checks required by the CI job will be run on all staged files when you commit – and abort the commit if any issues are found (in which case you should fix the issues and commit again).


## Tests
SleepECG uses [`pytest`](https://docs.pytest.org/) for testing. The structure of `sleepecg/test/` follows that of the package itself, e.g. the test module for `sleepecg.io.nsrr` would be `sleepecg/test/io/test_nsrr.py`. If a new test requires a package that is not part of the core dependencies (i.e. listed under `install_requires` in `setup.cfg`), make sure to add it to the optional requirement categories `dev` and `cibw`.

To run the tests, execute
```
pytest
```
in the project or package root. The tests for the C extension can be excluded using
```
pytest -m "not c_extension"
```

## Releases
Follow these steps to release a new version of SleepECG:
- In `sleepecg/__init__.py` remove the `-dev` suffix in `__version__`.
    - In case of a patch release, modify the version number accordingly.
- In `CHANGELOG.md`, update `## [UNRELEASED] - YYYY-MM-DD` to contain the version number and current date.
- Commit these changes as `Prepare vX.Y.Z release` and push.
- [Create a new release](https://github.com/cbrnr/sleepecg/releases/new) on GitHub.
    - Create a new tag where the target version is prefixed with a `v`, e.g. `v0.4.0`.
    - Use the tag as the release title.
    - Mention the most important changes in the release description and include a link to the changelog.
- This triggers the [`release.yml`](https://github.com/cbrnr/sleepecg/blob/main/.github/workflows/release.yml) workflow, which builds the wheels and publishes the package on [PyPI](https://pypi.org/project/sleepecg).

This concludes the new release. Now prepare the source for the next planned release as follows:
- Update `__version__` in `sleepecg/__init__.py` to the next planned version and append `-dev`.
- Start a new section at the top of `CHANGELOG.md` titled `## [UNRELEASED] - YYYY-MM-DD`.
- Commit these changes as `Prepare vX.Y.Z-dev` and push.
