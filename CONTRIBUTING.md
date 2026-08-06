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


## Development environment

We recommend using [uv](https://docs.astral.sh/uv/) to install and manage your Python environment. Install the tool according to the instructions on the website (there is no need to install Python separately, as uv will automatically download and install a suitable version).

In a terminal, change to the `sleepecg` folder containing your fork and run the following command:

```
uv sync --locked --all-extras --all-groups
```

This installs SleepECG in editable mode together with all development dependencies (for style checking, testing, and building documentation). Any changes to the source code are directly reflected in the installed package. You can then run a command inside this environment with `uv run <command>`, for example `uv run pytest`.


## Code style

SleepECG adheres to [PEP 8](https://www.python.org/dev/peps/pep-0008/) and [Ruff](https://astral.sh/ruff), with the following exceptions/specifications:
- Each source file contains the following header:
    ```python
    # © SleepECG developers
    #
    # License: BSD (3-clause)
    ```
- The maximum line length is `92`.
- [Type hints](https://www.python.org/dev/peps/pep-0484/) are encouraged.

Coding and documentation style are checked via a CI job using [Ruff](https://docs.astral.sh/ruff/) and [mypy](https://mypy-lang.org/). To make sure your contribution passes those checks, run the following commands inside your local clone before pushing:

```
uv run ruff check
uv run ruff format
uv run mypy src
```

`ruff check` reports linting issues (many can be fixed automatically with `uv run ruff check --fix`), and `ruff format` reformats the code in place.


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


## Tests

SleepECG uses [`pytest`](https://docs.pytest.org/) for testing. The structure of `sleepecg/tests/` follows that of the package itself, e.g. the test module for `sleepecg.io.nsrr` would be `sleepecg/tests/io/test_nsrr.py`. If a new test requires a package that is not part of the core dependencies, make sure to add it to the optional requirement categories `dev` and `cibw`.

To run the tests, execute

```
uv run pytest
```

in the project or package root. The tests for the C extension can be excluded using

```
uv run pytest -m "not c_extension"
```

## Releases

Follow these steps to make a new [PyPI](https://pypi.org/project/sleepecg/) release (requires write permissions for GitHub and PyPI project sites):

- Remove the `.dev0` suffix from the `version` field in `pyproject.toml` (and/or adapt the version to be released if necessary)
- Update the section in `CHANGELOG.md` corresponding to the new release with the version and current date
- Run `uv lock` to update the lockfile
- Commit these changes and push
- Create a new release on GitHub and use the version as the tag name (make sure to prepend the version with a `v`, e.g. `v0.7.0`)
- A GitHub Action takes care of building and uploading wheels to PyPI

This concludes the new release. Now prepare the source for the next planned release as follows:

- Update the `version` field to the next planned release and append `.dev0`
- Start a new section at the top of `CHANGELOG.md` titled `## [UNRELEASED] - YYYY-MM-DD`
- Run `uv lock` to update the lockfile

Don't forget to push these changes!


## Upgrading dependencies

To upgrade all locked dependencies to their latest allowed versions, run:

```
uv lock --upgrade
```

Commit the resulting `uv.lock` changes. This can be done at any time, independently of a release.
