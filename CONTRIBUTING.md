# Contributing to SleepECG
If you want to implement a new feature, fix an existing bug or help improve SleepECG in any other way (such as adding or improving documentation), please consider submitting a [pull request](https://github.com/cbrnr/sleepecg/pulls) on GitHub. It might be a good idea to open an [issue](https://github.com/cbrnr/sleepecg/issues) beforehand and discuss your planned contributions with the developers.

Before you start working on your contribution, please make sure to follow the guidelines described in this document.


## GitHub workflow
### Setup
- Create a [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) of the [GitHub repository](https://github.com/cbrnr/sleepecg).
- Clone the fork to your machine: `git clone https://github.com/<your-username>/sleepecg`.
- Make sure your [username](https://docs.github.com/en/get-started/getting-started-with-git/setting-your-username-in-git) and [email](https://docs.github.com/en/github/setting-up-and-managing-your-github-user-account/managing-email-preferences/setting-your-commit-email-address#setting-your-commit-email-address-in-git) are configured to match your GitHub account.
- Add the original repository (also called _upstream_) as a remote to your local clone: `git remote add upstream git@github.com:cbrnr/sleepecg.git`.


### Add a feature or fix a bug
- Create and switch to a new branch: `git switch -c branch-name` (`branch-name` should be representative of what you are working on).
- Make changes and commit them.
- Push the changes to your remote fork.
- Create a [pull request (PR)](https://github.com/cbrnr/sleepecg/pulls).
- Add an entry to `CHANGELOG.md` (section "UNRELEASED") where you mention the corresponding PR and (if you desire) your name.


### Rebasing
If another PR is merged while you are working on something, a merge conflict may arise. To resolve it, perform the following steps in your local clone:
- Switch to the main branch: `git switch main`.
- Pull the current changes: `git pull upstream main`.
- Switch back to the branch you are working on: `git switch branch-name`.
- Rebase your commits onto main: `git rebase main`.
- Push to your remote fork: `git push` (might require `git push --force`).


## Development enviroment
Make sure to use Python 3.7. You might want to [create a virtual environment](https://docs.python.org/3/library/venv.html#creating-virtual-environments) instead of installing everything into your main environment. In the root of the local clone of your fork, run

```
pip install -e .[dev]
```

When using the flag [`-e`](https://pip.pypa.io/en/stable/cli/pip_install/#install-editable), pip does not copy the package to `site-packges`, but creates a link to your local repository. Any changes to the source code are directly reflected in the "installed" package. Installing the optional `[dev]` dependencies makes sure all tools for style checking, testing, and building documentation locally are available.


## Code style
SleepECG adheres to [PEP 8](https://www.python.org/dev/peps/pep-0008/), with the following exceptions/specifications:
- Each source file contains a header listing the authors contributing to that module and mentions the license:
    ```python
    # Authors: Firstname Lastname
    #          Another Name
    #
    # License: BSD (3-clause)
    ```
- The maximum line length is `92` (instead of `79`).
    - If readability would suffer from a linebreak, append `# noqa` to the relevant line to disable line length checking.
- Single quotes are used, except if the string contains an apostrophe.
- [Type hints](https://www.python.org/dev/peps/pep-0484/) are encouraged.
- If a container literal, function definition or function call does not fit on one line:
    - Each argument or item is indented one level further than the function or container name.
    - The last argument or item has a trailing comma.
    - The closing parenthesis or bracket is indented at the same level as the starting line.
    ```python
    # Example
    def _download_file(
        url: str,
        target_filepath: Path,
        checksum: Optional[str] = None,
        checksum_type: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
    ```


## Public API
- Every non-public member (i.e. every member not intended to be accessed by an end user) is prefixed with an underscore: `_`.
- Inside a (sub-)package's `__init__.py`, public module members are imported explicitly.
- `__all__` is never set.
- To add a function or class to the API reference, list its _public_ name (e.g. `sleepecg.detect_heartbeats`, not `sleepecg.heartbeat_detection.detect_heartbeats`) in `doc/source/api.rst`.


## Documentation
For docstrings, SleepECG mainly follows [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html), with the following exceptions/specifications:
- The maximum line length is `75`.
- For parameters that may take multiple types, pipe characters are used instead of the word `or`, like this: `param_name : int | float`.
- For single return values, only the type is stated (no name).
- For multiple return values, both a name and a type are stated.
- Generators are treated similarly, so a type annotation `Iterator[int]` becomes just `int` in the docstring.


## pre-commit
Coding and documentation style are checked via a CI job. To make sure your contribution passes those checks, you can use [`pre-commit`](https://pre-commit.com/). To install the hooks configured in `.pre-commit-config.yml`, run

```
pre-commit install
```

inside your local clone. After that, the checks required in the CI job will be run on all staged files when you commit – and abort the commit in case any issues are found (in which case you should fix the found issues and commit again).


## Tests
SleepECG uses [`pytest`](https://docs.pytest.org/) for unit tests. The structure of `sleepecg/tests/` follows that of the package itself, e.g. the test module for `sleepecg.io.nsrr` would be `sleepecg/tests/io/test_nsrr.py`.
To run the tests, execute
```
pytest
```
in the project or package root. The tests for the C extension can be excluded using
```
pytest -m "not c_extension"
```
