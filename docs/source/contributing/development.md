# Development setup and workflow

Set up a local development environment, run the required checks, and submit changes.

(creating-a-development-environment)=

## Creating a development environment

Install `git` before you start.

### Installing git

See the [git installation guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
for detailed instructions.

- macOS (Intel and ARM): Install Xcode Command Line Tools from a terminal.

  ```sh
  xcode-select --install
  ```

- Windows 10 1709 (build 16299) or later: Run this command in Command Prompt or
  PowerShell.

  ```sh
  winget install --id Git.Git -e --source winget
  ```

If you are new to GitHub forks, see the [GitHub guide to contributing to
projects](https://docs.github.com/en/get-started/quickstart/contributing-to-projects).
It explains how to fork a repository, clone a fork, create a branch, push changes, and
open a pull request.

For more information, see these GitHub guides:

- [Fork a repository](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
- [Collaborate with pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests)
- [Work with forks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks)

(cloning-the-repository)=

### Cloning the repository

1. [Create a GitHub account](https://github.com/) if you do not have one.

2. Fork the [ERLabPy repository](https://github.com/kmnhan/erlabpy) with the `Fork`
   button. GitHub creates a copy under your account.

3. Clone your fork and add the main repository as `upstream`.

   ```sh
   git clone https://github.com/your-user-name/erlabpy.git
   cd erlabpy
   git remote add upstream https://github.com/kmnhan/erlabpy.git
   ```

(installing-uv)=

### Installing uv

ERLabPy uses [uv](https://docs.astral.sh/uv/) to manage the development environment.
See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)
to install it.

### Editable installation from source

An editable installation makes local source changes available without a reinstall.

1. Open a terminal at the root of the ERLabPy repository.

2. Run:

   ```sh
   uv sync --all-extras --dev --group pyqt6
   ```

   This command installs the default local test environment. It includes the primary
   Qt binding used in the fast CI workflow. To reproduce the weekly compatibility
   matrix with PySide6, also add `--group pyside6`.

### Updating the editable installation

After you update the main branch, run `uv sync` again to update an editable
installation.

(development-workflow)=

## Development workflow

Before you start, {ref}`create a local development environment
<creating-a-development-environment>`.

(update-the-main-branch)=

### Update the `main` branch

Before you start a change, update your local `main` branch from `upstream/main`. Then
create a feature branch from it.

```sh
git fetch upstream
git merge upstream/main
```

Resolve merge conflicts before you open a pull request. If you have uncommitted changes,
store them with `git stash` before the update. Restore them with `git stash apply` after
the update.

### Create a feature branch

Create a branch before you make changes. Keep `main` for production-ready code.

```sh
git switch -c shiny-new-feature
```

Keep each branch focused on one bug fix or feature. Push the branch to your GitHub fork:

```sh
git push --set-upstream origin shiny-new-feature
```

Git then records the relationship between the local branch and the branch on your fork.

### The editing workflow

1. Make changes. Follow the {ref}`code standards <code-standards>`. Follow
   {ref}`documentation` for documentation changes.

2. Inspect changed files with `git status`.

3. Inspect the changes with `git diff`.

4. Build the documentation after documentation changes. See
   {ref}`building-the-documentation-locally`.

### Running tests locally

The repository has two CI workflows:

- The fast pull-request workflow runs the full suite once with coverage on locked
  dependencies. It uses Python 3.13 and PyQt6. It runs in multiple shards.
- The weekly compatibility workflow tests upgraded dependencies with Python 3.11 to
  3.14, PyQt6, and PySide6.

For local development, run:

```sh
uv run pytest
uv run pytest -m compat
uv run python -m scripts.ci_test_groups --check-partition
```

The `compat` marker selects compatibility smoke tests for non-primary CI lanes. The
partition check verifies that the fast CI shards cover each test file once.

Shard definitions are in `scripts/_ci_test_groups.py`. If you add a top-level test
module under `tests/analysis/`, `tests/interactive/`, `tests/io/`, or `tests/`, update
that file. Assign the module to one coverage shard. Add it to compatibility smoke tests
only when it gives broad cross-version or cross-binding coverage.

`tests/conftest.py` assigns the `compat`, `gui`, and `serial` markers during collection.
Keep these rules centralized. Do not add CI-only markers to unrelated test modules.

### Commit and push your changes

Stage the files that you intend to commit. Then create a commit message that follows the
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.

```sh
git add <files>
git commit -m "type(scope): summary"
git push
```

### Open a pull request

Open a pull request when the change is ready for review or when you need feedback.
Describe the change and identify areas that need review. Use a draft pull request when
the work is not ready to merge.

(code-standards)=

## Code standards

- [Ruff](https://github.com/astral-sh/ruff) enforces import sorting, formatting, and
  linting.

- [mypy](https://mypy.readthedocs.io) performs static type checking. Add type
  annotations to new code when practical.

- [prek](https://github.com/j178/prek) is recommended. It checks code and commit
  messages before a commit. Run `prek install` at the repository root to install the
  configured hooks.

- Follow these rules for Qt code:

  - Import Qt bindings from [qtpy](https://github.com/spyder-ide/qtpy). Import only
    top-level modules.

    ```python
    from qtpy import QtWidgets, QtCore, QtGui
    ```

  - Use fully qualified Qt6 enum names. For example, use
    {obj}`QtCore.Qt.CheckState.Checked` instead of {obj}`QtCore.Qt.Checked`.

  - Use the PySide6 signal and slot syntax: `QtCore.Signal` and `QtCore.Slot`.

  - When you use Qt Designer, keep the `.ui` file beside the Python file that uses it.
    Import it with `qtpy.uic.loadUiType`.

  - For example, if `mywidget.py` and `mywidget.ui` are in
    `src/erlab/interactive/`, `mywidget.py` can contain:

    ```python
    import importlib.resources

    from qtpy import uic

    import erlab


    class MyWidget(
        *uic.loadUiType(
            str(importlib.resources.files(erlab.interactive).joinpath("mywidget.ui"))
        )
    ):
        def __init__(self):
            super().__init__()
            self.setupUi(self)
    ```

  - Start the real ImageTool manager in tests only when the test requires an active
    manager instance. For manager-aware dispatch paths, patch
    `erlab.interactive.imagetool.manager.is_running` and `show_in_manager` instead.

:::{note}

Parts of this page are based on the contributor guides for
[pandas](https://pandas.pydata.org/docs/dev/development/contributing.html) and
[xarray](https://docs.xarray.dev/en/stable/contributing.html).

:::
