# Contributing Guide

:::{note}

Parts of this document are based on [Contributing to
pandas](http://pandas.pydata.org/pandas-docs/stable/contributing.html) and [Contributing
to xarray](https://docs.xarray.dev/en/stable/contributing.html).

:::

We welcome your enthusiasm! All contributions, including bug reports, bug fixes,
documentation improvements, enhancement suggestions, and other ideas are welcome.

If you have any questions, feel free to ask us! The recommended place to ask questions
is [GitHub Discussions](https://github.com/kmnhan/erlabpy/discussions).

## Bug reports and enhancement requests

If you find a bug in the code or documentation, do not hesitate to submit a ticket to
the [Issue Tracker](https://github.com/kmnhan/erlabpy/issues). You are also welcome to
post feature requests or pull requests.

When reporting a bug, see this [stackoverflow article for tips on writing a good bug
report](https://stackoverflow.com/help/mcve), and this [article on minimal bug
reports](https://matthewrocklin.com/minimal-bug-reports).

(creating-a-development-environment)=

## Creating a development environment

First, you will need to install `git` if you do not already have it.

### Installing git

Below are some quick instructions for installing git on various operating systems. For
more detailed instructions, see the [git installation
guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

- macOS (Intel & ARM): get Xcode Command Line Tools by running in your terminal window:

  ```sh
  xcode-select --install
  ```

- Windows 10 1709 (build 16299) or later: run in command prompt or PowerShell:

  ```sh
  winget install --id Git.Git -e --source winget
  ```

If you are new to contributing to projects through forking on GitHub, take a look at the
[GitHub documentation for contributing to
projects](https://docs.github.com/en/get-started/quickstart/contributing-to-projects).
GitHub provides a quick tutorial using a test repository that may help you become more
familiar with forking a repository, cloning a fork, creating a feature branch, pushing
changes and making pull requests.

Below are some useful resources for learning more about forking and pull requests on
GitHub:

- the [GitHub documentation for forking a repo](https://docs.github.com/en/get-started/quickstart/fork-a-repo).
- the [GitHub documentation for collaborating with pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests).
- the [GitHub documentation for working with forks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks).

(cloning-the-repository)=

### Cloning the repository

1. [Create an account](https://github.com/) on GitHub if you do not already have one.

2. You will need your own copy of erlabpy (aka fork) to work on the code. Go to the
   [erlabpy repository](https://github.com/kmnhan/erlabpy) and hit the `Fork` button
   near the top of the page. This creates a copy of the code under your account on the
   GitHub server.

3. Clone your fork to your machine:

   ```sh
   git clone https://github.com/your-user-name/erlabpy.git
   cd erlabpy
   git remote add upstream https://github.com/kmnhan/erlabpy.git
   ```

   This creates the directory `erlabpy` and connects your repository to the upstream
   (main project) *erlabpy* repository.

(installing-uv)=

### Installing uv

If you are used to working with virtual environments and package managers, the modern
way to install the package is with [uv](https://docs.astral.sh/uv/). For installation
instructions, see the [uv
documentation](https://docs.astral.sh/uv/getting-started/installation/).

### Editable installation from source

An editable installation allows you to make changes to the code and see the changes
reflected in the package without having to reinstall it. Before installing:

- Make sure you have [cloned the repository](#cloning-the-repository).
- Make sure you have [installed uv](#installing-uv).

1. Open a terminal and navigate to the root of the *erlabpy* repository.

2. Run:

   ```sh
   uv sync --all-extras --dev
   ```

### Updating the editable installation

- For minor updates with editable installs, it is sufficient to just [update the main
  branch](#update-the-main-branch) and run `uv sync` again.

(development-workflow)=

## Development workflow

Before starting any development, make sure you have [created a local development
environment](#creating-a-development-environment).

(update-the-main-branch)=

### Update the `main` branch

Before starting a new set of changes, fetch all changes from `upstream/main`, and start
a new feature branch from that. From time to time you should fetch the upstream changes
from GitHub:

```sh
git fetch upstream
git merge upstream/main
```

This will combine your commits with the latest *erlabpy* git `main`. If this leads to
merge conflicts, you must resolve these before submitting your pull request. Remember to
follow the commit message guidelines. If you have uncommitted changes, you will need to
`git stash` them prior to updating. This will effectively store your changes, which can
be reapplied after updating with `git stash apply`.

### Create a new feature branch

Create a branch to save your changes, even before you start making changes. You want
your `main branch` to contain only production-ready code:

```sh
git checkout -b shiny-new-feature
```

This changes your working directory to the `shiny-new-feature` branch. Keep any changes
in this branch specific to one bug or feature so it is clear what the branch brings to
*erlabpy*. You can have many "shiny-new-features" and switch in between them using the
`git checkout` command.

Generally, you will want to keep your feature branches on your public GitHub fork of
**erlabpy**. To do this, you `git push` this new branch up to your GitHub repo.
Generally (if you followed the instructions in these pages, and by default), git will
have a link to your fork of the GitHub repo, called `origin`. You push up to your own
fork with:

```sh
git push origin shiny-new-feature
```

In git >= 1.7 you can ensure that the link is correctly set by using the
`--set-upstream` option:

```sh
git push --set-upstream origin shiny-new-feature
```

From now on git will know that `shiny-new-feature` is related to the `shiny-new-feature
branch` in the GitHub repo.

### The editing workflow

1. Make some changes. Make sure to follow the [code standards](#code-standards) and
   [documentation standards](#documentation) when making changes.

2. See which files have changed with `git status`. You'll see a listing like this one:

   ```sh
   # On branch shiny-new-feature
   # Changed but not updated:
   #   (use "git add <file>..." to update what will be committed)
   #   (use "git checkout -- <file>..." to discard changes in working directory)
   #
   #  modified:   README
   ```

3. Check what the actual changes are with `git diff`.

4. Build the documentation for documentation changes. See the [documentation section](#building-the-documentation-locally) for more information.

### Commit and push your changes

1. To commit all modified files into the local copy of your repo, do `git commit -am 'A
   commit message'`. The commit message must follow the [Conventional
   Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.
2. To push the changes up to your forked repo on GitHub, do a `git push`.

### Open a pull request

When you're ready or need feedback on your code, open a Pull Request (PR) so that we can
give feedback and eventually include your suggested code into the `main` branch. [Pull
requests (PRs) on
GitHub](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests)
are the mechanism for contributing to the code and documentation.

Enter a title for the set of changes with some explanation of what you've done. Mention
anything you'd like particular attention for - such as a complicated change or some code
you are not happy with. If you don't think your request is ready to be merged, just say
so in your pull request message and use the "Draft PR" feature of GitHub. This is a good
way of getting some preliminary code review.

### Writing tests for data loader plugins

When contributing a new data loader plugin, it is recommended to write tests to ensure
that the plugin always returns the expected data for future package versions.

Since ARPES data required for testing take up a lot of space, we have a separate
repository for test data: [erlabpy-data](https://github.com/kmnhan/erlabpy-data).

Suppose you are contributing a new plugin, `<plugin_name>.py`. The plugin should be
placed in the `src/erlab/io/plugins/` directory. To write tests for the plugin, follow
these steps:

1. Fork [erlabpy-data](https://github.com/kmnhan/erlabpy-data) and clone it to your
   local machine.

2. Create a new directory in the root of the repository you cloned. The name of the
   directory should be the name of the plugin you are writing tests for.

3. Place the test data files into the directory you created in step 3. It's a good
   practice to also include a processed version of the data that the plugin should
   return, and use this as a reference in the tests. See preexisting directories and
   tests for examples.

4. Set the environment variable `ERLAB_TEST_DATA_DIR` to the path of the cloned
   [erlabpy-data](https://github.com/kmnhan/erlabpy-data) repository in your development
   environment. This will allow the tests to access the test data.

5. Now, we can work with the original [erlabpy](https://github.com/kmnhan/erlabpy)
   repository to write and run tests for the plugin. Add your tests in
   `tests/io/plugins/test_<plugin_name>.py`. You can use the `test_data_dir` fixture to
   access the test data directory. See other modules in the folder for examples.

6. Run the tests on your local machine with [pytest](https://docs.pytest.org/) and make
   sure they pass:

   ```sh
   uv run pytest tests/io/plugins/test_<plugin_name>.py
   ```

7. Now, it's time to apply your changes. First, push your changes to your fork of
   [erlabpy-data](https://github.com/kmnhan/erlabpy-data) and create a pull request to
   the main repository.

8. Once your pull request to [erlabpy-data](https://github.com/kmnhan/erlabpy-data) is
   merged, update the `DATA_COMMIT_HASH` and `DATA_KNOWN_HASH` attributes in
   `tests/conftest.py`.

   - `DATA_COMMIT_HASH` should be the commit hash of
     [erlabpy-data](https://github.com/kmnhan/erlabpy-data) that contains your test
     data. This will ensure that the version of the test data used in the tests is
     consistent.

     :::{note} Hitting the copy icon next to the commit hash on the [commit
     history](https://github.com/kmnhan/erlabpy-data/commits/main/) page will copy the
     full hash to your clipboard. :::

   - `DATA_KNOWN_HASH` is the file hash of the test data tarball. This will ensure that
     the test data has not been modified or corrupted since the last time the tests were
     run.

     The hash is calculated by [this
     workflow](https://github.com/kmnhan/erlabpy-data/actions/workflows/checksum.yml)
     for each push to main. It can be copied from the workflow summary corresponding to
     the commit you wish to refer to.

9. Following the [development workflow](#development-workflow), push your changes
   including the new plugin, test files, and updated `tests/conftest.py` to your
   development branch, and create a pull request.

(code-standards)=

## Code standards

- Import sorting, formatting, and linting are enforced with
  [Ruff](https://github.com/astral-sh/ruff).

- Static type checking is performed with [mypy](https://mypy.readthedocs.io). If you are
  used to working with type annotations, please try to add them to any new code you
  contribute.

- If you wish to contribute, using [pre-commit](https://pre-commit.com) is recommended.
  This will ensure that your code and commit message is properly formatted before you
  commit it. A pre-commit configuration file is included in the repository, and you can
  install it by running `pre-commit install` in the repository root.

- When writing code that uses Qt, please adhere to the following rules:

  - Import all Qt bindings from [qtpy](https://github.com/spyder-ide/qtpy), and only
    import the top level modules:

    ```python
    from qtpy import QtWidgets, QtCore, QtGui
    ```

  - Use fully qualified enum names from Qt6 instead of the short-form enums from Qt5, i.
    e., {obj}`QtCore.Qt.CheckState.Checked` instead of {obj}`QtCore.Qt.Checked`.

  - Use the signal and slot syntax from PySide6 (`QtCore.Signal` and `QtCore.Slot`
    instead of `QtCore.pyqtSignal` and `QtCore.pyqtSlot`).

  - When using Qt Designer, place `.ui` files in the same directory as the Python file
    that uses them. The files must be imported using the `qtpy.uic.loadUiType`.

    ```python
    from qtpy import uic

    class MyWidget(*uic.loadUiType(os.path.join(os.path.dirname(__file__), "mywidget.ui"))):
        def __init__(self):
            super().__init__()
            self.setupUi(self)
    ```

(documentation)=

## Documentation

The documentation is written in **reStructuredText**, which is almost like writing in
plain English, and built using [Sphinx](http://sphinx-doc.org/). The Sphinx
Documentation has an excellent [introduction to
reST](http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html). Review
the Sphinx docs to perform more complex changes to the documentation as well.

Some other important things to know about the docs:

- The documentation consists of two parts: the docstrings in the code itself and the
  docs in `erlabpy/docs/source/`.

  The docstrings are meant to provide a clear explanation of the usage of the individual
  functions, while the documentation in this folder consists of tutorial-like overviews
  per topic together with some other information.

- The docstrings follow the **NumPy Docstring Standard**, which is used widely in the
  Scientific Python community. This standard specifies the format of the different
  sections of the docstring. Refer to the [documentation for the Numpy docstring
  format](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard) and
  the [Sphinx
  examples](https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html)
  for detailed explanation and examples, or look at some of the existing functions to
  extend it in a similar manner.

- The documentation is automatically updated by Read the Docs when a new commit is
  pushed to `main`.

- Type annotations that follow {pep}`484` are recommended in the code, which are
  automatically included in the documentation. Hence, you may omit the type information
  from the docstring for well-annotated functions.

- We aim to follow the recommendations from the [Python documentation](https://devguide.python.org/documentation/start-documenting/index.html#sections)
  and the [Sphinx reStructuredText documentation](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#sections)
  for section markup characters:

  - `*` with overline, for chapters
  - `=`, for heading
  - `-`, for sections
  - `~`, for subsections
  - `**bold**`, for **bold** text

(building-the-documentation-locally)=

### Building the documentation locally

Clone the repository and navigate to the root of the repository. Make sure you have
[installed uv](#installing-uv). Install the documentation dependencies by running:

```sh
uv sync --all-extras --dev --group docs
```

then build the documentation by running:

```sh
uv run --directory docs make html
```

Then you can find the HTML output files in the `docs/build/html/`.

To see what the documentation now looks like with your changes, you can view the HTML
build locally by opening the files in your local browser. For example, if you normally
use Google Chrome as your browser, you could enter:

```sh
google-chrome docs/build/html/index.html
```

in the terminal. You should now see a new tab pop open in your local browser showing the
documentation. The different pages of this local build of the documentation are linked
together, so you can browse the whole documentation by following links the same way you
would on the hosted website.
