******************
Contributing Guide
******************

.. note::

  Parts of this document are based on `Contributing to pandas
  <http://pandas.pydata.org/pandas-docs/stable/contributing.html>`_ and
  `Contributing to xarray
  <https://docs.xarray.dev/en/stable/contributing.html>`_.

We welcome your enthusiasm! All contributions, including bug reports, bug fixes,
documentation improvements, enhancement suggestions, and other ideas are welcome.

If you have any questions, feel free to ask us! The recommended place to ask questions
is `GitHub Discussions <https://github.com/kmnhan/erlabpy/discussions>`_.

Bug reports and enhancement requests
====================================

If you find a bug in the code or documentation, do not hesitate to submit a ticket to
the `Issue Tracker <https://github.com/kmnhan/erlabpy/issues>`_. You are also welcome to
post feature requests or pull requests.

When reporting a bug, see this `stackoverflow article for tips on writing a good bug
report <https://stackoverflow.com/help/mcve>`_, and this `article on minimal bug reports
<https://matthewrocklin.com/minimal-bug-reports>`_.

Creating a development environment
==================================

First, you will need to install `git` and `conda` (or `mamba`).

Installing git
--------------

Below are some quick instructions for installing git on various operating systems. For
more detailed instructions, see the `git installation guide
<https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_.

* macOS (Intel & ARM): get Xcode Command Line Tools by running in your terminal window:

  .. code-block:: sh

      xcode-select --install

* Windows 10 1709 (build 16299) or later: run in command prompt or PowerShell:

  .. code-block:: sh

      winget install --id Git.Git -e --source winget

If you are new to contributing to projects through forking on GitHub, take a look at the
`GitHub documentation for contributing to projects
<https://docs.github.com/en/get-started/quickstart/contributing-to-projects>`_. GitHub
provides a quick tutorial using a test repository that may help you become more familiar
with forking a repository, cloning a fork, creating a feature branch, pushing changes
and making pull requests.

Below are some useful resources for learning more about forking and pull requests on GitHub:

* the `GitHub documentation for forking a repo <https://docs.github.com/en/get-started/quickstart/fork-a-repo>`_.
* the `GitHub documentation for collaborating with pull requests <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests>`_.
* the `GitHub documentation for working with forks <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks>`_.


Cloning the repository
----------------------

1. `Create an account <https://github.com/>`_ on GitHub if you do not already have one.

2. You will need your own copy of erlabpy (aka fork) to work on the code. Go to the
   `erlabpy repository <https://github.com/kmnhan/erlabpy>`_ and hit the ``Fork`` button
   near the top of the page. This creates a copy of the code under your account on the
   GitHub server.

3. Clone your fork to your machine::

    git clone https://github.com/your-user-name/erlabpy.git
    cd erlabpy
    git remote add upstream https://github.com/kmnhan/erlabpy.git

   This creates the directory `erlabpy` and connects your repository to the upstream
   (main project) *erlabpy* repository.


.. _Installing conda:

Installing conda
----------------

Before starting any development, you'll need to create an isolated environment under a
package manager like conda. If you don't have conda installed, the recommended way is to
install `miniforge <https://github.com/conda-forge/miniforge>`_. The Scikit-HEP project
has a `great guide <https://scikit-hep.org/user/installing-conda>`_  for installing
conda.

.. hint::

  - `Mamba <https://mamba.readthedocs.io>`_ is a faster alternative to conda with
    additional features. It is installed alongside conda when you install miniforge.

Editable installation from source
---------------------------------

An editable installation allows you to make changes to the code and see the changes
reflected in the package without having to reinstall it. Before installing:

- Make sure you have `cloned the repository <#cloning-the-repository>`_.
- Make sure you have :ref:`installed conda or mamba <Installing Conda>`.
- ``cd`` to the *erlabpy* source directory.

1. Create and activate a mamba (or conda) environment.

   .. note::

     Replace :code:`<envname>`  with the environment name you prefer.

   .. hint::

     If using conda, replace :code:`mamba` with :code:`conda`.

   .. code-block:: sh

     mamba env create -f environment.yml -n <envname>
     mamba activate <envname>


2. Build and install the package.

   .. note::

      The ``editable_mode=compat`` setting enables static analysis tools to work with
      the package. See `this issue <https://github.com/pypa/setuptools/issues/3518>`_
      for more information.

   .. code-block:: sh

     pip install -e ".[dev]" --config-settings editable_mode=compat

These two steps will create the new environment, and not touch any of your existing
environments, nor any existing Python installation.

To view your environments::

      mamba env list

To return to your root environment::

      mamba deactivate

Updating the editable installation
----------------------------------

* For minor updates with editable installs, it is sufficient to just :ref:`update the
  main branch <update-main-branch>`.

* When there are changes to the dependencies, you should also update the environment:

  .. hint::

    If using conda, replace :code:`mamba` with :code:`conda`.

  .. code-block:: bash

    mamba env update -f environment.yml -n <envname>

* In case of major changes, it is recommended to rebuild the package.

  .. code-block:: bash

    mamba activate <envname>
    pip install -e . --force-reinstall --no-deps --config-settings editable_mode=compat

.. _development.workflow:

Development workflow
====================

Before starting any development, make sure you have `created a local development environment <#creating-a-development-environment>`_.

Update the ``main`` branch
--------------------------

.. _update-main-branch:

Before starting a new set of changes, fetch all changes from ``upstream/main``, and
start a new feature branch from that. From time to time you should fetch the upstream
changes from GitHub: ::

    git fetch upstream
    git merge upstream/main

This will combine your commits with the latest *erlabpy* git ``main``. If this leads to
merge conflicts, you must resolve these before submitting your pull request. Remember to
follow the commit message guidelines. If you have uncommitted changes, you will need to
``git stash`` them prior to updating. This will effectively store your changes, which
can be reapplied after updating with ``git stash apply``.


Create a new feature branch
---------------------------

Create a branch to save your changes, even before you start making changes. You want
your ``main branch`` to contain only production-ready code::

    git checkout -b shiny-new-feature

This changes your working directory to the ``shiny-new-feature`` branch.  Keep any
changes in this branch specific to one bug or feature so it is clear what the branch
brings to *erlabpy*. You can have many "shiny-new-features" and switch in between them
using the ``git checkout`` command.

Generally, you will want to keep your feature branches on your public GitHub fork of
**erlabpy**. To do this, you ``git push`` this new branch up to your GitHub repo.
Generally (if you followed the instructions in these pages, and by default), git will
have a link to your fork of the GitHub repo, called ``origin``. You push up to your own
fork with: ::

    git push origin shiny-new-feature

In git >= 1.7 you can ensure that the link is correctly set by using the
``--set-upstream`` option: ::

    git push --set-upstream origin shiny-new-feature

From now on git will know that ``shiny-new-feature`` is related to the
``shiny-new-feature branch`` in the GitHub repo.


The editing workflow
--------------------

1. Make some changes. Make sure to follow the :ref:`code standards
   <development.code-standards>` and the `documentation standards <#documentation>`_.

2. See which files have changed with ``git status``. You'll see a listing like this one: ::

    # On branch shiny-new-feature
    # Changed but not updated:
    #   (use "git add <file>..." to update what will be committed)
    #   (use "git checkout -- <file>..." to discard changes in working directory)
    #
    #  modified:   README

3. Check what the actual changes are with ``git diff``.

4. Build the documentation for documentation changes. See the `documentation section
   <#building-the-documentation-locally>`_ for more information.

Commit and push your changes
----------------------------

1. To commit all modified files into the local copy of your repo, do ``git commit -am 'A
   commit message'``. The commit message must follow the `Conventional Commits
   <https://www.conventionalcommits.org/en/v1.0.0/>`_ specification.

2. To push the changes up to your forked repo on GitHub, do a ``git push``.

Open a pull request
-------------------

When you're ready or need feedback on your code, open a Pull Request (PR) so that we can
give feedback and eventually include your suggested code into the ``main`` branch. `Pull
requests (PRs) on GitHub
<https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests>`_
are the mechanism for contributing to the code and documentation.

Enter a title for the set of changes with some explanation of what you've done. Mention
anything you'd like particular attention for - such as a complicated change or some code
you are not happy with. If you don't think your request is ready to be merged, just say
so in your pull request message and use the "Draft PR" feature of GitHub. This is a good
way of getting some preliminary code review.


Writing tests for data loader plugins
-------------------------------------

When contributing a new data loader plugin, it is important to write tests to ensure
that the plugin works as expected over time.

Since ARPES data required for testing take up a lot of space, we have a separate
repository for test data: `erlabpy-data <https://github.com/kmnhan/erlabpy-data>`_.

Suppose you are contributing a new plugin, ``src/erlab/io/plugins/<plugin_name>.py``. To
add tests, follow these steps:

1. Fork `erlabpy-data <https://github.com/kmnhan/erlabpy-data>`_ and clone it to your
   local machine.

2. Create a new directory in the root of the repository you cloned. The name of the
   directory should be the name of the plugin you are writing tests for.

3. Place the test data files into the directory you created in step 3. It's a good
   practice to also include a processed version of the data that the plugin should
   return, and use this as a reference in the tests.

4. Set the environment variable `ERLAB_TEST_DATA_DIR` to the path of the cloned
   `erlabpy-data <https://github.com/kmnhan/erlabpy-data>`_ repository in your
   development environment. This will allow the tests to access the test data.

5. Now, we can work with the original `erlabpy <https://github.com/kmnhan/erlabpy>`_
   repository to write and run tests for the plugin. Add your tests in
   `tests/io/plugins/test_<plugin_name>.py`. You can use the `test_data_dir` fixture to
   access the test data directory. See other modules in the folder for examples.

6. Run the tests using `pytest <https://docs.pytest.org/>`_ and make sure they pass.

7. Now, it's time to apply your changes. First, push your changes to `erlabpy-data
   <https://github.com/kmnhan/erlabpy-data>`_ and create a pull request.

8. Once your pull request to `erlabpy-data <https://github.com/kmnhan/erlabpy-data>`_ is
   merged, update the `DATA_COMMIT_HASH` and `DATA_KNOWN_HASH` attributes in
   `tests/conftest.py`.

   - `DATA_COMMIT_HASH` should be the commit hash of `erlabpy-data
     <https://github.com/kmnhan/erlabpy-data>`_ that contains your test data. This will
     ensure that the version of the test data used in the tests is consistent.

     .. note::

       Hitting the copy icon next to the commit hash on the `commit history
       <https://github.com/kmnhan/erlabpy-data/commits/main/>`_ page will copy the full
       hash to your clipboard.

   - `DATA_KNOWN_HASH` is the file hash of the test data tarball. This will ensure that
     the test data has not been modified or corrupted since the last time the tests were
     run.

     To calculate the hash, first download the tarball from the GitHub API::

         https://api.github.com/repos/kmnhan/erlabpy-data/tarball/<commit_hash>

     The new hash can be calculated by running the following command in the terminal:

     .. code-block:: bash

         openssl sha256 path/to/kmnhan-erlabpy-data-<commit_hash>.tar.gz

     or using `pooch <https://github.com/fatiando/pooch>`_:

     .. code-block:: python

         import pooch

         pooch.file_hash("path/to/kmnhan-erlabpy-data-<commit_hash>.tar.gz")

.. _development.code-standards:

Code standards
==============

- Import sorting, formatting, and linting are enforced with `Ruff
  <https://github.com/astral-sh/ruff>`_.

- Static type checking is performed with `mypy <https://mypy.readthedocs.io>`_. If you
  are used to working with type annotations, please try to add them to any new code you
  contribute.

- If you wish to contribute, using `pre-commit <https://pre-commit.com>`_ is
  recommended. This will ensure that your code and commit message is properly formatted
  before you commit it. A pre-commit configuration file is included in the repository,
  and you can install it by running ``pre-commit install`` in the repository root.

- When writing code that uses Qt, please adhere to the following rules:

  * Import all Qt bindings from `qtpy <https://github.com/spyder-ide/qtpy>`_, and only
    import the top level modules: ::

      from qtpy import QtWidgets, QtCore, QtGui

  * Use fully qualified enum names from Qt6 instead of the short-form enums from Qt5, i.
    e., `QtCore.Qt.CheckState.Checked` instead of `QtCore.Qt.Checked`.

  * Use the signal and slot syntax from PySide6 (``QtCore.Signal`` and ``QtCore.Slot``
    instead of ``QtCore.pyqtSignal`` and ``QtCore.pyqtSlot``).

  * When using Qt Designer, place ``.ui`` files in the same directory as the Python file
    that uses them. The files must be imported using the ``qtpy.uic.loadUiType``. ::

      from qtpy import uic

      class MyWidget(*uic.loadUiType(os.path.join(os.path.dirname(__file__), "mywidget.ui"))):
          def __init__(self):
              super().__init__()
              self.setupUi(self)

Documentation
=============

The documentation is written in **reStructuredText**, which is almost like writing in
plain English, and built using `Sphinx <http://sphinx-doc.org/>`__. The Sphinx
Documentation has an excellent `introduction to reST
<http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`__. Review the
Sphinx docs to perform more complex changes to the documentation as well.

Some other important things to know about the docs:

- The documentation consists of two parts: the docstrings in the code itself and the
  docs in ``erlabpy/docs/source/``.

  The docstrings are meant to provide a clear explanation of the usage of the individual
  functions, while the documentation in this folder consists of tutorial-like overviews
  per topic together with some other information.

- The docstrings follow the **NumPy Docstring Standard**, which is used widely in the
  Scientific Python community. This standard specifies the format of the different
  sections of the docstring. Refer to the `documentation for the Numpy docstring format
  <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_ and the
  `Sphinx examples
  <https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html>`_ for
  detailed explanation and examples, or look at some of the existing functions to extend
  it in a similar manner.

- The documentation is automatically updated by Read the Docs when a new commit is
  pushed to ``main``.

- Type annotations that follow :pep:`484` are recommended in the code, which are
  automatically included in the documentation. Hence, you may omit the type information
  from the docstring for well-annotated functions.

- We aim to follow the recommendations from the `Python documentation
  <https://devguide.python.org/documentation/start-documenting/index.html#sections>`_
  and the `Sphinx reStructuredText documentation
  <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#sections>`_
  for section markup characters:

  - ``*`` with overline, for chapters

  - ``=``, for heading

  - ``-``, for sections

  - ``~``, for subsections

  - ``**`` text ``**``, for **bold** text


Building the documentation locally
----------------------------------

Check whether all documentation dependencies are installed with

.. code-block:: sh

    pip install -r docs/requirements.txt

then build the documentation by running:

.. code-block:: sh

    cd docs/
    make clean
    make html

Then you can find the HTML output files in the folder ``erlabpy/docs/build/html/``.

To see what the documentation now looks like with your changes, you can view the HTML
build locally by opening the files in your local browser. For example, if you normally
use Google Chrome as your browser, you could enter::

    google-chrome build/html/index.html

in the terminal, running from within the ``doc/`` folder. You should now see a new tab
pop open in your local browser showing the documentation. The different pages of this
local build of the documentation are linked together, so you can browse the whole
documentation by following links the same way you would on the hosted website.
