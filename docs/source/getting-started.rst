***************
Getting Started
***************

Welcome to ERLabPy! This documentation will guide you through the installation process
and provide an overview of the package.

If you are new to programming with python, `Scientific Python Lectures
<https://github.com/jrjohansson/scientific-python-lectures>`_ is a great place to start.

Data structures in ERLabPy are represented using `xarray <https://docs.xarray.dev/>`_\
:cite:p:`hoyer2017xarray`, which provides a powerful data structure for working with
multi-dimensional arrays. Check out the `xarray tutorial <https://tutorial.xarray.dev/>`_
and the `xarray user guide <https://docs.xarray.dev/en/stable/index.html>`_ to get
familiar with xarray.


.. _installing:

Installing
==========

ERLabPy depends on a number of scientific python libraries. The recommended way to
install ERLabPy is via conda. If you do not have conda installed, follow the :ref:`conda
installation instructions <Installing conda>`. Once you have a working conda
environment, you can install ERLabPy with the conda command line tool: ::

  conda install -c conda-forge erlab

Or with the recommended dependencies: ::

  conda install -c conda-forge erlab pyside6 hvplot ipywidgets

If you require other :ref:`optional dependencies`, add them to the line above.

.. hint::

  If you are using conda on macOS, you might experience degraded performance with the
  default BLAS and LAPACK libraries. For Apple Silicon macs, use `Accelerate
  <https://developer.apple.com/accelerate/>`_: ::

    conda install "libblas=*=*accelerate"

  For Intel macs, use `MKL
  <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html>`_:
  ::

    conda install "libblas=*=*mkl"

  To prevent conda from switching back to the default libraries upon updating, see the
  `conda-forge documentation
  <https://conda-forge.org/docs/maintainer/knowledge_base/#switching-blas-implementation>`_.

If you don’t use conda, you can install ERLabPy with pip: ::

  python -m pip install erlab

Optional dependency groups can be installed with the following commands: ::

  python -m pip install erlab[viz]       # Install optional dependencies for visualization
  python -m pip install erlab[io]        # Install optional dependencies for file I/O
  python -m pip install erlab[perf]      # Install optional dependencies for performance
  python -m pip install erlab[misc]      # Install miscellaneous optional dependencies
  python -m pip install erlab[complete]  # Install all optional dependencies except development dependencies

See the :ref:`optional dependencies` section for all available groups and their
contents.

If you wish to install ERLabPy from source, see the :doc:`contributing`.

Importing
=========

Once installed, you can import ERLabPy in your Python scripts or interactive sessions.

The following import conventions are recommended for ERLabPy modules: ::

  import erlab.analysis as era
  import erlab.interactive as eri
  import erlab.io
  import erlab.plotting.erplot as eplt

Along with frequently used modules, your import statements may look like this: ::

  import erlab.analysis as era
  import erlab.interactive as eri
  import erlab.io
  import erlab.plotting.erplot as eplt
  import matplotlib.pyplot as plt
  import numpy as np
  import xarray as xr

  xr.set_options(keep_attrs=True)

Of course, you may not need all of these modules for every script.

.. note::

  The interactive plotting module, :mod:`erlab.interactive`, requires a Qt library such
  as PyQt6 or PySide6. If you do not have one installed, ERLabPy will notify you when
  you try to import the module.

Dependencies
============

ERLabPy requires a number of scientific python libraries to function. The following
table lists some of the most important packages that ERLabPy depends on that are used in
various fields of the scientific python ecosystem. Links to their documentation are
listed below as a reference.

.. list-table::
    :header-rows: 1
    :stub-columns: 1
    :widths: auto

    * - Package
      - Used in
    * - `numpy <https://numpy.org/doc/stable/>`_
      - Computation and array manipulation, linear algebra
    * - `scipy <https://docs.scipy.org/doc/scipy/index.html>`_
      - Linear algebra, signal processing, and image processing
    * - `xarray <https://docs.xarray.dev/>`_
      - Data storage and manipulation
    * - `numba <https://numba.pydata.org/>`_
      - Just-in-time compilation for vast speedups
    * - `matplotlib <https://matplotlib.org>`_
      - Plotting
    * - `lmfit <https://lmfit.github.io/lmfit-py/>`_
      - Solving optimization problems and curve fitting

For interactive plotting, a Qt library such as PyQt6 or PySide6 is required. To ensure
compatibility and keep the namespace clean, ERLabPy imports Qt bindings from `qtpy
<https://github.com/spyder-ide/qtpy>`_, which will automatically select the appropriate
library based on what is installed.

See the :doc:`user-guide/index` to start using ERLabPy!

.. _optional dependencies:

Optional dependencies
---------------------

See :ref:`installing` for instructions on how to install optional dependencies with pip.

For a full list of dependencies and optional dependencies, take a look at the
``[project]`` and ``[project.optional-dependencies]`` section in `pyproject.toml
<https://github.com/kmnhan/erlabpy/blob/main/pyproject.toml>`_:

.. literalinclude:: ../../pyproject.toml
   :language: toml
   :start-at: dependencies = [
   :end-before: [project.scripts]

Notes on compatibility
----------------------

- ERLabPy supports Python 3.11 and later.
- There are some `known compatibility issues
  <https://github.com/kmnhan/erlabpy/issues/17>`_ with PyQt5 and PySide2, so it is
  recommended to use the newer PyQt6 or PySide6 if possible.
- If you meet any unexpected behaviour while using IPython's `autoreload extension
  <https://ipython.readthedocs.io/en/stable/config/extensions/autoreload.html>`_, try
  excluding the following modules: ::

    %aimport -erlab.io -erlab.accessors
