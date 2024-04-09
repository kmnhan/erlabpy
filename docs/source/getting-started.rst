***************
Getting Started
***************

Installing
==========

The recommended way to install ERLabPy is via `conda
<https://docs.conda.io/en/latest/>`_. If you do not have conda installed, follow
the :ref:`installation instructions <Installing conda>`. Once you have a working
conda environment, you can install ERLabPy with the conda command line tool: ::

  conda install -c conda-forge erlab

.. hint::

  If you are using macOS, you might experience degraded performance with the
  default BLAS and LAPACK libraries. For Apple Silicon macs, use `Accelerate
  <https://developer.apple.com/accelerate/>`_: ::

    conda install "libblas=*=*accelerate"

  For Intel macs, use `MKL
  <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html>`_:
  ::

    conda install "libblas=*=mkl"

  To prevent conda from switching back to the default libraries, see the
  `conda-forge documentation
  <https://conda-forge.org/docs/maintainer/knowledge_base/#switching-blas-implementation>`_.

If you don’t use conda, you can install ERLabPy with pip: ::

  pip install erlab

If you wish to install ERLabPy from source, see the :doc:`contributing`.

Dependencies
============

ERLabPy is installed with many different python libraries. Some key packages and
links to their documentation are listed below as a reference. In particular,
this documentation assumes basic familiarity with the first four packages, which
will be sufficient for most use cases.

.. list-table::
    :header-rows: 1
    :stub-columns: 1
    :widths: auto

    * - Package
      - Used in
    * - `numpy <https://numpy.org/doc/stable/>`_
      - Computation and array manipulation, linear algebra
    * - `xarray <https://docs.xarray.dev/en/stable/>`_
      - Data storage and manipulation
    * - `matplotlib <https://matplotlib.org>`_
      - Plotting
    * - `scipy <https://docs.scipy.org/doc/scipy/index.html>`_
      - Linear algebra, signal processing, and image processing
    * - `lmfit <https://lmfit.github.io/lmfit-py/>`_
      - Optimization problems including curve fitting
    * - `pyqtgraph <https://pyqtgraph.readthedocs.io/en/latest/>`_
      - Interactive plotting (i.e., imagetool)

ERLabPy also requires a Qt library such as PyQt5, PyQt6, PySide2, or PySide6. To
ensure compatibility and keep the namespace clean, ERLabPy imports Qt bindings
from `qtpy <https://github.com/spyder-ide/qtpy>`_, which will automatically
select the appropriate library based on what is installed. Be aware that there
are some known compatibility issues with PyQt5 and PySide2, so it is recommended
to use the newer PyQt6 or PySide6 if possible.

See the :doc:`user-guide/index` to start using ERLabPy!


Optional dependencies
---------------------

The following packages are optional dependencies that are not installed by
default. They are used in only specific functions, or is not used at all but is
listed just for convenience.

.. list-table::
    :header-rows: 1
    :stub-columns: 1
    :widths: auto

    * - Package
      - Description
    * - `csaps <https://github.com/espdev/csaps>`_
      - Multidimensional smoothing splines
    * - `hvplot <https://github.com/holoviz/hvplot>`_ and `bokeh
        <https://github.com/bokeh/bokeh>`_
      - Interactive plotting
    * - `cmasher <https://cmasher.readthedocs.io>`_,
        `cmocean <https://matplotlib.org/cmocean/>`_, and
        `colorcet <https://colorcet.holoviz.org>`_
      - More colormaps!

For a full list of dependencies and optional dependencies, take a look at the ``[project]`` and ``[project.optional-dependencies]`` section in
`pyproject.toml <https://github.com/kmnhan/erlabpy/blob/main/pyproject.toml>`_:

.. literalinclude:: ../../pyproject.toml
   :language: toml
   :start-at: dependencies = [
   :end-before: [project.urls]
