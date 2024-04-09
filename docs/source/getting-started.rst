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

Dependencies
============

ERLabPy is installed with many different python libraries. Some key packages and
links to their documentation are listed below as a reference. In particular,
this documentation assumes basic familiarity with the first four packages, which
are sufficient for most use cases.

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

For the full list of dependencies, see the `requirements.txt` file.

See the :doc:`userguide` to start using ERLabPy!


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
