===============
Getting Started
===============


Installing
==========

Follow instructions on the `GitHub repository <https://github.com/kmnhan/erlabpy>`_.

Core Dependencies
=================

ERLabPy relies on some python libraries. Links to some key libraries and their documentation are listed below, ordered by importance.

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
    * - `csaps <https://csaps.readthedocs.io/en/latest/>`_
      - Smoothing splines
    * - `joblib <https://joblib.readthedocs.io/en/stable/>`_
      - Parallel processing when numba is impractical
    * - `numba <https://numba.readthedocs.io/en/stable/index.html>`_
      - Acceleration of some performance critical code using just-in-time compilation

For the list of packages and modules provided by ERLabPy, see :doc:`reference`.