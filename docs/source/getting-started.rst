***************
Getting Started
***************

Installing
==========

It is recommended to use a virtual environment to avoid conflicts with other
packages. If you do not have conda installed, :ref:`install it <Installing
conda>` before following the instructions below.

1. Download `environment.yml <https://github.com/kmnhan/erlabpy/blob/main/environment.yml>`_.

2. ``cd`` to the directory containing the file.

3. Create and activate a new virtual environment:

   .. hint::

     If on Apple silicon, use `environment_apple.yml
     <https://github.com/kmnhan/erlabpy/blob/main/environment_apple.yml>`_
     instead to use BLAS and LAPACK implementations from `Accelerate
     <https://developer.apple.com/accelerate/>`_.

   .. note::

     Replace :code:`<envname>` with the environment name you prefer.

   .. code-block:: sh

     conda env create -f environment.yml -n <envname>
     conda activate <envname>

4. Install the package from PyPI: ::

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

See the :doc:`userguide` to start using ERLabPy!