=======
ERLabPy
=======


Python macros for ERLab.

===============
Getting Started
===============

Installing
==========

-------------
Prerequisites
-------------

Installation requires `git` and `conda`. 

Installing Git
--------------

* macOS (Intel & ARM): get Xcode Command Line Tools by running in your terminal window: 

  .. code-block:: bash

      xcode-select --install

* Windows 10 1709 (build 16299) or later: run in command prompt or PowerShell:
  
  .. code-block:: powershell

      winget install --id Git.Git -e --source winget

* Otherwise: `Install git <https://git-scm.com/downloads>`_


Installing Conda
----------------

* `Install conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_ 
* Miniconda is recommended to save disk space.

------------
Installation
------------

Editable Installation from Source
---------------------------------


1. Configure conda channels.

   .. code-block:: bash

      conda config --prepend channels conda-forge
      conda config --set channel_priority strict


2. Clone the repository to your preferred directory (:code:`my/directory`).

   .. code-block:: bash

      cd my/directory
      git clone https://github.com/kmnhan/erlabpy.git


3. Create a conda environment and activate it.
   Here, replace :code:`envname` with the environment name you prefer.
   If on Apple silicon, replace :code:`environment.yml` with :code:`environment_apple.yml`.

   .. code-block:: bash

      cd erlabpy
      conda env create -f environment.yml -n envname
      conda activate envname


4. Install the repository.
   
   .. code-block:: bash

      cd src
      git clone https://github.com/kmnhan/erlabpy.git
      cd erlabpy
      pip install -e . --config-settings editable_mode=compat


=================
Core Dependencies
=================

ERLabPy relies on some python libraries. Some key packages and links to their documentation are listed below, ordered by importance. In particular, this documentation assumes familiarity with the first four packages, which are sufficient for ARPES data analysis. 

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