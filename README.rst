***************
Getting Started
***************

Installing
==========

Installation requires `git` and `conda` (or `mamba`). 

Installing Git
--------------

* macOS (Intel & ARM): get Xcode Command Line Tools by running in your terminal window: 

  .. code-block:: bash

      xcode-select --install

* Windows 10 1709 (build 16299) or later: run in command prompt or PowerShell:
  
  .. code-block:: bash

      winget install --id Git.Git -e --source winget

* Otherwise: `Install git <https://git-scm.com/downloads>`_


Installing Conda
----------------

- `Install conda
  <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_
  or `install mamba
  <https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html>`_.

  - When using conda, miniconda is recommended to save disk space.
  - `Mamba <https://mamba.readthedocs.io/en/latest/>`_ is a faster alternative
    to conda with additional features.
  - Installing `miniforge <https://github.com/conda-forge/miniforge>`_ will
    install both conda and mamba.

    - On a mac, miniforge can be installed with `homebrew <https://brew.sh>`_:

      .. code-block:: bash

        brew install miniforge


- Configure channels. 

  .. code-block:: bash

    conda config --prepend channels conda-forge
    conda config --set channel_priority strict


Editable Installation from Source
---------------------------------

1. Clone the repository to your preferred directory (:code:`my/directory`).

   .. code-block:: bash

      cd my/directory
      git clone https://github.com/kmnhan/erlabpy.git


2. Create a conda environment and activate it.
   Here, replace :code:`envname` with the environment name you prefer.
   If on Apple silicon, replace :code:`environment.yml` with :code:`environment_apple.yml`.

   .. code-block:: bash

      cd erlabpy
      conda env create -f environment.yml -n envname
      conda activate envname


3. Install the repository.
   
   .. code-block:: bash

      cd src
      git clone https://github.com/kmnhan/erlabpy.git
      cd erlabpy
      pip install -e . --config-settings editable_mode=compat


Core Dependencies
=================

ERLabPy is installed with many different python libraries. Some key packages and
links to their documentation are listed below, ordered by importance. In
particular, this documentation assumes familiarity with the first four packages,
which are sufficient for ARPES data analysis.

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