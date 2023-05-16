
#######
ERLabPy
#######

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

Python macros for ERLab.

***************
Getting Started
***************

Prerequisites
=============

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

Installation
============

1. Download `environment.yml <https://raw.githubusercontent.com/kmnhan/erlabpy/main/environment.yml>`_.
2. Open a terminal window where the file is located and run:
   
   .. code-block:: bash
      
      conda env create -f environment.yml -n envname

   Replace :code:`envname` with the environment name.

Editable Installation from Source
=================================

1. Go to your preferred directory.

   .. code-block:: bash
      
      cd my/directory

2. Clone repository.
   
   .. code-block:: bash
      mkdir src
      cd src
      git clone https://github.com/kmnhan/erlabpy.git

3. Create a conda environment and install with pip.

   .. code-block:: bash

      conda config --prepend channels conda-forge
      conda config --set channel_priority strict
      cd erlabpy
      conda env create -f environment.yml -n envname
      pip install -e . --config-settings editable_mode=compat

   Here, replace :code:`envname` with the environment name.
   
   If on Apple silicon, replace :code:`environment.yml` with :code:`environment_apple.yml`.