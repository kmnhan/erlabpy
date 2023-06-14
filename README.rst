
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

Editable Installation from Source
=================================


1. Download `environment.yml <https://raw.githubusercontent.com/kmnhan/erlabpy/main/environment.yml>`_ to your preferred directory (:code:`my/directory`). If on Apple silicon, download `environment_apple.yml <https://raw.githubusercontent.com/kmnhan/erlabpy/main/environment_apple.yml>`_ instead.
2. Open a terminal window and go to the directory.

   .. code-block:: bash
      
      cd my/directory

3. Create a conda environment and activate it.
   Here, replace :code:`envname` with the environment name you prefer.
   Again, if on Apple silicon, replace :code:`environment.yml` with :code:`environment_apple.yml`.

   .. code-block:: bash

      conda config --prepend channels conda-forge
      conda config --set channel_priority strict
      conda env create -f environment.yml -n envname
      conda activate envname

   

4. Clone and install the repository.
   
   .. code-block:: bash

      cd src
      git clone https://github.com/kmnhan/erlabpy.git
      cd erlabpy
      pip install -e . --config-settings editable_mode=compat


**************************
Building the documentation
**************************

Install requirements
====================

.. code-block:: bash

   conda activate envname
   conda install sphinx, sphinx-autodoc-typehints, furo -y
   pip install sphinx-qt-documentation

Build
-----

.. code-block:: bash

   cd my/directory/erlabpy

.. code-block:: bash

   cd docs
   make clean
   make html && make latexpdf
   cd ..