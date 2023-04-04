
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

* Windows 10 1709 (build 16299) or later: type this command in command prompt or Powershell.
  
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